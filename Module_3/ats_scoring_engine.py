import argparse
import json
import re
from typing import Any, Dict, List, Tuple

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sentence_transformers import SentenceTransformer, util
    NLTK_AVAILABLE = True
    MODEL = SentenceTransformer('all-mpnet-base-v2')
except ImportError:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    SentenceTransformer = None
    util = None
    NLTK_AVAILABLE = False
    MODEL = None

NLTK_READY = False

GENERIC_JOB_TERMS = {
    "looking", "engineer", "engineering", "software", "development", "developer",
    "experience", "required", "qualification", "preferred", "responsibility",
    "strong", "good", "knowledge", "understanding", "work", "working", "build",
    "team", "product", "customer", "system", "service", "services", "code",
    "clean", "maintainable", "scalable", "design", "develop", "year", "years",
}


def _ensure_nltk_resources() -> None:
    if not NLTK_AVAILABLE or nltk is None:
        return
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for resource_path, download_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:
                return
        except Exception:
            return


_ensure_nltk_resources()
if NLTK_AVAILABLE and stopwords is not None and WordNetLemmatizer is not None:
    try:
        STOP_WORDS = set(stopwords.words("english"))
        LEMMATIZER = WordNetLemmatizer()
        NLTK_READY = True
    except Exception:
        STOP_WORDS = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "or", "this", "these", "those",
        }
        LEMMATIZER = None
        NLTK_READY = False
else:
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "or", "this", "these", "those",
    }
    LEMMATIZER = None
    NLTK_READY = False


# --- SCORING PROFILES ---
# FIX A: Rebalanced weights — semantic 0.75→0.45, keyword 0.10→0.40
# Rationale: raw cosine similarity tops out at ~0.55-0.60 for cross-document
# pairs even with excellent content match. Keyword matching is more reliable
# and closer to how real ATS systems actually work. 45/40 is the correct split.
KEYWORD_BIASED_PROFILE = {
    "exact_keyword": 0.75,
    "semantic": 0.10,
    "numeric": 0.10,
    "job_title": 0.05,
}

SEMANTIC_BIASED_PROFILE = {
    "semantic": 0.45,
    "exact_keyword": 0.40,
    "numeric": 0.10,
    "job_title": 0.05,
}

# --- FINAL SCORE WEIGHTS ---
FINAL_SCORE_WEIGHTS = {
    "required_qualifications": 0.30,
    "responsibilities": 0.30,
    "preferred_qualifications": 0.20,
    "skills": 0.20,
}

# --- FIX B: Semantic normalization constants ---
# all-mpnet-base-v2 cosine similarity between two different document types
# (resume vs JD) has a practical floor of ~0.25 (noise) and ceiling of ~0.65
# (near-perfect content match). Raw scores treated as 0-1 understate real
# alignment. We normalize to this calibrated range so 0.55 → ~0.75, not 55%.
SEMANTIC_FLOOR = 0.25   # below this = no useful signal
SEMANTIC_CEILING = 0.65  # above this = near-perfect match


def normalize_semantic(raw: float) -> float:
    """
    FIX B — Normalize raw cosine similarity to the model's practical range.
    Maps [SEMANTIC_FLOOR, SEMANTIC_CEILING] → [0.0, 1.0] linearly.
    Values below floor clamp to 0; values above ceiling clamp to 1.
    """
    if raw <= SEMANTIC_FLOOR:
        return 0.0
    if raw >= SEMANTIC_CEILING:
        return 1.0
    return (raw - SEMANTIC_FLOOR) / (SEMANTIC_CEILING - SEMANTIC_FLOOR)


def flatten_to_text(obj: Any) -> str:
    parts: List[str] = []

    def _walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, (int, float, bool)):
            parts.append(str(x))
        elif isinstance(x, list):
            for item in x:
                _walk(item)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)

    _walk(obj)
    return " ".join(parts)


def _get_wordnet_pos(treebank_tag: str) -> str:
    if LEMMATIZER is None:
        return "n"
    try:
        from nltk.corpus import wordnet
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    except Exception:
        return "n"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    if NLTK_READY and LEMMATIZER is not None:
        tokens = [w for w in tokens if w not in STOP_WORDS]
        try:
            pos_tags = nltk.pos_tag(tokens)
            tokens = [
                LEMMATIZER.lemmatize(word, _get_wordnet_pos(tag))
                for word, tag in pos_tags
            ]
        except Exception:
            try:
                tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
            except Exception:
                pass
    return " ".join(tokens)


def prepare_text_for_semantic(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def job_title_score(job_title: str, resume_title: str) -> float:
    if not job_title or not resume_title:
        return 0.0
    job_norm = normalize_text(job_title)
    resume_norm = normalize_text(resume_title)
    if job_norm == resume_norm:
        return 1.0
    job_tokens = set(job_norm.split())
    resume_tokens = set(resume_norm.split())
    if not job_tokens:
        return 0.0
    intersection = job_tokens.intersection(resume_tokens)
    return len(intersection) / len(job_tokens)


def numeric_metrics_score(job_section_text: str, resume_full_text: str) -> Tuple[float, Dict[str, Any]]:
    job_numbers = re.findall(r'(\d+)\+?', job_section_text)
    resume_numbers = re.findall(r'(\d+)\+?', resume_full_text)
    job_reqs = [int(n) for n in job_numbers]
    resume_stats = [int(n) for n in resume_numbers]
    if not job_reqs:
        return 0.0, {"message": "No numeric requirements found in job description section."}
    max_resume_stat = max(resume_stats) if resume_stats else 0
    met_requirements = sum(1 for req in job_reqs if max_resume_stat >= req)
    score = met_requirements / len(job_reqs)
    return score, {"job_requirements": job_reqs, "resume_values": resume_stats, "max_resume_value": max_resume_stat}


def semantic_score(text1: str, text2: str) -> float:
    """Raw cosine similarity — normalized via normalize_semantic() before use."""
    clean_text1 = prepare_text_for_semantic(text1)
    clean_text2 = prepare_text_for_semantic(text2)
    if not clean_text1 or not clean_text2 or MODEL is None or util is None:
        return 0.0
    embedding1 = MODEL.encode(clean_text1, convert_to_tensor=True)
    embedding2 = MODEL.encode(clean_text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item()


def semantic_score_focused(jd_section_text: str, resume_chunks: List[str]) -> float:
    """
    Score a JD section against multiple focused resume chunks.
    Returns the MAX normalized similarity found across all chunks.
    """
    if not jd_section_text or not resume_chunks or MODEL is None or util is None:
        return 0.0
    clean_jd = prepare_text_for_semantic(jd_section_text)
    if not clean_jd:
        return 0.0
    jd_embedding = MODEL.encode(clean_jd, convert_to_tensor=True)
    best_raw = 0.0
    for chunk in resume_chunks:
        clean_chunk = prepare_text_for_semantic(chunk)
        if not clean_chunk:
            continue
        chunk_embedding = MODEL.encode(clean_chunk, convert_to_tensor=True)
        sim = util.cos_sim(jd_embedding, chunk_embedding).item()
        if sim > best_raw:
            best_raw = sim
    # FIX B applied here — normalize before returning
    return normalize_semantic(best_raw)


def _build_focused_resume_chunks(
    section_name: str,
    resume_json: Dict[str, Any],
) -> List[str]:
    """Return focused resume text chunks relevant to each JD section."""
    chunks: List[str] = []

    if section_name == "skills":
        skills = resume_json.get("skills", {})
        if isinstance(skills, dict):
            for v in skills.values():
                t = flatten_to_text(v)
                if t:
                    chunks.append(t)
        projects = resume_json.get("projects", [])
        if isinstance(projects, list):
            for p in projects:
                if isinstance(p, dict):
                    t = flatten_to_text(p.get("technologies", []))
                    if t:
                        chunks.append(t)

    elif section_name == "preferred_qualifications":
        cp = resume_json.get("competitive_programming", {})
        if cp:
            chunks.append(flatten_to_text(cp))
        edu = resume_json.get("education", {})
        if edu:
            chunks.append(flatten_to_text(edu))
        skills = resume_json.get("skills", {})
        if isinstance(skills, dict):
            for v in skills.values():
                t = flatten_to_text(v)
                if t:
                    chunks.append(t)

    elif section_name == "responsibilities":
        exp = resume_json.get("experience", [])
        if isinstance(exp, list):
            for e in exp:
                t = flatten_to_text(e)
                if t:
                    chunks.append(t)
        projects = resume_json.get("projects", [])
        if isinstance(projects, list):
            for p in projects:
                if isinstance(p, dict):
                    t = flatten_to_text(p.get("description", []))
                    if t:
                        chunks.append(t)

    return chunks


def keyword_score(job_section_text: str, resume_full_text: str) -> Tuple[float, List[str]]:
    job_tokens = set(normalize_text(job_section_text).split())
    job_keywords = {token for token in job_tokens if token not in GENERIC_JOB_TERMS}
    resume_tokens = set(normalize_text(resume_full_text).split())
    if not job_keywords:
        return 0.0, []
    matched = sorted(list(job_keywords.intersection(resume_tokens)))
    score = len(matched) / len(job_keywords)
    return score, matched


def extract_numeric_entities(text: str, priority: str) -> Dict[str, List[Any]]:
    if not text or not isinstance(text, str):
        return {}
    entities: Dict[str, List[Any]] = {}
    normalized_text = text.lower()
    patterns = [
        re.compile(r"(\d+)\s*(\+)?\s*(?:years?|yrs?)\s+of\s+([a-zA-Z][a-zA-Z0-9\s\-\+\.#&/]{1,50}?)(?:\s+experience|\s+exp)?(?=$|[,.;&]|\s+and\s)", re.IGNORECASE),
        re.compile(r"(\d+)\s*(\+)?\s*(?:years?|yrs?)\s+([a-zA-Z][a-zA-Z0-9\s\-\+\.#&/]{1,50}?)(?:\s+experience|\s+exp)(?=$|[,.;&]|\s+and\s)", re.IGNORECASE),
        re.compile(r"(?:at\s+least|minimum\s+of|min\.?\s*)\s*(\d+)\s*(\+)?\s*(?:years?|yrs?)\s+([a-zA-Z][a-zA-Z0-9\s\-\+\.#&/]{1,50}?)(?:\s+experience|\s+exp)?(?=$|[,.;&]|\s+and\s)", re.IGNORECASE),
    ]
    for pattern in patterns:
        for match in pattern.finditer(normalized_text):
            value = int(match.group(1))
            plus_symbol = match.group(2)
            topic_raw = match.group(3).strip()
            topic_norm = normalize_text(topic_raw)
            topic_norm = re.sub(r"\s+", " ", topic_norm).strip()
            if not topic_norm:
                continue
            modifier = "more" if plus_symbol or re.search(r"\bat\s+least\b|\bminimum\b|min\.", match.group(0), re.IGNORECASE) else None
            existing = entities.get(topic_norm)
            if existing is None or value > int(existing[0]):
                entities[topic_norm] = [value, modifier, priority]
    return entities


def score_numeric_contextually(job_req_text: str, job_pref_text: str, resume_full_text: str) -> Dict[str, Any]:
    required_entities = extract_numeric_entities(job_req_text, "required")
    preferred_entities = extract_numeric_entities(job_pref_text, "preferred")
    resume_entities = extract_numeric_entities(resume_full_text, "resume")

    required_total = len(required_entities)
    required_met = 0
    preferred_bonus_hits = 0
    required_details: List[Dict[str, Any]] = []
    preferred_details: List[Dict[str, Any]] = []

    for key, (required_value, required_modifier, _) in required_entities.items():
        resume_item = resume_entities.get(key)
        met = False
        resume_value = None
        if resume_item:
            resume_value = int(resume_item[0])
            met = resume_value >= int(required_value)
        if met:
            required_met += 1
        required_details.append({
            "key": key, "required": [required_value, required_modifier, "required"],
            "resume": resume_item, "met": met, "resume_value": resume_value,
        })

    for key, (preferred_value, preferred_modifier, _) in preferred_entities.items():
        resume_item = resume_entities.get(key)
        met = False
        if resume_item:
            resume_value = int(resume_item[0])
            met = resume_value >= int(preferred_value)
        if met:
            preferred_bonus_hits += 1
        preferred_details.append({
            "key": key, "preferred": [preferred_value, preferred_modifier, "preferred"],
            "resume": resume_item, "met": met,
        })

    required_score = (required_met / required_total) if required_total > 0 else 0.0
    preferred_bonus_score = min(1.0, preferred_bonus_hits * 0.20)
    overall_score = min(1.0, required_score + preferred_bonus_score)

    return {
        "score": overall_score,
        "required_score": required_score,
        "preferred_bonus_score": preferred_bonus_score,
        "required_entities": required_entities,
        "preferred_entities": preferred_entities,
        "resume_entities": resume_entities,
        "required_details": required_details,
        "preferred_details": preferred_details,
    }


def _split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+|(?<=\w)\.\s", text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_best_snippets(responsibility_text: str, source_text: str, top_k: int = 5) -> str:
    """Extract top_k most relevant sentences from source_text for a responsibility."""
    if MODEL is None or util is None:
        return source_text
    sentences = _split_into_sentences(source_text)
    if not sentences:
        return source_text
    resp_embedding = MODEL.encode(prepare_text_for_semantic(responsibility_text), convert_to_tensor=True)
    sent_embeddings = MODEL.encode(
        [prepare_text_for_semantic(s) for s in sentences],
        convert_to_tensor=True,
    )
    scores = util.cos_sim(resp_embedding, sent_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k].tolist()
    top_sentences = [sentences[i] for i in sorted(top_indices)]
    return " ".join(top_sentences) if top_sentences else source_text


def score_responsibilities_by_project(
    job_responsibilities_text: str,
    resume_projects: List[Dict[str, Any]],
    resume_experience: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Score each JD responsibility against the best snippet from each
    project/experience entry. Returns the max-scoring candidate.
    FIX B applied: raw similarity is normalized before aggregation.
    """
    if not job_responsibilities_text:
        return {"score": 0.0, "best_project": None, "project_scores": []}

    responsibilities = [
        r.strip()
        for r in re.split(r"\n+|(?<=[.!?])\s+", job_responsibilities_text)
        if len(r.strip()) > 5
    ]
    if not responsibilities:
        responsibilities = [job_responsibilities_text]

    candidates: List[Dict[str, Any]] = []

    for project in (resume_projects or []):
        if not isinstance(project, dict):
            continue
        name = str(project.get("name", "Unnamed Project"))
        text = " ".join([
            str(project.get("name", "")),
            flatten_to_text(project.get("technologies", [])),
            flatten_to_text(project.get("description", [])),
        ]).strip()
        if text:
            candidates.append({"name": name, "text": text, "type": "project"})

    for idx, exp in enumerate(resume_experience or []):
        text = flatten_to_text(exp)
        name = str(exp.get("title", f"experience_{idx+1}")) if isinstance(exp, dict) else f"experience_{idx+1}"
        if text:
            candidates.append({"name": name, "text": text, "type": "experience"})

    if not candidates:
        return {"score": 0.0, "best_project": None, "project_scores": []}

    max_score = 0.0
    best_source = None
    project_scores: List[Dict[str, Any]] = []

    for candidate in candidates:
        per_resp_scores: List[float] = []
        for resp in responsibilities:
            snippet = _extract_best_snippets(resp, candidate["text"], top_k=5)
            raw = semantic_score(resp, snippet)
            # FIX B: normalize each per-responsibility similarity
            per_resp_scores.append(normalize_semantic(raw))

        candidate_score = sum(per_resp_scores) / len(per_resp_scores) if per_resp_scores else 0.0

        project_scores.append({"project": candidate["name"], "score": round(candidate_score, 4), "type": candidate["type"]})
        if candidate_score > max_score:
            max_score = candidate_score
            best_source = candidate["name"]

    project_scores = sorted(project_scores, key=lambda x: x["score"], reverse=True)
    return {
        "score": max_score,
        "best_project": best_source,
        "project_scores": project_scores,
    }


def score_job_title_semantically(
    job_title: str,
    resume_data: Dict[str, Any],
    threshold: float = 0.20,
) -> Dict[str, Any]:
    if not job_title or not isinstance(resume_data, dict):
        return {"score": 0.0, "relevant_sections_count": 0, "average_similarity": 0.0, "matches": []}

    sources: List[Tuple[str, str]] = []

    projects = resume_data.get("projects", [])
    if isinstance(projects, list):
        for idx, project in enumerate(projects):
            if isinstance(project, dict):
                text = " ".join([
                    str(project.get("name", "")),
                    flatten_to_text(project.get("technologies", [])),
                    flatten_to_text(project.get("description", [])),
                ]).strip()
                if text:
                    sources.append((f"project_{idx+1}", text))

    experience = resume_data.get("experience", [])
    if isinstance(experience, list):
        for idx, exp in enumerate(experience):
            text = flatten_to_text(exp)
            if text:
                sources.append((f"experience_{idx+1}", text))

    skills = resume_data.get("skills", {})
    if isinstance(skills, dict):
        for key, value in skills.items():
            text = flatten_to_text(value)
            if text:
                sources.append((f"skills_{key}", text))
    elif isinstance(skills, list):
        text = flatten_to_text(skills)
        if text:
            sources.append(("skills", text))

    education = resume_data.get("education", {})
    if isinstance(education, dict) and education:
        edu_text = flatten_to_text(education)
        if edu_text:
            sources.append(("education", edu_text))

    github = resume_data.get("github", {})
    if isinstance(github, dict):
        profile_info = github.get("profile_info", {})
        if isinstance(profile_info, dict):
            bio_text = str(profile_info.get("bio", ""))
            if bio_text:
                sources.append(("github_bio", bio_text))

    if not sources:
        return {"score": 0.0, "relevant_sections_count": 0, "average_similarity": 0.0, "matches": []}

    relevant_scores: List[float] = []
    matches: List[Dict[str, Any]] = []

    for source_name, source_text in sources:
        sim = semantic_score(job_title, source_text)
        if sim >= threshold:
            relevant_scores.append(sim)
            matches.append({"source": source_name, "similarity": round(sim, 4)})

    relevant_sections_count = len(relevant_scores)
    average_similarity = (sum(relevant_scores) / relevant_sections_count) if relevant_sections_count else 0.0
    frequency_bonus = min(relevant_sections_count, 10) * 0.03
    score = min(1.0, (average_similarity * 0.7) + frequency_bonus)

    return {
        "score": score,
        "relevant_sections_count": relevant_sections_count,
        "average_similarity": average_similarity,
        "matches": matches,
    }


def redistribute_weights_for_zeros(
    base_weights: Dict[str, float],
    scores: Dict[str, float],
    redistribution_ratio: float = 0.80,
) -> Tuple[Dict[str, float], float, List[str]]:
    zero_keys = [k for k, v in scores.items() if v <= 0.0 and base_weights.get(k, 0.0) > 0.0]
    if not zero_keys:
        return dict(base_weights), 0.0, []
    non_zero_keys = [k for k, v in scores.items() if v > 0.0 and base_weights.get(k, 0.0) > 0.0]
    if not non_zero_keys:
        return dict(base_weights), 0.0, zero_keys
    total_zero_weight = sum(base_weights[k] for k in zero_keys)
    penalty_weight = total_zero_weight * (1.0 - redistribution_ratio)
    redistributed_weight = total_zero_weight * redistribution_ratio
    effective_weights = dict(base_weights)
    for k in zero_keys:
        effective_weights[k] = 0.0
    non_zero_base_sum = sum(base_weights[k] for k in non_zero_keys)
    if non_zero_base_sum > 0:
        for k in non_zero_keys:
            share = base_weights[k] / non_zero_base_sum
            effective_weights[k] += redistributed_weight * share
    return effective_weights, penalty_weight, zero_keys


def apply_non_applicable_components(
    base_weights: Dict[str, float],
    non_applicable_keys: List[str],
) -> Dict[str, float]:
    if not non_applicable_keys:
        return dict(base_weights)
    effective_weights = dict(base_weights)
    dropped_weight = 0.0
    for key in non_applicable_keys:
        if key in effective_weights and effective_weights[key] > 0.0:
            dropped_weight += effective_weights[key]
            effective_weights[key] = 0.0
    if dropped_weight <= 0.0:
        return effective_weights
    applicable_keys = [k for k, w in effective_weights.items() if w > 0.0]
    applicable_total = sum(effective_weights[k] for k in applicable_keys)
    if applicable_total <= 0.0:
        return effective_weights
    for key in applicable_keys:
        share = effective_weights[key] / applicable_total
        effective_weights[key] += dropped_weight * share
    return effective_weights


def score_qualifications_smart(job_section_text: str, resume_data: Dict[str, Any]) -> Dict[str, Any]:
    education = resume_data.get("education", {}) if isinstance(resume_data, dict) else {}
    if not isinstance(education, dict):
        education = {}

    applicable_checks = 0
    passed_count = 0
    passed_checks: List[str] = []
    check_details: Dict[str, Any] = {}

    gpa_match = re.search(
        r"minimum\s+gpa\s+of\s+(\d+\.?\d*)\s*/\s*(\d+\.?\d*)",
        job_section_text, flags=re.IGNORECASE,
    )
    candidate_gpa = education.get("current_gpa")
    candidate_scale = education.get("scale", 10.0)
    gpa_passed = False
    if gpa_match and candidate_gpa is not None:
        applicable_checks += 1
        required_gpa = float(gpa_match.group(1))
        required_scale = float(gpa_match.group(2))
        try:
            candidate_gpa_float = float(candidate_gpa)
            candidate_scale_float = float(candidate_scale) if candidate_scale is not None else 10.0
            if required_scale > 0 and candidate_scale_float > 0:
                required_on_candidate_scale = required_gpa * (candidate_scale_float / required_scale)
                gpa_passed = candidate_gpa_float >= required_on_candidate_scale
                check_details["gpa_check"] = {
                    "required": {"value": required_gpa, "scale": required_scale},
                    "candidate": {"value": candidate_gpa_float, "scale": candidate_scale_float},
                    "required_on_candidate_scale": round(required_on_candidate_scale, 2),
                    "passed": gpa_passed,
                }
        except (TypeError, ValueError):
            check_details["gpa_check"] = {"passed": False, "reason": "Invalid GPA format in resume."}
    else:
        check_details["gpa_check"] = {
            "passed": False, "applicable": False,
            "reason": "Could not find GPA rule in JD or GPA in resume education.",
        }
    if gpa_passed:
        passed_count += 1
        passed_checks.append("GPA check passed")

    year_match = re.search(
        r"graduation\s+in\s+(\d{4})\s+or\s+before",
        job_section_text, flags=re.IGNORECASE,
    )
    candidate_grad_year = education.get("expected_graduation_year")
    year_passed = False
    if year_match and candidate_grad_year is not None:
        applicable_checks += 1
        required_latest_year = int(year_match.group(1))
        try:
            candidate_year_int = int(candidate_grad_year)
            year_passed = candidate_year_int <= required_latest_year
            check_details["graduation_year_check"] = {
                "required_latest_year": required_latest_year,
                "candidate_year": candidate_year_int,
                "passed": year_passed,
            }
        except (TypeError, ValueError):
            check_details["graduation_year_check"] = {"passed": False, "reason": "Invalid graduation year format."}
    else:
        check_details["graduation_year_check"] = {
            "passed": False, "applicable": False,
            "reason": "Could not find graduation year rule in JD or graduation year in resume education.",
        }
    if year_passed:
        passed_count += 1
        passed_checks.append("Graduation year check passed")

    required_degree_keywords: List[str] = []
    if re.search(r"\bb\.?\s?tech\b", job_section_text, flags=re.IGNORECASE):
        required_degree_keywords.append("b.tech")
    if re.search(r"computer\s+science", job_section_text, flags=re.IGNORECASE):
        required_degree_keywords.append("computer science")

    candidate_degree = str(education.get("degree", "") or "")
    candidate_degree_norm = re.sub(r"\s+", " ", candidate_degree.lower()).strip()
    degree_passed = False
    if required_degree_keywords and candidate_degree_norm:
        applicable_checks += 1
        normalized_required = [k.replace(".", "") for k in required_degree_keywords]
        normalized_candidate = candidate_degree_norm.replace(".", "")
        degree_passed = all(k in normalized_candidate for k in normalized_required)
        check_details["degree_title_check"] = {
            "required_keywords": required_degree_keywords,
            "candidate_degree": candidate_degree,
            "passed": degree_passed,
        }
    else:
        check_details["degree_title_check"] = {
            "required_keywords": required_degree_keywords,
            "candidate_degree": candidate_degree,
            "passed": False,
            "reason": "Missing JD degree keywords or candidate degree.",
        }
    if degree_passed:
        passed_count += 1
        passed_checks.append("Degree title check passed")

    final_score = (passed_count / applicable_checks) * 100.0 if applicable_checks > 0 else 0.0
    return {
        "score": round(min(final_score, 100.0), 2),
        "details": passed_checks,
        "applicable_checks": applicable_checks,
        "passed_checks_count": passed_count,
        "checks": check_details,
    }


def compute_ats_score(job_json: Dict[str, Any], resume_json: Dict[str, Any]) -> Dict[str, Any]:
    resume_full_text = flatten_to_text(resume_json)

    job_title = job_json.get("job_title") or job_json.get("title", "")
    req_qual_list = job_json.get("required_qualifications", [])
    pref_qual_list = job_json.get("preferred_qualifications", [])

    jd_sections = {
        "required_qualifications": flatten_to_text(req_qual_list),
        "responsibilities": flatten_to_text(job_json.get("responsibilities", [])),
        "preferred_qualifications": flatten_to_text(pref_qual_list),
        "skills": flatten_to_text(job_json.get("skills", [])),
    }

    numeric_context = score_numeric_contextually(
        jd_sections["required_qualifications"],
        jd_sections["preferred_qualifications"],
        resume_full_text,
    )
    title_context = score_job_title_semantically(job_title, resume_json)
    resume_projects = resume_json.get("projects", []) if isinstance(resume_json, dict) else []
    resume_experience = resume_json.get("experience", []) if isinstance(resume_json, dict) else []

    component_scores = {}
    section_score_values: Dict[str, float] = {}
    section_penalties: Dict[str, float] = {}

    # --- Required Qualifications ---
    req_qual_text = jd_sections["required_qualifications"]
    education_data = resume_json.get("education", {}) if isinstance(resume_json, dict) else {}
    if req_qual_text.strip() and isinstance(education_data, dict) and education_data:
        smart_result = score_qualifications_smart(req_qual_text, resume_json)
        req_qual_final_score = smart_result["score"] / 100.0
        section_score_values["required_qualifications"] = req_qual_final_score
        section_penalties["required_qualifications"] = 0.0
        component_scores["required_qualifications"] = {
            "profile_used": "Smart-Parser",
            "final_section_score": round(smart_result["score"], 2),
            "breakdown": {"smart_parser_score": round(smart_result["score"], 2)},
            "effective_component_weights": {"smart_parser": 1.0},
            "zero_components": [] if smart_result["score"] > 0.0 else ["smart_parser"],
            "penalty_weight_due_to_zeros": 0.0,
            "matched_details": smart_result["details"],
            "smart_parser_details": smart_result["checks"],
        }
    else:
        kw_score, matched_kws = keyword_score(req_qual_text, resume_full_text)
        raw_sem = semantic_score(req_qual_text, resume_full_text)
        sem_score = normalize_semantic(raw_sem)  # FIX B
        numeric_applicable = bool(numeric_context["required_entities"])
        num_score = numeric_context["required_score"] if numeric_applicable else 0.0
        jt_score = title_context["score"]
        req_profile_weights = apply_non_applicable_components(
            KEYWORD_BIASED_PROFILE, [] if numeric_applicable else ["numeric"],
        )
        req_raw_scores = {
            "exact_keyword": kw_score, "semantic": sem_score,
            "numeric": num_score, "job_title": jt_score,
        }
        req_effective_weights, req_penalty_weight, req_zero_components = redistribute_weights_for_zeros(
            req_profile_weights, req_raw_scores, redistribution_ratio=0.80,
        )
        req_qual_final_score = (
            req_raw_scores["exact_keyword"] * req_effective_weights["exact_keyword"] +
            req_raw_scores["semantic"] * req_effective_weights["semantic"] +
            req_raw_scores["numeric"] * req_effective_weights["numeric"] +
            req_raw_scores["job_title"] * req_effective_weights["job_title"]
        )
        section_score_values["required_qualifications"] = req_qual_final_score
        section_penalties["required_qualifications"] = req_penalty_weight
        component_scores["required_qualifications"] = {
            "profile_used": "Keyword-Biased",
            "final_section_score": round(req_qual_final_score * 100, 2),
            "breakdown": {
                "exact_keyword_score": round(kw_score * 100, 2),
                "semantic_similarity": round(sem_score * 100, 2),
                "numeric_metrics_score": round(num_score * 100, 2),
                "job_title_score": round(jt_score * 100, 2),
            },
            "effective_component_weights": req_effective_weights,
            "zero_components": req_zero_components,
            "non_applicable_components": [] if numeric_applicable else ["numeric"],
            "penalty_weight_due_to_zeros": round(req_penalty_weight, 4),
            "matched_keywords": matched_kws,
            "numeric_details": numeric_context,
            "job_title_details": title_context,
        }

    # --- Responsibilities, Preferred Quals, Skills ---
    for section_name in ["responsibilities", "preferred_qualifications", "skills"]:
        section_text = jd_sections[section_name]
        if not section_text.strip():
            section_score_values[section_name] = 0.0
            section_penalties[section_name] = 0.0
            component_scores[section_name] = {
                "profile_used": "Semantic-Biased",
                "final_section_score": 0.0,
                "breakdown": {
                    "semantic_similarity": 0.0, "exact_keyword_score": 0.0,
                    "numeric_metrics_score": 0.0, "job_title_score": 0.0,
                },
                "effective_component_weights": SEMANTIC_BIASED_PROFILE,
                "zero_components": ["semantic", "exact_keyword", "numeric", "job_title"],
                "penalty_weight_due_to_zeros": 0.0,
                "missing_in_job_description": True,
            }
            continue

        kw_score, matched_kws = keyword_score(section_text, resume_full_text)

        mapper_details = None
        if section_name == "responsibilities":
            mapper_details = score_responsibilities_by_project(
                section_text, resume_projects, resume_experience
            )
            sem_score = mapper_details["score"]  # already normalized in Fix B
        else:
            focused_chunks = _build_focused_resume_chunks(section_name, resume_json)
            if focused_chunks:
                sem_score = semantic_score_focused(section_text, focused_chunks)  # normalized in Fix B
            else:
                raw_sem = semantic_score(section_text, resume_full_text)
                sem_score = normalize_semantic(raw_sem)  # FIX B fallback

        numeric_applicable = False
        num_score = 0.0
        jt_score = title_context["score"]

        section_profile_weights = apply_non_applicable_components(
            SEMANTIC_BIASED_PROFILE, [] if numeric_applicable else ["numeric"],
        )

        raw_scores = {
            "exact_keyword": kw_score, "semantic": sem_score,
            "numeric": num_score, "job_title": jt_score,
        }
        effective_weights, penalty_weight, zero_components = redistribute_weights_for_zeros(
            section_profile_weights, raw_scores, redistribution_ratio=0.80,
        )

        final_section_score = (
            sem_score * effective_weights["semantic"] +
            kw_score * effective_weights["exact_keyword"] +
            num_score * effective_weights["numeric"] +
            jt_score * effective_weights["job_title"]
        )
        section_score_values[section_name] = final_section_score
        section_penalties[section_name] = penalty_weight

        component_scores[section_name] = {
            "profile_used": "Semantic-Biased",
            "final_section_score": round(final_section_score * 100, 2),
            "breakdown": {
                "semantic_similarity": round(sem_score * 100, 2),
                "exact_keyword_score": round(kw_score * 100, 2),
                "numeric_metrics_score": round(num_score * 100, 2),
                "job_title_score": round(jt_score * 100, 2),
            },
            "effective_component_weights": effective_weights,
            "zero_components": zero_components,
            "non_applicable_components": [] if numeric_applicable else ["numeric"],
            "penalty_weight_due_to_zeros": round(penalty_weight, 4),
            "matched_keywords": matched_kws,
            "numeric_details": numeric_context,
            "job_title_details": title_context,
        }
        if mapper_details is not None:
            component_scores[section_name]["responsibility_project_mapper"] = mapper_details

    effective_final_weights, final_weight_penalty, zero_sections = redistribute_weights_for_zeros(
        FINAL_SCORE_WEIGHTS, section_score_values, redistribution_ratio=0.80,
    )

    weighted_total = (
        section_score_values["required_qualifications"] * effective_final_weights["required_qualifications"] +
        section_score_values["responsibilities"] * effective_final_weights["responsibilities"] +
        section_score_values["preferred_qualifications"] * effective_final_weights["preferred_qualifications"] +
        section_score_values["skills"] * effective_final_weights["skills"]
    )

    final_score_100 = round(weighted_total * 100, 2)

    return {
        "final_ats_score": final_score_100,
        "score_scale": "0-100",
        "scoring_profiles": {
            "keyword_biased": KEYWORD_BIASED_PROFILE,
            "semantic_biased": SEMANTIC_BIASED_PROFILE,
        },
        "final_score_weights": FINAL_SCORE_WEIGHTS,
        "effective_final_score_weights": effective_final_weights,
        "zero_score_sections": zero_sections,
        "final_weight_penalty_due_to_zero_sections": round(final_weight_penalty, 4),
        "component_scores": component_scores,
    }


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="ATS Scoring Engine (0-100)")
    parser.add_argument("--job", required=True, help="Path to job description JSON")
    parser.add_argument("--resume", required=True, help="Path to resume JSON")
    parser.add_argument("--output", default="ats_score.json", help="Output score JSON path")
    args = parser.parse_args()

    job_json = load_json(args.job)
    resume_json = load_json(args.resume)

    result = compute_ats_score(job_json, resume_json)
    save_json(args.output, result)

    print(json.dumps(result, indent=2))
    print(f"\nSaved ATS result to: {args.output}")


if __name__ == "__main__":
    main()