import argparse
import json
import re
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from sentence_transformers import SentenceTransformer, util
    NLTK_AVAILABLE = True
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    nltk = None
    stopwords = None
    WordNetLemmatizer = None
    SentenceTransformer = None
    util = None
    NLTK_AVAILABLE = False
    MODEL = None

NLTK_READY = False


# --- SCORING PROFILES ---

# Profile A: Emphasizes exact keyword matches.
# Use for sections where specific terminology is critical (e.g., Required Qualifications).
KEYWORD_BIASED_PROFILE = {
    "exact_keyword": 0.75,
    "semantic": 0.10,
    "numeric": 0.10,
    "job_title": 0.05,
}

# Profile B: Emphasizes contextual understanding.
# Use for sections where concepts are more important than keywords (e.g., Responsibilities).
SEMANTIC_BIASED_PROFILE = {
    "semantic": 0.75,
    "exact_keyword": 0.10,
    "numeric": 0.10,
    "job_title": 0.05,
}

# --- FINAL SCORE WEIGHTS ---
# Defines the importance of each job description section in the final score.
FINAL_SCORE_WEIGHTS = {
    "required_qualifications": 0.30,
    "responsibilities": 0.30,
    "preferred_qualifications": 0.20,
    "skills": 0.20,
}


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


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    if NLTK_READY:
        tokens = [w for w in tokens if w not in STOP_WORDS]
        if LEMMATIZER is not None:
            try:
                tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
            except Exception:
                pass
    return " ".join(tokens)


def job_title_score(job_title: str, resume_title: str) -> float:
    """Scores the match between the job title and resume title."""
    if not job_title or not resume_title:
        return 0.0
    
    job_norm = normalize_text(job_title)
    resume_norm = normalize_text(resume_title)

    if job_norm == resume_norm:
        return 1.0
    
    # Give partial credit for overlapping words
    job_tokens = set(job_norm.split())
    resume_tokens = set(resume_norm.split())
    
    if not job_tokens:
        return 0.0
        
    intersection = job_tokens.intersection(resume_tokens)
    return len(intersection) / len(job_tokens)


def numeric_metrics_score(job_section_text: str, resume_full_text: str) -> Tuple[float, Dict[str, Any]]:
    """Extracts and compares numeric values like years of experience."""
    # Simple regex to find numbers, optionally followed by '+'
    job_numbers = re.findall(r'(\d+)\+?', job_section_text)
    resume_numbers = re.findall(r'(\d+)\+?', resume_full_text)

    job_reqs = [int(n) for n in job_numbers]
    resume_stats = [int(n) for n in resume_numbers]

    if not job_reqs:
        return 1.0, {"message": "No numeric requirements found in job description section."}

    # Find the highest numeric value in the resume (e.g., for "5+ years")
    max_resume_stat = max(resume_stats) if resume_stats else 0
    
    # Check if the resume's max value meets each requirement
    met_requirements = sum(1 for req in job_reqs if max_resume_stat >= req)
    
    score = met_requirements / len(job_reqs)
    
    return score, {"job_requirements": job_reqs, "resume_values": resume_stats, "max_resume_value": max_resume_stat}


def semantic_score(text1: str, text2: str) -> float:
    """Computes semantic similarity using a sentence-transformer model."""
    if not text1.strip() or not text2.strip() or MODEL is None:
        return 0.0
    
    # Compute embedding for both texts
    embedding1 = MODEL.encode(text1, convert_to_tensor=True)
    embedding2 = MODEL.encode(text2, convert_to_tensor=True)
    
    # Compute cosine-similarity
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item()


def keyword_score(job_section_text: str, resume_full_text: str) -> Tuple[float, List[str]]:
    """Calculates exact keyword match score."""
    job_tokens = set(normalize_text(job_section_text).split())
    resume_tokens = set(normalize_text(resume_full_text).split())
    
    if not job_tokens:
        return 0.0, []

    matched = sorted(list(job_tokens.intersection(resume_tokens)))
    score = len(matched) / len(job_tokens) if len(job_tokens) > 0 else 0.0
    return score, matched


def redistribute_weights_for_zeros(
    base_weights: Dict[str, float],
    scores: Dict[str, float],
    redistribution_ratio: float = 0.80,
) -> Tuple[Dict[str, float], float, List[str]]:
    """
    Redistribute weight from zero-score items to non-zero items.
    - `redistribution_ratio` portion is redistributed.
    - Remaining portion is treated as penalty (lost weight).
    Returns: (effective_weights, penalty_weight, zero_keys)
    """
    zero_keys = [k for k, v in scores.items() if v <= 0.0 and base_weights.get(k, 0.0) > 0.0]
    if not zero_keys:
        return dict(base_weights), 0.0, []

    non_zero_keys = [k for k, v in scores.items() if v > 0.0 and base_weights.get(k, 0.0) > 0.0]
    if not non_zero_keys:
        # Nothing to redistribute to; keep original and full penalty logic cannot be applied usefully.
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


def compute_ats_score(job_json: Dict[str, Any], resume_json: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Extract and prepare text from resume
    resume_full_text = flatten_to_text(resume_json)
    resume_title = resume_json.get("current_title", "")

    # 2. Extract text and title from job description
    job_title = job_json.get("title", "")
    jd_sections = {
        "required_qualifications": flatten_to_text(job_json.get("required_qualifications", [])),
        "responsibilities": flatten_to_text(job_json.get("responsibilities", [])),
        "preferred_qualifications": flatten_to_text(job_json.get("preferred_qualifications", [])),
        "skills": flatten_to_text(job_json.get("skills", [])),
    }

    # 3. Calculate scores for each section using the Profile-Based system
    component_scores = {}
    section_score_values: Dict[str, float] = {}
    section_penalties: Dict[str, float] = {}
    
    # --- Required Qualifications (Keyword Biased) ---
    req_qual_text = jd_sections["required_qualifications"]
    kw_score, matched_kws = keyword_score(req_qual_text, resume_full_text)
    sem_score = semantic_score(req_qual_text, resume_full_text)
    num_score, num_breakdown = numeric_metrics_score(req_qual_text, resume_full_text)
    jt_score = job_title_score(job_title, resume_title)

    req_raw_scores = {
        "exact_keyword": kw_score,
        "semantic": sem_score,
        "numeric": num_score,
        "job_title": jt_score,
    }
    req_effective_weights, req_penalty_weight, req_zero_components = redistribute_weights_for_zeros(
        KEYWORD_BIASED_PROFILE,
        req_raw_scores,
        redistribution_ratio=0.80,
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
        "penalty_weight_due_to_zeros": round(req_penalty_weight, 4),
        "matched_keywords": matched_kws,
        "numeric_details": num_breakdown,
    }

    # --- Responsibilities, Preferred Quals, Skills (Semantic Biased) ---
    for section_name in ["responsibilities", "preferred_qualifications", "skills"]:
        section_text = jd_sections[section_name]
        if not section_text.strip():
            section_score_values[section_name] = 0.0
            section_penalties[section_name] = 0.0
            component_scores[section_name] = {
                "profile_used": "Semantic-Biased",
                "final_section_score": 0.0,
                "breakdown": {
                    "semantic_similarity": 0.0,
                    "exact_keyword_score": 0.0,
                    "numeric_metrics_score": 0.0,
                    "job_title_score": 0.0,
                },
                "effective_component_weights": SEMANTIC_BIASED_PROFILE,
                "zero_components": ["semantic", "exact_keyword", "numeric", "job_title"],
                "penalty_weight_due_to_zeros": 0.0,
                "missing_in_job_description": True,
            }
            continue

        kw_score, _ = keyword_score(section_text, resume_full_text)
        sem_score = semantic_score(section_text, resume_full_text)
        num_score, num_breakdown = numeric_metrics_score(section_text, resume_full_text)
        jt_score = job_title_score(job_title, resume_title)

        raw_scores = {
            "exact_keyword": kw_score,
            "semantic": sem_score,
            "numeric": num_score,
            "job_title": jt_score,
        }
        effective_weights, penalty_weight, zero_components = redistribute_weights_for_zeros(
            SEMANTIC_BIASED_PROFILE,
            raw_scores,
            redistribution_ratio=0.80,
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
            "penalty_weight_due_to_zeros": round(penalty_weight, 4),
            "numeric_details": num_breakdown,
        }

    # 4. Redistribute top-level section weights if any section score is zero.
    effective_final_weights, final_weight_penalty, zero_sections = redistribute_weights_for_zeros(
        FINAL_SCORE_WEIGHTS,
        section_score_values,
        redistribution_ratio=0.80,
    )

    # 5. Calculate the final weighted ATS score
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
