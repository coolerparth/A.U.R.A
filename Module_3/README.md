# Resume Scoring Engine

A Python project that:
1. Builds a structured `resume.json` from your base profile + coding-platform data.
2. Scores the resume against a job description using a hybrid ATS model.

---

## Current Project Scope

This repository contains two workflows:

- **Resume Data Pipeline** (`pipeline.py`)
  - Reads `individual1.json`
  - Extracts usernames from profile links
  - Pulls data from GitHub, LeetCode, Codeforces, CodeChef
  - Merges everything into `resume.json`

- **ATS Scoring Engine** (`ats_scoring_engine.py`)
  - Reads `job_description.json` and `resume.json`
  - Computes section-wise scores with weighted profiles
  - Produces `ats_score.json`

---

## Repository Structure

```text
Resume_Scoring_Engine/
├── pipeline.py
├── ats_scoring_engine.py
├── download_model.py
├── requirements.txt
├── README.md
│
├── individual1.json
├── job_description.json
├── resume.json
├── ats_score.json
│
└── extraction/
    ├── extract_links.py
    ├── extract_all.py
    ├── merge_data.py
    ├── github/
    ├── leetcode/
    ├── codeforces/
    └── codechef/
```

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional but recommended (pre-download semantic model):

```bash
python download_model.py
```

---

## 1) Resume Data Pipeline

### Input
Edit `individual1.json`.

### Run
```bash
python pipeline.py
```

### Custom paths
```bash
python pipeline.py --input my_input.json --output my_resume.json
```

### Behavior notes
- Missing platform links are safely skipped.
- Output keeps stable keys for platforms.
- Pipeline currently uses GitHub, LeetCode, Codeforces, and CodeChef extractors.

---

## 2) ATS Scoring Engine

### Run
```bash
python ats_scoring_engine.py --job job_description.json --resume resume.json --output ats_score.json
```

### Inputs
- `job_description.json`
- `resume.json`

### Output
- `ats_score.json`

---

## ATS Scoring Design (Current)

Final score combines 4 JD sections:

- `required_qualifications`: **30%**
- `responsibilities`: **30%**
- `preferred_qualifications`: **20%**
- `skills`: **20%**

### Profile-based sub-scoring

Each section uses one profile:

#### Keyword-Biased profile
(used for `required_qualifications`)
- `exact_keyword`: 0.75
- `semantic`: 0.10
- `numeric`: 0.10
- `job_title`: 0.05

#### Semantic-Biased profile
(used for `responsibilities`, `preferred_qualifications`, `skills`)
- `semantic`: 0.75
- `exact_keyword`: 0.10
- `numeric`: 0.10
- `job_title`: 0.05

---

## Zero-Score Redistribution Rule (Current)

If any score is zero:

- **80%** of that weight is redistributed among non-zero peers
- **20%** is retained as penalty

This applies both:
- inside each section (component level), and
- across final section weights (section level), when needed.

Diagnostics returned in output include:
- `effective_component_weights`
- `zero_components`
- `penalty_weight_due_to_zeros`
- `effective_final_score_weights`
- `zero_score_sections`
- `final_weight_penalty_due_to_zero_sections`

---

## ATS Output Structure (Summary)

`ats_score.json` contains:
- `final_ats_score` (0–100)
- `scoring_profiles`
- `final_score_weights`
- `effective_final_score_weights`
- `component_scores` for each section with:
  - `final_section_score`
  - `breakdown` (`exact_keyword_score`, `semantic_similarity`, `numeric_metrics_score`, `job_title_score`)
  - redistribution diagnostics

---

## Typical Command Flow

```bash
python pipeline.py
python ats_scoring_engine.py --job job_description.json --resume resume.json --output ats_score.json
```

---

## Troubleshooting

### `semantic_similarity` is always `0.0`
- Ensure dependencies are installed in active venv.
- Run model pre-download.

```bash
pip install -r requirements.txt
python download_model.py
```

### ATS score is unexpectedly low
Usually due to:
- low keyword overlap,
- title mismatch (`job_title_score = 0`),
- low semantic overlap.

### Git keeps showing Python cache changes
Ignore and untrack `__pycache__` / `.pyc` via `.gitignore` and git index cleanup.

---

## Notes

- Weights are configurable in `ats_scoring_engine.py`.
- Re-run scorer after weight/profile changes to regenerate `ats_score.json`.
