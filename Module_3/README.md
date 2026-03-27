# Smart Resume Scoring Engine

## Overview
Smart Resume Scoring Engine is an end-to-end Python system that takes a candidate JSON and a job description JSON, enriches the candidate profile with platform data (GitHub, LeetCode, Codeforces, CodeChef), builds a unified `resume.json`, and generates a detailed ATS result in `ats_score.json`.

## Features
- Proprietary deterministic scoring algorithm (no LLM-based score generation)
- Multi-platform technical data integration
- Semantic and keyword-based matching against JD sections
- Weighted, section-wise ATS scoring with detailed diagnostics
- Single-command central pipeline execution

## Installation
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional model warm-up:

```bash
python download_model.py
```

## Usage

Run the full workflow (extraction + merge + scoring) with one command:

```bash
python pipeline.py --candidate individual1.json --job job_description.json --output ats_score.json
```

What this command does:
1. Reads the candidate JSON.
2. Extracts platform usernames and fetches profile data.
3. Merges all data into `resume.json`.
4. Scores `resume.json` against the job description.
5. Writes detailed ATS results to the output path.

## Test Pipeline (Interactive)

Use the interactive tester when you already have a resume JSON and want to score quickly.

```bash
python test_pipeline.py
```

The script will ask for:
1. Job description JSON path
2. Resume JSON path

It writes the result to `ats_score.json` in the project root.

Example interactive input:
1. `job_description.json`
2. `resume2.json`

Direct test command (non-interactive):

```bash
python test_pipeline.py --job job_description.json --resume resume2.json --output ats_score.json
```

## Output
- `resume.json`: merged candidate data with extracted platform information
- `ats_score.json`: final composite score and section/component-level breakdown
