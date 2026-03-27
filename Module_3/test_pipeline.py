"""Interactive test pipeline: prompt for JD and resume paths, output ATS JSON."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ats_scoring_engine import compute_ats_score, load_json, save_json


def _prompt_existing_file(prompt_text: str) -> str:
    while True:
        raw = input(prompt_text).strip().strip('"').strip("'")
        if not raw:
            print("Path cannot be empty. Please try again.")
            continue
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            return path
        print(f"File not found: {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive ATS test pipeline")
    parser.add_argument("--job", help="Path to job description JSON")
    parser.add_argument("--resume", help="Path to resume JSON")
    parser.add_argument("--output", default="ats_score.json", help="Output ATS score JSON path")
    return parser.parse_args()


def main() -> None:
    print("=" * 68)
    print("ATS TEST PIPELINE")
    print("=" * 68)

    args = _parse_args()

    if args.job:
        job_path = os.path.abspath(args.job)
        if not os.path.isfile(job_path):
            raise FileNotFoundError(f"Job description file not found: {job_path}")
    else:
        job_path = _prompt_existing_file("Enter path to job description JSON: ")

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume file not found: {resume_path}")
    else:
        resume_path = _prompt_existing_file("Enter path to resume JSON: ")

    output_path = os.path.abspath(args.output)

    job_json = load_json(job_path)
    resume_json = load_json(resume_path)
    result = compute_ats_score(job_json, resume_json)
    save_json(output_path, result)

    print("\nATS scoring complete.")
    print(f"Output written to: {output_path}")
    print(f"Final ATS Score: {result.get('final_ats_score')}")


if __name__ == "__main__":
    main()
