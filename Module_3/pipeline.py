"""Central pipeline: candidate JSON -> resume.json -> ats_score.json."""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ats_scoring_engine import compute_ats_score, load_json, save_json
from extraction.extract_all import extract_all_profiles
from extraction.extract_links import generate_config_from_individual
from extraction.merge_data import merge_all_data


def run_pipeline(candidate_path: str, job_path: str, output_path: str, resume_path: str = "resume.json") -> bool:
    """Run extraction, merge, and ATS scoring in one command."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    extraction_dir = os.path.join(repo_root, "extraction")

    candidate_abs = os.path.abspath(candidate_path)
    job_abs = os.path.abspath(job_path)
    output_abs = os.path.abspath(output_path)
    resume_abs = os.path.abspath(resume_path)

    print("\n" + "=" * 72)
    print("SMART RESUME SCORING ENGINE - CENTRAL PIPELINE")
    print("=" * 72)
    print(f"Candidate Input : {candidate_abs}")
    print(f"Job Input       : {job_abs}")
    print(f"Resume Output   : {resume_abs}")
    print(f"Score Output    : {output_abs}")
    print("=" * 72)

    try:
        print("\n[1/4] Extracting profile configuration from candidate JSON")
        config = generate_config_from_individual(individual_json_path=candidate_abs)
        print(f"  Found platforms: {', '.join(sorted(config.keys())) if config else 'none'}")

        print("\n[2/4] Extracting platform data")
        original_dir = os.getcwd()
        os.chdir(extraction_dir)
        try:
            extract_all_profiles(config)
        finally:
            os.chdir(original_dir)

        print("\n[3/4] Merging extracted platform data into resume JSON")
        os.chdir(extraction_dir)
        try:
            merge_all_data(individual_json_path=candidate_abs, output_path=resume_abs)
        finally:
            os.chdir(original_dir)

        print("\n[4/4] Computing ATS score")
        job_json = load_json(job_abs)
        resume_json = load_json(resume_abs)
        result = compute_ats_score(job_json, resume_json)
        save_json(output_abs, result)

        print("\n" + "=" * 72)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 72)
        print(f"Generated: {resume_abs}")
        print(f"Generated: {output_abs}")
        print(f"Final ATS Score: {result.get('final_ats_score')}")
        return True
    except Exception as exc:
        print("\nPipeline failed")
        print(str(exc))
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Central pipeline for extraction + merge + ATS scoring",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to raw candidate JSON (example: individual1.json)",
    )
    parser.add_argument(
        "--job",
        required=True,
        help="Path to job description JSON (example: job_description.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to ATS output JSON (example: ats_score.json)",
    )
    parser.add_argument(
        "--resume-output",
        default="resume.json",
        help="Path to intermediate merged resume JSON (default: resume.json)",
    )

    args = parser.parse_args()
    ok = run_pipeline(
        candidate_path=args.candidate,
        job_path=args.job,
        output_path=args.output,
        resume_path=args.resume_output,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
