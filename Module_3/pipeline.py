"""
Main Pipeline for Resume Data Extraction and Generation

This script orchestrates the entire process:
1. Reads individual1.json
2. Extracts profile links/usernames directly from JSON
3. Fetches data from all competitive programming platforms and GitHub
4. Merges everything into a final resume.json

Usage:
    python pipeline.py
    
Or with custom input file:
    python pipeline.py --input my_resume.json
"""

import sys
import os
import argparse

# Add extraction directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'extraction'))

from extraction.extract_links import generate_config_from_individual
from extraction.extract_all import extract_all_profiles
from extraction.merge_data import merge_all_data


def run_pipeline(input_file='individual1.json', output_file='resume.json'):
    """
    Run the complete pipeline from individual JSON to final resume JSON
    
    Args:
        input_file: Path to input JSON file with basic resume info (default: individual1.json)
        output_file: Path to output final resume JSON (default: resume.json)
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "RESUME GENERATION PIPELINE")
    print("=" * 70)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    print("=" * 70)
    
    try:
        # Step 1: Extract links/usernames directly from individual JSON
        print("\n" + "▶ " * 35)
        print("STEP 1: EXTRACTING PROFILE LINKS FROM INPUT FILE")
        print("▶ " * 35)
        
        config = generate_config_from_individual(individual_json_path=input_file)
        
        # Report how many profiles were found, but always continue —
        # platforms not present in the input will be skipped during extraction
        # and will appear as empty entries in the final resume.json
        found_count = len([v for v in config.values() if v])
        print(f"\n✓ Step 1 Complete: {found_count} profile(s) found — pipeline continues regardless")
        
        # Step 2: Extract data from all platforms
        print("\n" + "▶ " * 35)
        print("STEP 2: EXTRACTING DATA FROM PLATFORMS")
        print("▶ " * 35)
        
        # Change to extraction directory to run extraction
        original_dir = os.getcwd()
        os.chdir('extraction')
        
        try:
            extract_all_profiles(config)
        finally:
            # Change back to original directory
            os.chdir(original_dir)
        
        print("\n✓ Step 2 Complete: Platform data extracted")
        
        # Step 3: Merge all data into final resume
        print("\n" + "▶ " * 35)
        print("STEP 3: MERGING DATA INTO FINAL RESUME")
        print("▶ " * 35)
        
        # Change to extraction directory for merging
        os.chdir('extraction')
        
        try:
            resume = merge_all_data(
                individual_json_path=f'../{input_file}',
                output_path=f'../{output_file}'
            )
        finally:
            os.chdir(original_dir)
        
        print("\n✓ Step 3 Complete: Final resume generated")
        
        # Final success message
        print("\n" + "=" * 70)
        print(" " * 20 + "✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📄 Your final resume has been generated: {output_file}")
        print("\nThe resume includes:")
        print("  • Personal information and links")
        print("  • Projects and skills")
        print("  • Competitive programming statistics")
        print("  • GitHub profile and repository data")
        print("\n" + "=" * 70)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease make sure:")
        print(f"  1. {input_file} exists in the current directory")
        print("  2. The file contains valid JSON data")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with error:")
        print(f"   {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Resume Generation Pipeline - Extract and merge resume data from multiple sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                              # Use default individual1.json
  python pipeline.py --input mydata.json          # Use custom input file
  python pipeline.py --input data.json --output final_resume.json
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='individual1.json',
        help='Input JSON file with basic resume info (default: individual1.json)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='resume.json',
        help='Output JSON file for final resume (default: resume.json)'
    )
    
    args = parser.parse_args()
    
    # Run the pipeline
    success = run_pipeline(args.input, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
