import json
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(__file__))

from github.git import extract_github_data
from leetcode.leetcode import extract_leetcode_profile
from codeforces.codeforces import extract_codeforces_profile
from codechef.codechef import extract_codechef_profile
from extract_links import extract_config_from_individual

def _clear_json(filepath):
    """Remove a stale data file so the merge step won't pick up an old run's result"""
    if os.path.exists(filepath):
        os.remove(filepath)

def extract_all_profiles(config):
    """Extract data from all platforms and save to respective folders."""
    
    print("=" * 60)
    print("PROFILE DATA EXTRACTION - CODING PLATFORMS")
    print("=" * 60)
    
    # GitHub Extraction
    print("\n[1/4] GitHub:")
    if config.get('github_profile'):
        try:
            username = config['github_profile']
            print(f"  → Username: {username}")
            github_data = extract_github_data(username)
            with open('github/github_data.json', 'w') as f:
                json.dump(github_data, f, indent=2)
            print("  ✓ GitHub data saved successfully")
        except Exception as e:
            print(f"  ✗ GitHub extraction failed: {str(e)}")
    else:
        _clear_json('github/github_data.json')
        print("  ○ Skipping — github_profile not provided in input")
    
    # LeetCode Extraction
    print("\n[2/4] LeetCode:")
    if config.get('leetcode'):
        try:
            username = config['leetcode']
            print(f"  → Username: {username}")
            
            leetcode_data = extract_leetcode_profile(username)
            
            # Save to leetcode folder
            with open('leetcode/leetcode_data.json', 'w') as f:
                json.dump(leetcode_data, f, indent=2)
            
            print("  ✓ LeetCode data saved successfully")
        except Exception as e:
            print(f"  ✗ LeetCode extraction failed: {str(e)}")
    else:
        _clear_json('leetcode/leetcode_data.json')
        print("  ○ Skipping — leetcode username not provided in input")
    
    # Codeforces Extraction
    print("\n[3/4] Codeforces:")
    if config.get('codeforces'):
        try:
            username = config['codeforces']
            print(f"  → Username: {username}")
            
            codeforces_data = extract_codeforces_profile(username)
            
            # Save to codeforces folder
            with open('codeforces/codeforces_data.json', 'w') as f:
                json.dump(codeforces_data, f, indent=2)
            
            print("  ✓ Codeforces data saved successfully")
        except Exception as e:
            print(f"  ✗ Codeforces extraction failed: {str(e)}")
    else:
        _clear_json('codeforces/codeforces_data.json')
        print("  ○ Skipping — codeforces username not provided in input")
    
    # CodeChef Extraction
    print("\n[4/4] CodeChef:")
    if config.get('codechef'):
        try:
            username = config['codechef']
            print(f"  → Username: {username}")
            
            codechef_data = extract_codechef_profile(username)
            
            # Save to codechef folder
            with open('codechef/codechef_data.json', 'w') as f:
                json.dump(codechef_data, f, indent=2)
            
            print("  ✓ CodeChef data saved successfully")
        except Exception as e:
            print(f"  ✗ CodeChef extraction failed: {str(e)}")
    else:
        _clear_json('codechef/codechef_data.json')
        print("  ○ Skipping — codechef username not provided in input")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print("\nData saved in respective folders:")
    print("  • github/github_data.json")
    print("  • leetcode/leetcode_data.json")
    print("  • codeforces/codeforces_data.json")
    print("  • codechef/codechef_data.json")
    print("=" * 60)

if __name__ == "__main__":
    # Standalone fallback mode: read directly from ../individual1.json
    cfg = extract_config_from_individual('../individual1.json')
    extract_all_profiles(cfg)
