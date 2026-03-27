"""
Merge individual1.json with all extracted platform data into final resume.json
"""
import json
import os

def load_json_safe(filepath):
    """Safely load JSON file, return None if file doesn't exist"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠ Warning: Could not load {filepath}: {str(e)}")
            return None
    return None

def merge_all_data(individual_json_path='../individual1.json', output_path='../resume.json'):
    """
    Merge individual1.json with all extracted data into resume.json
    """
    print("\n" + "=" * 60)
    print("MERGING DATA INTO FINAL RESUME")
    print("=" * 60)
    
    # Adjust paths if running from extraction folder
    if not os.path.exists(individual_json_path):
        individual_json_path = 'individual1.json'
    
    # Load base data from individual1.json
    print("\n[1/6] Loading base data from individual1.json...")
    individual_data = load_json_safe(individual_json_path)
    
    if not individual_data:
        raise FileNotFoundError(f"Cannot find individual1.json")
    
    print("  ✓ Base data loaded")
    
    # Initialize final resume structure — copy all fields as-is from individual data.
    # Do NOT set defaults or invent values; honour whatever the input had (including empty lists/dicts).
    resume = {
        "personal_info": {
            "name": individual_data.get("name", ""),
            "links": individual_data.get("links", {})
        },
        "projects": individual_data.get("projects", []),
        "skills": individual_data.get("skills", {}),
        "experience": individual_data.get("experience", []),
        "education": individual_data.get("education", {}),
        # Always present with keys for every platform, empty dict if not extracted
        "competitive_programming": {
            "leetcode": {},
            "codechef": {},
            "codeforces": {}
        },
        # Always present, empty dicts if not extracted
        "github": {
            "profile_info": {},
            "tech_stack": {}
        }
    }
    
    # Preserve competitive_programming_links exactly as given in input (may be empty strings)
    if "competitive_programming_links" in individual_data:
        resume["personal_info"]["competitive_programming_links"] = individual_data["competitive_programming_links"]
    
    # Load LeetCode data — key is always present; populated if data exists, empty dict otherwise
    print("\n[2/6] Loading LeetCode data...")
    leetcode_data = load_json_safe('leetcode/leetcode_data.json')
    if leetcode_data:
        resume["competitive_programming"]["leetcode"] = leetcode_data
        print("  ✓ LeetCode data merged")
    else:
        print("  ○ LeetCode not extracted — entry remains empty")
    
    # Load CodeChef data — key is always present; populated if data exists, empty dict otherwise
    print("\n[3/6] Loading CodeChef data...")
    codechef_data = load_json_safe('codechef/codechef_data.json')
    if codechef_data:
        resume["competitive_programming"]["codechef"] = codechef_data
        print("  ✓ CodeChef data merged")
    else:
        print("  ○ CodeChef not extracted — entry remains empty")
    
    # Load Codeforces data — key is always present; populated if data exists, empty dict otherwise
    print("\n[4/6] Loading Codeforces data...")
    codeforces_data = load_json_safe('codeforces/codeforces_data.json')
    if codeforces_data:
        resume["competitive_programming"]["codeforces"] = codeforces_data
        print("  ✓ Codeforces data merged")
    else:
        print("  ○ Codeforces not extracted — entry remains empty")
    
    # Load GitHub data — keys are always present; populated if data exists, empty otherwise
    print("\n[5/6] Loading GitHub data...")
    github_data = load_json_safe('github/github_data.json')

    if github_data:
        resume["github"]["profile_info"] = github_data.get("profile_info", {})
        resume["github"]["tech_stack"]    = github_data.get("tech_stack",   {})
        print("  ✓ GitHub data merged")
    else:
        print("  ○ GitHub not extracted — entry remains empty")
    
    # Save final resume
    print("\n[6/6] Saving final resume.json...")
    
    # Adjust output path
    if output_path.startswith('../'):
        output_path = output_path
    else:
        # We're in root, save to root
        if os.path.exists('extraction'):
            output_path = 'resume.json'
        else:
            output_path = '../resume.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Final resume saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    
    # Print summary
    print("\nResume Summary:")
    print(f"  • Projects: {len(resume.get('projects', []))}")
    print(f"  • Skills: {len(resume.get('skills', {}))}")
    print(f"  • Experience: {len(resume.get('experience', []))}")
    
    cp_platforms = len(resume.get('competitive_programming', {}))
    if cp_platforms > 0:
        print(f"  • Competitive Programming Platforms: {cp_platforms}")
        for platform in resume.get('competitive_programming', {}).keys():
            print(f"    - {platform.title()}")
    
    if resume.get('github', {}).get('profile_info'):
        print(f"  • GitHub: Profile & Repository data included")
    
    print("=" * 60)
    
    return resume

if __name__ == "__main__":
    merge_all_data()
