"""
Extract profile links/usernames directly from individual JSON.
"""
import json
import os
import re

def extract_username_from_url(url, platform):
    """Extract username from profile URL"""
    if not url or url.strip() == "":
        return None
    
    url = url.strip()
    
    if platform == "github":
        # https://github.com/username
        match = re.search(r'github\.com/([^/]+)/?$', url)
        return match.group(1) if match else None
    
    elif platform == "leetcode":
        # https://leetcode.com/username/ or just username
        if 'leetcode.com' in url:
            match = re.search(r'leetcode\.com/(?:u/)?([^/]+)/?', url)
            return match.group(1) if match else None
        return url  # Assume it's just the username
    
    elif platform == "codeforces":
        # https://codeforces.com/profile/username or just username
        if 'codeforces.com' in url:
            match = re.search(r'codeforces\.com/profile/([^/]+)/?', url)
            return match.group(1) if match else None
        return url  # Assume it's just the username
    
    elif platform == "codechef":
        # https://www.codechef.com/users/username or just username
        if 'codechef.com' in url:
            match = re.search(r'codechef\.com/users/([^/]+)/?', url)
            return match.group(1) if match else None
        return url  # Assume it's just the username
    
    return None

def extract_config_from_individual(individual_json_path='../individual1.json'):
    """
    Extract links from individual JSON and return normalized usernames/config.
    """
    # Adjust path if running from extraction folder
    if not os.path.exists(individual_json_path):
        individual_json_path = 'individual1.json'  # Try root directory
    
    if not os.path.exists(individual_json_path):
        raise FileNotFoundError(f"Cannot find individual1.json at {individual_json_path}")
    
    # Load individual1.json
    with open(individual_json_path, 'r') as f:
        data = json.load(f)
    
    config = {}
    
    # Extract GitHub link
    github_url = data.get('links', {}).get('github', '')
    if github_url:
        github_username = extract_username_from_url(github_url, 'github')
        if github_username:
            config['github_profile'] = github_username
    
    # Extract competitive programming links
    cp_links = data.get('competitive_programming_links', {})
    
    leetcode = cp_links.get('leetcode', '')
    if leetcode:
        leetcode_username = extract_username_from_url(leetcode, 'leetcode')
        if leetcode_username:
            config['leetcode'] = leetcode_username
    
    codeforces = cp_links.get('codeforces', '')
    if codeforces:
        codeforces_username = extract_username_from_url(codeforces, 'codeforces')
        if codeforces_username:
            config['codeforces'] = codeforces_username
    
    codechef = cp_links.get('codechef', '')
    if codechef:
        codechef_username = extract_username_from_url(codechef, 'codechef')
        if codechef_username:
            config['codechef'] = codechef_username
    
    found = [k for k, v in config.items() if v]
    print("✓ Configuration extracted directly from input JSON")
    print(f"  Found {len(found)} platform(s) to extract")
    for key in found:
        print(f"    • {key}: {config[key]}")

    not_found = [p for p in ['github_profile', 'leetcode', 'codeforces', 'codechef'] if not config.get(p)]
    for key in not_found:
        print(f"    ○ {key}: (not provided — will be skipped)")
    
    return config


# Backward-compatible alias
generate_config_from_individual = extract_config_from_individual

if __name__ == "__main__":
    extract_config_from_individual()
