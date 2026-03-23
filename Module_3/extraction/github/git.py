import requests
import re
import os
import time

GITHUB_API = "https://api.github.com"

# Load a GitHub Personal Access Token from the .env file if available.
# Without a token: 60 requests/hour.  With a token: 5000 requests/hour.
# To add one: create a token at https://github.com/settings/tokens
# then add a line   github_token: ghp_xxxx   to extraction/.env
def _load_token():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.lower().startswith('github_token'):
                    return line.split(':', 1)[1].strip()
    return None

_TOKEN = _load_token()

HEADERS = {
    "Accept": "application/vnd.github+json",
    **(  {"Authorization": f"Bearer {_TOKEN}"} if _TOKEN else {}  ),
}

# Languages that are build/config artefacts, not programming skills
_SKIP_LANGUAGES = {
    "Batchfile", "PowerShell", "Shell", "Makefile", "Dockerfile",
    "CMake", "YAML", "JSON", "XML", "TOML", "INI", "Text",
}

# Keywords scanned inside repo descriptions and READMEs to detect frameworks/tools
# beyond what the GitHub language API already reports.
FRAMEWORK_KEYWORDS = [
    # Web frameworks
    "react", "angular", "vue", "nextjs", "express", "nodejs", "django", "flask",
    "fastapi", "spring", "laravel", "rails", "nestjs", "svelte",
    # ML / Data Science
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "opencv", "hugging face", "transformers", "langchain",
    # Databases
    "mongodb", "postgresql", "mysql", "sqlite", "redis", "elasticsearch", "firebase",
    # DevOps / Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "github actions", "jenkins", "ci/cd",
    # APIs / Protocols
    "graphql", "rest", "grpc", "websocket",
    # Other tooling
    "solidity", "web3", "kafka", "rabbitmq", "celery", "nginx", "linux",
]


def _detect_keywords(text):
    """Return any FRAMEWORK_KEYWORDS found in the given text (case-insensitive)."""
    text_lower = text.lower()
    return {
        kw for kw in FRAMEWORK_KEYWORDS
        if re.search(rf"\b{re.escape(kw)}\b", text_lower)
    }


import time

def _fetch_all_pages(url, params=None):
    """Fetch every page of a GitHub list endpoint and return the combined list."""
    results = []
    page = 1
    params = dict(params or {})
    params["per_page"] = 100
    while True:
        params["page"] = page
        resp = requests.get(url, headers=HEADERS, params=params)
        if resp.status_code == 403:
            remaining = resp.headers.get("X-RateLimit-Remaining", "?")
            reset     = resp.headers.get("X-RateLimit-Reset", "?")
            raise RuntimeError(
                f"GitHub API rate limit exceeded (remaining={remaining}, "
                f"resets at unix={reset}). "
                f"Add a github_token to extraction/.env to raise the limit to 5000 req/hr."
            )
        if resp.status_code != 200:
            break
        batch = resp.json()
        if not batch:
            break
        results.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return results


def _get_repo_languages(username, repo_name):
    """
    Fetch language bytes for one repo.
    Returns a dict {language: bytes} filtered to skill-relevant languages only.
    """
    url  = f"{GITHUB_API}/repos/{username}/{repo_name}/languages"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code == 403:
        remaining = resp.headers.get("X-RateLimit-Remaining", "?")
        reset     = resp.headers.get("X-RateLimit-Reset", "?")
        raise RuntimeError(
            f"GitHub API rate limit exceeded while fetching {repo_name} "
            f"(remaining={remaining}, resets at unix={reset}). "
            f"Add a github_token to extraction/.env to raise the limit to 5000 req/hr."
        )
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, dict) and "message" not in data:
            return {lang: b for lang, b in data.items() if lang not in _SKIP_LANGUAGES}
    return {}


# -----------------------------
# MAIN EXTRACTION ENTRY POINT
# -----------------------------

def extract_github_data(username):
    """
    Extract GitHub profile and aggregate the complete tech stack across
    ALL of the user's own (non-fork) repositories.

    Returns:
        profile_info  – basic identity fields
        tech_stack    – aggregated language bytes, ranked language list,
                        and extra frameworks/tools detected from repo
                        descriptions and topics
    """
    # --- Profile ---
    profile = requests.get(f"{GITHUB_API}/users/{username}", headers=HEADERS).json()

    # --- All owned (non-fork) repos ---
    all_repos = _fetch_all_pages(
        f"{GITHUB_API}/users/{username}/repos",
        params={"type": "owner"}
    )
    own_repos = [r for r in all_repos if not r.get("fork")]
    print(f"  → Scanning {len(own_repos)} repos for tech stack...")

    aggregated_languages = {}   # language → total bytes across all repos
    all_topics           = set()
    detected_tools       = set()

    for repo in own_repos:
        repo_name = repo["name"]

        # Language bytes for this repo
        repo_langs = _get_repo_languages(username, repo_name)
        for lang, byte_count in repo_langs.items():
            aggregated_languages[lang] = aggregated_languages.get(lang, 0) + byte_count

        # Topics (often framework/tool names)
        topics = repo.get("topics") or []
        all_topics.update(topics)

        # Scan description for additional framework/tool keywords
        detected_tools.update(_detect_keywords(repo.get("description") or ""))
        # Scan repo name as well (many repos are named after the tool they use)
        detected_tools.update(_detect_keywords(repo_name.replace("-", " ").replace("_", " ")))

    # Topics on GitHub are often tool names — fold them in
    detected_tools.update(all_topics)

    # Rank languages by total byte count (most-used first)
    languages_ranked = sorted(
        aggregated_languages, key=aggregated_languages.get, reverse=True
    )

    return {
        "profile_info": {
            "username":    profile.get("login"),
            "name":        profile.get("name"),
            "bio":         profile.get("bio"),
            "public_repos": profile.get("public_repos"),
            "followers":    profile.get("followers"),
            "profile_url":  profile.get("html_url"),
        },
        "tech_stack": {
            "languages":        aggregated_languages,
            "languages_ranked": languages_ranked,
            "topics_and_tools": sorted(detected_tools),
        },
    }


# -----------------------------
# MAIN DRIVER
# -----------------------------

if __name__ == "__main__":
    with open('../.env', 'r') as f:
        username = None
        for line in f:
            if line.lower().startswith('github_profile'):
                username = line.split(':', 1)[1].strip()
                break
    if username:
        import json
        data = extract_github_data(username)
        with open('github_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("✓ GitHub data saved to github_data.json")

    profile_json = extract_profile(username)
    repo_json = extract_repo(repo_url)

    # Save profile JSON to file
    with open('profile_data.json', 'w') as f:
        json.dump(profile_json, f, indent=2)
    print("\n✓ Profile data saved to profile_data.json")

    # Save repo JSON to file
    with open('repo_data.json', 'w') as f:
        json.dump(repo_json, f, indent=2)
    print("✓ Repository data saved to repo_data.json")
