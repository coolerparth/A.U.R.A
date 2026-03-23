import requests

CODEFORCES_API = "https://codeforces.com/api"


def extract_codeforces_profile(username):
    """
    Extract skill-relevant Codeforces data for resume scoring:
      - Rating & rank          (competitive programming level)
      - Total contests         (experience)
      - Problems solved        (volume of practice)
      - Rating distribution    (difficulty distribution of solved problems)
      - Languages used         (programming language breadth)
    """
    try:
        # --- User info ---
        user_resp = requests.get(f"{CODEFORCES_API}/user.info?handles={username}")
        user_data = user_resp.json()
        if user_data["status"] != "OK":
            return {"error": "User not found", "username": username}
        user = user_data["result"][0]

        # --- Rating history (for total contests count) ---
        rating_resp  = requests.get(f"{CODEFORCES_API}/user.rating?handle={username}")
        rating_data  = rating_resp.json()
        total_contests = 0
        if rating_data["status"] == "OK":
            total_contests = len(rating_data["result"])

        # --- Submissions (last 500 for stats) ---
        subs_resp = requests.get(
            f"{CODEFORCES_API}/user.status?handle={username}&from=1&count=500"
        )
        subs_data = subs_resp.json()

        solved_problems    = set()
        rating_distribution = {}   # problem rating -> count of distinct solved
        languages_used     = {}    # language -> accepted submission count

        if subs_data["status"] == "OK":
            for sub in subs_data["result"]:
                if sub.get("verdict") != "OK":
                    continue
                problem = sub["problem"]
                uid = f"{problem.get('contestId', '')}{problem.get('index', '')}"
                if uid not in solved_problems:
                    solved_problems.add(uid)
                    if "rating" in problem:
                        r = problem["rating"]
                        rating_distribution[r] = rating_distribution.get(r, 0) + 1
                lang = sub.get("programmingLanguage", "Unknown")
                languages_used[lang] = languages_used.get(lang, 0) + 1

        return {
            "username":    user.get("handle"),
            "rating":      user.get("rating"),
            "max_rating":  user.get("maxRating"),
            "rank":        user.get("rank"),
            "max_rank":    user.get("maxRank"),
            "total_contests":   total_contests,
            "problems_solved":  len(solved_problems),
            "rating_distribution": rating_distribution,
            "languages_used":   languages_used,
        }

    except Exception as e:
        return {"error": str(e), "username": username}


if __name__ == "__main__":
    import json
    with open('../.env', 'r') as f:
        username = None
        for line in f:
            if line.lower().startswith('codeforces'):
                username = line.split(':', 1)[1].strip()
                break
    if username:
        data = extract_codeforces_profile(username)
        with open('codeforces_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("✓ Codeforces data saved to codeforces_data.json")

    else:
        print("Codeforces username not found in .env file")
