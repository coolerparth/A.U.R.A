import requests

LEETCODE_API = "https://leetcode.com/graphql"


def extract_leetcode_profile(username):
    """
    Extract skill-relevant LeetCode data for resume scoring:
      - Problems solved by difficulty  (measures DSA depth)
      - Skill tags declared on profile (explicit skills)
      - Languages used for submissions (programming language breadth)
      - Contest rating / percentile    (competitive level)
    """

    query = """
    query getUserProfile($username: String!) {
        matchedUser(username: $username) {
            username
            profile {
                skillTags
            }
            submitStats {
                acSubmissionNum {
                    difficulty
                    count
                }
            }
            languageProblemCount {
                languageName
                problemsSolved
            }
        }
        userContestRanking(username: $username) {
            attendedContestsCount
            rating
            globalRanking
            topPercentage
        }
    }
    """

    try:
        response = requests.post(
            LEETCODE_API,
            json={"query": query, "variables": {"username": username}},
            headers={"Content-Type": "application/json"},
        )
        data = response.json()

        if not data.get("data") or not data["data"].get("matchedUser"):
            return {"error": "User not found or API error", "username": username}

        user    = data["data"]["matchedUser"]
        contest = data["data"].get("userContestRanking") or {}

        ac = {
            item["difficulty"]: item["count"]
            for item in user["submitStats"]["acSubmissionNum"]
        }

        # languageProblemCount may not be present for all accounts
        lang_counts = [
            {"language": lp["languageName"], "problems_solved": lp["problemsSolved"]}
            for lp in (user.get("languageProblemCount") or [])
        ]

        return {
            "username": user["username"],
            "problems_solved": {
                "total":  ac.get("All",    0),
                "easy":   ac.get("Easy",   0),
                "medium": ac.get("Medium", 0),
                "hard":   ac.get("Hard",   0),
            },
            "skill_tags": user["profile"].get("skillTags", []),
            "languages_used": lang_counts,
            "contest": {
                "attended":       contest.get("attendedContestsCount", 0),
                "rating":         round(contest.get("rating", 0) or 0, 2),
                "global_ranking": contest.get("globalRanking", 0),
                "top_percentage": contest.get("topPercentage", 0),
            },
        }

    except Exception as e:
        return {"error": str(e), "username": username}


if __name__ == "__main__":
    import json
    with open('../.env', 'r') as f:
        username = None
        for line in f:
            if line.lower().startswith('leetcode'):
                username = line.split(':', 1)[1].strip()
                break
    if username:
        data = extract_leetcode_profile(username)
        with open('leetcode_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print("✓ LeetCode data saved to leetcode_data.json")


# -----------------------------
# MAIN DRIVER
# -----------------------------

if __name__ == "__main__":
    # Read from .env file
    with open('../.env', 'r') as f:
        lines = f.readlines()
        leetcode_username = None
        for line in lines:
            if 'leetcode' in line.lower():
                leetcode_username = line.split(':', 1)[1].strip()
                break
    
    if leetcode_username:
        print(f"Processing LeetCode profile: {leetcode_username}")
        
        profile_data = extract_leetcode_profile(leetcode_username)
        
        # Save to JSON file
        with open('leetcode_data.json', 'w') as f:
            json.dump(profile_data, f, indent=2)
        print("✓ LeetCode data saved to leetcode_data.json")
    else:
        print("LeetCode username not found in .env file")
