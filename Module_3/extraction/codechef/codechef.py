import requests
import re
from bs4 import BeautifulSoup

CODECHEF_BASE = "https://www.codechef.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://www.codechef.com/",
}


def extract_codechef_profile(username):
    """
    Extract skill-relevant CodeChef data for resume scoring:
      - current_rating & highest_rating  (competitive level)
      - stars                             (tier badge)
      - global_rank                       (standing)
      - problems_solved                   (volume of practice)
      - total_contests                    (experience)
    """
    
    try:
        profile_url = f"{CODECHEF_BASE}/users/{username}"
        session = requests.Session()
        response = session.get(profile_url, headers=_HEADERS, timeout=15)

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}",
                "username": username,
                "profile_url": profile_url,
            }

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        # --- Current rating ---
        current_rating = None
        for tag, cls in [("div", "rating-number"), ("span", "rating-number"), ("div", "rating")]:
            elem = soup.find(tag, class_=cls)
            if elem:
                m = re.search(r"\d+", elem.get_text())
                if m:
                    current_rating = int(m.group())
                    break

        # --- Highest rating ---
        highest_rating = None
        m = re.search(r"Highest\s+Rating[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            highest_rating = int(m.group(1))

        # --- Stars ---
        stars = None
        m = re.search(r"(\d+)\s*[★\*]", text)
        if m:
            stars = f"{m.group(1)}★"

        # --- Global rank ---
        global_rank = None
        m = re.search(r"Global\s+Rank[:\s]*(\d+)", text, re.IGNORECASE)
        if m:
            global_rank = int(m.group(1))

        # --- Problems solved ---
        problems_solved = 0
        section = soup.find("section", class_="rating-data-section problems-solved")
        if section:
            h5 = section.find("h5")
            if h5:
                m = re.search(r"\d+", h5.get_text())
                if m:
                    problems_solved = int(m.group())
        if not problems_solved:
            for pattern in [
                r"Fully\s+Solved[:\s]*(\d+)",
                r"Problems?\s+Solved[:\s]*(\d+)",
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    problems_solved = int(m.group(1))
                    break

        # --- Total contests ---
        total_contests = 0
        section = soup.find("section", class_="rating-data-section contest-participated-count")
        if section:
            h5 = section.find("h5")
            if h5:
                m = re.search(r"\d+", h5.get_text())
                if m:
                    total_contests = int(m.group())
        if not total_contests:
            for pattern in [
                r"Contests?\s+Participated[:\s]*(\d+)",
                r"Total\s+Contests[:\s]*(\d+)",
            ]:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    total_contests = int(m.group(1))
                    break

        return {
            "username":       username,
            "profile_url":    profile_url,
            "current_rating": current_rating,
            "highest_rating": highest_rating,
            "stars":          stars,
            "global_rank":    global_rank,
            "problems_solved": problems_solved,
            "total_contests": total_contests,
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timeout", "username": username}
    except Exception as e:
        return {"error": str(e), "username": username}


if __name__ == "__main__":
    import json
    with open("../.env", "r") as f:
        username = None
        for line in f:
            if line.lower().startswith("codechef"):
                username = line.split(":", 1)[1].strip()
                break
    if username:
        data = extract_codechef_profile(username)
        with open("codechef_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("✓ CodeChef data saved to codechef_data.json")
