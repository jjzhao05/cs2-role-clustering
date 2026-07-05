import sys
import time
import requests
import mwparserfromhell
import pandas as pd

BASE = "https://liquipedia.net/counterstrike/api.php"

HEADERS = {
    "User-Agent": "cs2-role-classifier/0.1 (tyylerrose224@gmail.com)",
    "Accept-Encoding": "gzip",
}

REQUEST_DELAY = 30
INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "output.csv"
OUTPUT_CSV = "liquipedia_player_roles.csv"

ROLE_ALIAS: dict[str, str] = {
    "awp": "AWPer",
    "awper": "AWPer",

    "rifle": "Rifler",
    "rifler": "Rifler",
    "support": "Rifler",

    "igl": "IGL",
    "in-game leader": "IGL",

    "entry": "Entry Fragger",
    "entry fragger": "Entry Fragger",
    "entryfragger": "Entry Fragger",

    "lurk": "Lurker",
    "lurker": "Lurker",

    "coach": "Coach",
    "analyst": "Analyst",
    "caster": "Caster",
}

REVIEW_ROLES = {"Coach", "Analyst", "Caster", "Unknown"}


def normalize_role(raw: str) -> str:
    cleaned = raw.strip()
    return ROLE_ALIAS.get(cleaned.lower(), cleaned) if cleaned else "Unknown"


def normalize_roles(raw_roles: list[str]) -> list[str]:
    return sorted({normalize_role(r) for r in raw_roles})


def load_players(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df.iloc[:, 0].dropna().astype(str).str.strip().unique().tolist())


def fetch_wikitext(session: requests.Session, page: str) -> str | None:
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": page,
        "rvprop": "content",
        "format": "json",
        "formatversion": "2",
    }

    r = session.get(BASE, params=params, timeout=20)
    r.raise_for_status()

    page_data = r.json()["query"]["pages"][0]
    if page_data.get("missing"):
        return None
    return page_data["revisions"][0]["content"]


def extract_roles(wikitext: str) -> list[str]:
    code = mwparserfromhell.parse(wikitext)

    for template in code.filter_templates():
        name = template.name.strip().lower().replace("_", " ")
        if name != "infobox player":
            continue

        roles: list[str] = []
        for key in ["roles", "role", "role2"]:
            if template.has(key):
                value = str(template.get(key).value).strip()
                roles.extend(r.strip() for r in value.replace("<br>", ",").split(",") if r.strip())

        return sorted(set(roles))

    return []


def fetch_roles_for_player(
    session: requests.Session,
    page: str,
) -> tuple[list[str], str | None, str | None]:
    """
    Try to fetch Liquipedia roles for a player page.

    Liquipedia player pages are usually lowercase handles, but input CSVs may
    contain mixed-case handles like NiKo, cadiaN, electroNic. So we try the
    original page first, then a fully lowercased version.
    """
    page = page.strip()
    candidates = dict.fromkeys([page, page.lower()])  # dedupe, preserve order

    for candidate in candidates:
        try:
            text = fetch_wikitext(session, candidate)
            if text is not None:
                return extract_roles(text), None, candidate
        except Exception as e:
            return [], str(e), candidate

    return [], None, None


def main() -> None:
    players = load_players(INPUT_CSV)
    print(f"Found {len(players)} unique players in '{INPUT_CSV}'\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    rows: list[dict] = []

    for i, page in enumerate(players, 1):
        raw_roles, error, matched_page = fetch_roles_for_player(session, page)
        canonical = normalize_roles(raw_roles) if raw_roles else []
        role_1, role_2 = (canonical + ["Unknown", ""])[:2]

        row = {
            "player_page": page,
            "matched_liquipedia_page": matched_page or "",
            "role_1": role_1,
            "role_2": role_2,
            "needs_review": role_1 in REVIEW_ROLES or role_2 in REVIEW_ROLES,
        }
        if error:
            row["error"] = error
        rows.append(row)

        role_display = f"{role_1} + {role_2}" if role_2 else role_1
        review_flag = "  ★ REVIEW" if row["needs_review"] else ""
        lookup_display = f"{page} → {matched_page}" if matched_page and matched_page != page else page

        status = f"[{i:>3}/{len(players)}] {lookup_display:<30} → {role_display}{review_flag}"
        if error:
            status += f"  ⚠  {error}"
        print(status)

        if i < len(players):
            time.sleep(REQUEST_DELAY)

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done. Results written to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()