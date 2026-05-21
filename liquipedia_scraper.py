import sys
import time
import requests
import mwparserfromhell
import pandas as pd

BASE = "https://liquipedia.net/counterstrike/api.php"

# Liquipedia requires a descriptive User-Agent; fill in your contact info.
HEADERS = {
    "User-Agent": "cs2-role-classifier/0.1 (your_email@example.com)",
    "Accept-Encoding": "gzip",
}

REQUEST_DELAY = 2   # seconds between requests (Liquipedia rate limit)
INPUT_CSV  = sys.argv[1] if len(sys.argv) > 1 else "output.csv"
OUTPUT_CSV = "liquipedia_player_roles.csv"


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

# Canonical role names (the single source of truth for output values).
CANONICAL_ROLES = {
    "AWPer",
    "Rifler",
    "IGL",
    "Entry Fragger",
    "Lurker",
    "Support",
    "Coach",
    "Analyst",
    "Caster",
}

# Maps any raw variant → canonical role.
# Keys are lowercased + stripped before lookup.
ROLE_ALIAS: dict[str, str] = {
    # AWPer
    "awp":        "AWPer",
    "awper":      "AWPer",
    # Rifler  (support → Rifler as requested)
    "rifle":      "Rifler",
    "rifler":     "Rifler",
    "support":    "Rifler",
    # IGL
    "igl":        "IGL",
    "in-game leader": "IGL",
    # Entry
    "entry":          "Entry",
    "entry fragger":  "Entry",
    "entryfragger":   "Entry",
    # Lurker
    "lurk":    "Lurker",
    "lurker":  "Lurker",
    # Staff / non-player roles
    "coach":    "Coach",
    "analyst":  "Analyst",
    "caster":   "Caster",
}


def normalize_role(raw: str) -> str:
    """Return the canonical role for *raw*, or 'Unknown' if unrecognised."""
    return ROLE_ALIAS.get(raw.strip().lower(), "Unknown")


def normalize_roles(raw_roles: list[str]) -> list[str]:
    """
    Normalize a list of raw role strings.

    Steps
    -----
    1. Map each raw value to its canonical form.
    2. Drop duplicates that arise after mapping (e.g. 'rifle' + 'rifler' → one 'Rifler').
    3. Return a deterministically sorted list so output is stable.
    """
    seen: set[str] = set()
    result: list[str] = []
    for raw in raw_roles:
        canonical = normalize_role(raw)
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return sorted(result)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_players(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df.iloc[:, 0].dropna().astype(str).unique().tolist())


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
    pages = r.json()["query"]["pages"]
    if pages[0].get("missing"):
        return None
    return pages[0]["revisions"][0]["content"]


def extract_roles(wikitext: str) -> list[str]:
    code = mwparserfromhell.parse(wikitext)
    for template in code.filter_templates():
        name = template.name.strip().lower().replace("_", " ")
        if name == "infobox player":
            roles: list[str] = []
            for key in ["roles", "role", "role2"]:
                if template.has(key):
                    value = str(template.get(key).value).strip()
                    if value:
                        roles.extend(
                            r.strip()
                            for r in value.replace("<br>", ",").split(",")
                            if r.strip()
                        )
            return sorted(set(roles))
    return []


def fetch_roles_for_player(
    session: requests.Session, page: str
) -> tuple[list[str], str | None]:
    candidates = [page]
    if page and page[0].islower():
        candidates.append(page[0].upper() + page[1:])

    for candidate in candidates:
        try:
            text = fetch_wikitext(session, candidate)
            if text is not None:
                return extract_roles(text), None
        except requests.HTTPError as e:
            return [], str(e)
        except Exception as e:
            return [], str(e)

    return [], None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    players = load_players(INPUT_CSV)
    print(f"Found {len(players)} unique players in '{INPUT_CSV}'\n")

    session = requests.Session()
    session.headers.update(HEADERS)

    rows: list[dict] = []
    for i, page in enumerate(players, 1):
        raw_roles, error = fetch_roles_for_player(session, page)

        canonical = normalize_roles(raw_roles) if raw_roles else []
        role_1 = canonical[0] if len(canonical) > 0 else "Unknown"
        role_2 = canonical[1] if len(canonical) > 1 else ""

        row: dict = {"player_page": page, "role_1": role_1, "role_2": role_2}
        if error:
            row["error"] = error

        rows.append(row)

        role_display = f"{role_1} + {role_2}" if role_2 else role_1
        status = f"[{i:>3}/{len(players)}] {page:<20} → {role_display}"
        if error:
            status += f"  ⚠  {error}"
        print(status)

        if i < len(players):
            time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Done. Results written to '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()