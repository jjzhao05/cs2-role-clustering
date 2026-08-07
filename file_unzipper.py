import re
import shutil
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from awpy import Demo

PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = Path.home() / "Downloads"
DEMOS_DIR = PROJECT_DIR / "demos"
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
MAX_FILE_AGE_HOURS = 6
MIN_VALID_ROUNDS = 2


def _first_char_of_match(match: re.Match) -> str:
    return match.group()[0]

#Demo downloads have large strings of characters that are random IDs. This function removes those random IDs from the demo file names.

def strip_random_ids(name: str) -> str:
    tokens = re.split(r"([-_])", name)

    kept_parts = []
    for i in range(0, len(tokens), 2):
        word = tokens[i]

        if i + 1 < len(tokens):
            separator = tokens[i + 1]
        else:
            separator = ""

        is_junk = any(c.isupper() for c in word)
        if is_junk:
            continue

        kept_parts.append(word)
        kept_parts.append(separator)

    cleaned = "".join(kept_parts)
    cleaned = re.sub(r"[-_]{2,}", _first_char_of_match, cleaned) 
    cleaned = cleaned.strip("-_")

    if cleaned:
        return cleaned
    return name


ACTIVE_DUTY_MAPS = [
    "de_mirage", "de_inferno", "de_nuke", "de_overpass", "de_ancient",
    "de_anubis", "de_dust2",
]
MAP_PATTERN = re.compile("|".join(re.escape(m) for m in ACTIVE_DUTY_MAPS).encode())


def get_map_name(dem_path: Path) -> str:
    match = MAP_PATTERN.search(dem_path.read_bytes()[:8192])
    if match:
        return match.group().decode()
    return "unknown_map"


def same_file_content(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False

    def digest(path: Path) -> bytes:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.digest()

    return digest(left) == digest(right)


def validate_demo(dem_path: Path) -> tuple[bool, str]:
    """Perform a lightweight round-event parse before accepting a demo."""
    try:
        demo = Demo(str(dem_path))
        events = demo.parse_events(
            ["round_start", "round_freeze_end", "round_end", "round_officially_ended"]
        )
        round_ends = events.get("round_end")
        if round_ends is None or "tick" not in round_ends.columns:
            return False, "round_end events are missing"
        valid_round_ends = round_ends.filter(round_ends["tick"] > 0)
        if len(valid_round_ends) < MIN_VALID_ROUNDS:
            return False, f"only {len(valid_round_ends)} valid round_end event(s)"
    except Exception as exc:
        return False, str(exc)
    return True, ""


def main():
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now() - timedelta(hours=MAX_FILE_AGE_HOURS)

    for rar_path in DOWNLOADS_DIR.glob("*.rar"):
        if datetime.fromtimestamp(rar_path.stat().st_mtime) < cutoff:
            continue

        print(f"[process] {rar_path.name}")
        temp_dir = DOWNLOADS_DIR / "__temp__"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir()

        result = subprocess.run(
            [str(SEVEN_ZIP), "x", str(rar_path), f"-o{temp_dir}", "-y"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[skip] failed to extract {rar_path.name}: {result.stderr or result.stdout}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        event_name = strip_random_ids(rar_path.stem)
        moved = 0
        handled = 0

        for dem_path in temp_dir.rglob("*.dem"):
            base_name = f"{event_name}_{get_map_name(dem_path)}"
            valid, reason = validate_demo(dem_path)
            if not valid:
                print(f"[skip] invalid demo {dem_path.name}: {reason}")
                continue

            existing_matches = [
                path
                for path in DEMOS_DIR.glob("*.dem")
                if path.stem == base_name or path.stem.startswith(f"{base_name}_")
            ]
            duplicate = next(
                (path for path in existing_matches if same_file_content(dem_path, path)),
                None,
            )
            if duplicate is not None:
                print(f"[duplicate] {dem_path.name} matches {duplicate.name}")
                handled += 1
                continue

            dest_path = DEMOS_DIR / f"{base_name}.dem"
            counter = 2
            while dest_path.exists():
                dest_path = DEMOS_DIR / f"{base_name}_{counter}.dem"
                counter += 1

            shutil.move(str(dem_path), str(dest_path))
            print(f"[move] {dem_path.name} -> {dest_path.name}")
            moved += 1
            handled += 1

        if handled:
            rar_path.unlink()
            print(f"[delete] {rar_path.name} ({moved} new demo(s))")
        else:
            print(f"[keep] no demos found in {rar_path.name}")

        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[done]")


if __name__ == "__main__":
    main()
