import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

DOWNLOADS_DIR = Path(r"C:\Users\Jonathan Zhao\Downloads")
DEMOS_DIR = Path(r"C:\Users\Jonathan Zhao\Documents\GitHub\cs2-role-classifier\demos")
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
MAX_FILE_AGE_HOURS = 6


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

        for dem_path in temp_dir.rglob("*.dem"):
            base_name = f"{event_name}_{get_map_name(dem_path)}"
            dest_path = DEMOS_DIR / f"{base_name}.dem"
            counter = 2
            while dest_path.exists():
                dest_path = DEMOS_DIR / f"{base_name}_{counter}.dem"
                counter += 1

            shutil.move(str(dem_path), str(dest_path))
            print(f"[move] {dem_path.name} -> {dest_path.name}")
            moved += 1

        if moved:
            rar_path.unlink()
            print(f"[delete] {rar_path.name}")
        else:
            print(f"[keep] no demos found in {rar_path.name}")

        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[done]")


if __name__ == "__main__":
    main()