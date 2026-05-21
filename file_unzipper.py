import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

DOWNLOADS_DIR = Path(r"C:\Users\Jonathan Zhao\Downloads")
DEMOS_DIR = Path(r"C:\Users\Jonathan Zhao\Documents\GitHub\cs2-role-classifier\demos")
MAX_FILE_AGE_HOURS = 6

# change this if your 7-Zip is installed elsewhere
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe") 


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_recent(path: Path) -> bool:
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    cutoff = datetime.now() - timedelta(hours=MAX_FILE_AGE_HOURS)
    return modified >= cutoff


def extract_rar(rar_path: Path, temp_dir: Path) -> bool:
    if not SEVEN_ZIP.exists():
        print(f"[error] 7z not found at: {SEVEN_ZIP}")
        return False

    try:
        result = subprocess.run(
            [str(SEVEN_ZIP), "x", str(rar_path), f"-o{temp_dir}", "-y"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[skip] failed to extract {rar_path.name}")
            if result.stderr.strip():
                print(result.stderr.strip())
            elif result.stdout.strip():
                print(result.stdout.strip())
            return False

        return True

    except Exception as e:
        print(f"[skip] failed to extract {rar_path.name}: {e}")
        return False


def move_demos_skip_duplicates(src: Path, dest: Path) -> int:
    moved = 0
    skipped = 0

    for root, _, files in os.walk(src):
        for f in files:
            if not f.lower().endswith(".dem"):
                continue

            src_path = Path(root) / f
            dest_path = dest / f

            if dest_path.exists():
                skipped += 1
                print(f"[skip] duplicate: {f}")
                continue

            shutil.move(str(src_path), str(dest_path))
            moved += 1
            print(f"[move] {f}")

    print(f"[summary] moved={moved}, skipped_duplicates={skipped}")
    return moved


def main():
    ensure_dir(DEMOS_DIR)

    rars = [
        p for p in DOWNLOADS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".rar" and is_recent(p)
    ]

    if not rars:
        print("[info] no recent rar files found")
        return

    for rar_path in rars:
        print(f"[process] {rar_path.name}")

        temp_dir = DOWNLOADS_DIR / "__temp__"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir()

        if not extract_rar(rar_path, temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            continue

        moved = move_demos_skip_duplicates(temp_dir, DEMOS_DIR)

        if moved > 0:
            rar_path.unlink()
            print(f"[delete] {rar_path.name}")
        else:
            print(f"[keep] no new demos in {rar_path.name}")

        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[done]")


if __name__ == "__main__":
    main()