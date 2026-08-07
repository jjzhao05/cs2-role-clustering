from pathlib import Path
import subprocess
import sys
import time


PYTHON = sys.executable
PROJECT_DIR = Path(__file__).resolve().parent

UNZIP_SCRIPT = PROJECT_DIR / "file_unzipper.py"
PARSER_SCRIPT = PROJECT_DIR / "demo_parser.py"
CLUSTER_SCRIPT = PROJECT_DIR / "cluster_players.py"
EVALUATE_SCRIPT = PROJECT_DIR / "evaluate_clusters.py"
LABEL_SCRIPT = PROJECT_DIR / "label_clusters.py"
PLOT_SCRIPT = PROJECT_DIR / "plotter.py"
ABLATION_SCRIPT = PROJECT_DIR / "run_ablations.py"

DEMOS_DIR = PROJECT_DIR / "demos"
OUTPUT_CSV = PROJECT_DIR / "output.csv"


def run_script(script_path, args=None):
    if args is None:
        args = []

    cmd = [PYTHON, str(script_path), *args]
    print(f"\n[run] {subprocess.list2cmdline(cmd)}\n")
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def time_step(name, func):
    print(f"\n[{name}] starting...")
    start = time.time()
    func()
    elapsed = time.time() - start
    print(f"[{name}] done in {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    return elapsed


def main():
    print("=== CS2 Pipeline Start ===")
    total_start = time.time()

    steps = [
        (
            "extract",
            "Extracting and deduplicating demos",
            lambda: run_script(UNZIP_SCRIPT),
        ),
        (
            "parse",
            "Parsing demos into feature dataset",
            lambda: run_script(PARSER_SCRIPT, [str(DEMOS_DIR), str(OUTPUT_CSV)]),
        ),
        (
            "cluster",
            "Running clustering algorithms",
            lambda: run_script(CLUSTER_SCRIPT),
        ),
        (
            "evaluate",
            "Evaluating clusters",
            lambda: run_script(EVALUATE_SCRIPT),
        ),
        (
            "label",
            "Generating surrogate labels",
            lambda: run_script(LABEL_SCRIPT),
        ),
        (
            "plot",
            "Generating visualizations",
            lambda: run_script(PLOT_SCRIPT),
        ),
        (
            "ablation",
            "Running ablation analysis",
            lambda: run_script(ABLATION_SCRIPT),
        ),
    ]

    timings = {}
    for key, name, func in steps:
        timings[key] = time_step(name, func)

    total_elapsed = time.time() - total_start
    print("\n=== Pipeline Complete ===")
    print("\n[time] Stage breakdown:")
    for name, elapsed in timings.items():
        print(f"  {name:<10}: {elapsed:8.2f}s ({elapsed / 60:.2f} min)")
    print(f"\n[time] Total runtime: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")


if __name__ == "__main__":
    main()
