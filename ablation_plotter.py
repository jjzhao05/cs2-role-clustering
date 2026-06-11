from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("outputs/ablations")
PLOTS_DIR = Path("plots/ablations")

SIDES = ["ct", "t"]

METRIC_COLS = [
    "composite_score",
    "gt_ari",
    "gt_purity",
    "stability_mean",
    "silhouette",
]

DISPLAY_NAMES = {
    "composite_score": "Composite Score",
    "gt_ari": "Ground-Truth ARI",
    "gt_purity": "Ground-Truth Purity",
    "stability_mean": "Bootstrap Stability",
    "silhouette": "Silhouette Score",
}


def clean_ablation_name(name: str) -> str:
    return (
        name.replace("_", " ")
        .replace("positioning movement", "positioning")
        .replace("opening aggression", "opening")
        .replace("trading teamwork", "trading")
        .title()
    )


def load_best_results(side: str) -> pd.DataFrame:
    path = INPUT_DIR / side / "ablation_results_best.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ablation_clusters.py first.")

    df = pd.read_csv(path)

    df["ablation_display"] = df["ablation"].apply(clean_ablation_name)

    return df


def plot_metric_bar(df: pd.DataFrame, side: str, metric: str) -> None:
    work = df.dropna(subset=[metric]).copy()

    if work.empty:
        return

    work = work.sort_values(metric, ascending=True)

    fig_h = max(6, len(work) * 0.45)

    fig, ax = plt.subplots(figsize=(12, fig_h))

    bars = ax.barh(work["ablation_display"], work[metric])

    ax.set_xlabel(DISPLAY_NAMES.get(metric, metric))
    ax.set_ylabel("Ablation")
    ax.set_title(f"{side.upper()} Ablation Results — {DISPLAY_NAMES.get(metric, metric)}")

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.3f}",
            va="center",
            fontsize=10,
        )

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    out_dir = PLOTS_DIR / side
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{side}_{metric}_bar.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def plot_metric_heatmap(df: pd.DataFrame, side: str) -> None:
    available_metrics = [m for m in METRIC_COLS if m in df.columns]

    work = df[["ablation_display"] + available_metrics].copy()

    for col in available_metrics:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(how="all", subset=available_metrics)

    if work.empty:
        return

    work = work.sort_values("composite_score", ascending=False)

    values = work[available_metrics].to_numpy(dtype=float)

    fig_h = max(6, len(work) * 0.45)
    fig_w = max(10, len(available_metrics) * 2.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(values, aspect="auto")

    ax.set_xticks(np.arange(len(available_metrics)))
    ax.set_xticklabels(
        [DISPLAY_NAMES.get(m, m) for m in available_metrics],
        rotation=30,
        ha="right",
    )

    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work["ablation_display"])

    ax.set_title(f"{side.upper()} Ablation Metric Heatmap")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric Value")

    plt.tight_layout()

    out_dir = PLOTS_DIR / side
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{side}_metric_heatmap.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def plot_k_selection(df: pd.DataFrame, side: str) -> None:
    work = df.copy()
    work = work.sort_values("composite_score", ascending=True)

    fig_h = max(6, len(work) * 0.45)

    fig, ax = plt.subplots(figsize=(12, fig_h))

    bars = ax.barh(work["ablation_display"], work["k"])

    ax.set_xlabel("Best k")
    ax.set_ylabel("Ablation")
    ax.set_title(f"{side.upper()} Best k by Ablation")

    ax.set_xticks(sorted(work["k"].dropna().astype(int).unique()))

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" k={int(width)}",
            va="center",
            fontsize=10,
        )

    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    out_dir = PLOTS_DIR / side
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{side}_best_k.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def plot_feature_count_vs_score(df: pd.DataFrame, side: str) -> None:
    required = {"n_features", "composite_score", "ablation_display"}

    if not required.issubset(df.columns):
        return

    work = df.dropna(subset=["n_features", "composite_score"]).copy()

    if work.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(work["n_features"], work["composite_score"], s=90)

    for _, row in work.iterrows():
        ax.annotate(
            row["ablation_display"],
            (row["n_features"], row["composite_score"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Composite Score")
    ax.set_title(f"{side.upper()} Feature Count vs Composite Score")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir = PLOTS_DIR / side
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{side}_feature_count_vs_score.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def plot_all_k_curves(side: str) -> None:
    path = INPUT_DIR / side / "ablation_results_all.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if "composite_score" not in df.columns:
        return

    df["ablation_display"] = df["ablation"].apply(clean_ablation_name)

    fig, ax = plt.subplots(figsize=(14, 9))

    for ablation, group in df.groupby("ablation_display"):
        group = group.sort_values("k")

        ax.plot(
            group["k"],
            group["composite_score"],
            marker="o",
            linewidth=1.5,
            label=ablation,
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Composite Score")
    ax.set_title(f"{side.upper()} Composite Score Across k")

    ax.grid(True, alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        fontsize=9,
    )

    plt.tight_layout()

    out_dir = PLOTS_DIR / side
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{side}_composite_by_k.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def plot_side_comparison() -> None:
    frames = []

    for side in SIDES:
        path = INPUT_DIR / side / "ablation_results_best.csv"

        if path.exists():
            df = pd.read_csv(path)
            df["side"] = side.upper()
            df["ablation_display"] = df["ablation"].apply(clean_ablation_name)
            frames.append(df)

    if len(frames) < 2:
        return

    combined = pd.concat(frames, ignore_index=True)

    pivot = combined.pivot_table(
        index="ablation_display",
        columns="side",
        values="composite_score",
        aggfunc="mean",
    )

    pivot = pivot.dropna()

    if pivot.empty:
        return

    pivot = pivot.sort_values("T", ascending=True) if "T" in pivot.columns else pivot

    x = np.arange(len(pivot))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, max(7, len(pivot) * 0.45)))

    ax.barh(x - width / 2, pivot["CT"], height=width, label="CT")
    ax.barh(x + width / 2, pivot["T"], height=width, label="T")

    ax.set_yticks(x)
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Composite Score")
    ax.set_ylabel("Ablation")
    ax.set_title("CT vs T Ablation Comparison")

    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    out = PLOTS_DIR / "ct_vs_t_composite_comparison.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] {out}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for side in SIDES:
        print(f"\n=== {side.upper()} Ablation Plots ===")

        df = load_best_results(side)

        for metric in METRIC_COLS:
            if metric in df.columns:
                plot_metric_bar(df, side, metric)

        plot_metric_heatmap(df, side)
        plot_k_selection(df, side)
        plot_feature_count_vs_score(df, side)
        plot_all_k_curves(side)

    plot_side_comparison()

    print(f"\n[done] ablation plots saved in {PLOTS_DIR}")


if __name__ == "__main__":
    main()