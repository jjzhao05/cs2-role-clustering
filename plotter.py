from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_config import MODEL_EXCLUDE_COLUMNS, RESULT_ARTIFACT_COLUMNS

# ---------------------------------------------------------------------------
# Global font sizes
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 17,
})

INPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")

SIDES = ["ct", "t"]

TOP_N_PER_ALGORITHM = 2

RADAR_FEATURES = [
    "opening_kill_rate",
    "opening_death_rate",
    "opening_duel_success",
    "first_contact_rate",
    "first_contact_received_rate",
    "trade_kill_rate",
    "death_traded_rate",
    "flash_assists_per_round",
    "util_damage_per_round",
    "awp_kill_share",
    "rifle_kill_share",
    "multi_kill_rate",
    "smokes_per_round",
]

_DOMINANCE_EXCLUDE = MODEL_EXCLUDE_COLUMNS | RESULT_ARTIFACT_COLUMNS

_STD_SUFFIX = "_std"

TOP_FEATURES_PER_CLUSTER = 8


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def get_top_models(side: str, top_n: int = TOP_N_PER_ALGORITHM):

    scores_path = INPUT_DIR / side / "model_scores.csv"

    if not scores_path.exists():
        print(f"[skip] missing {scores_path}")
        return []

    df = pd.read_csv(scores_path)

    if df.empty:
        return []

    selected = []

    for method, group in df.groupby("method"):

        top = (
            group
            .dropna(subset=["composite_score"])
            .sort_values(
                ["composite_score", "name"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(top_n)
        )

        names = top["name"].tolist()

        selected.extend(names)

        print(f"[{side.upper()}] {method}: {names}")

    return selected


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_cluster_summary(players: pd.DataFrame):

    numeric_cols = players.select_dtypes(include="number").columns.tolist()

    exclude = MODEL_EXCLUDE_COLUMNS | RESULT_ARTIFACT_COLUMNS

    feature_cols = [c for c in numeric_cols if c not in exclude]

    non_noise = players[players["cluster"] != -1].copy()

    if non_noise.empty:
        return pd.DataFrame()

    return (
        non_noise
        .groupby("cluster")[feature_cols]
        .mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# PCA scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(players, method_name, output_dir):

    required = {"player_name", "pc1", "pc2", "cluster"}

    if not required.issubset(players.columns):
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    non_noise = players[players["cluster"] != -1]
    noise = players[players["cluster"] == -1]

    if not non_noise.empty:

        scatter = ax.scatter(
            non_noise["pc1"],
            non_noise["pc2"],
            c=non_noise["cluster"],
            cmap="tab10",
            s=90,
        )

        handles, labels = scatter.legend_elements(prop="colors")

        cluster_ids = sorted(non_noise["cluster"].unique())

        ax.legend(
            handles,
            [f"Cluster {c}" for c in cluster_ids],
            title="Cluster",
            loc="upper right",
        )

    if not noise.empty:

        ax.scatter(
            noise["pc1"],
            noise["pc2"],
            s=130,
            marker="x",
            linewidths=2,
            color="grey",
            label="Noise",
        )

    for _, row in players.iterrows():

        ax.annotate(
            row["player_name"],
            (row["pc1"], row["pc2"]),
            fontsize=10,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_title(f"PCA Scatter: {method_name}")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = output_dir / f"{method_name}_pca.png"

    plt.savefig(out, dpi=250)

    plt.close()

    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------

def _make_radar(ax, values, categories, title):

    n = len(categories)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()

    vals = values.tolist() + values.tolist()[:1]

    angles = angles + angles[:1]

    ax.plot(angles, vals, linewidth=2)

    ax.fill(angles, vals, alpha=0.25)

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(
        [c.replace("_", "\n") for c in categories],
        fontsize=11,
    )

    ax.set_yticklabels([])

    ax.set_title(title, y=1.10)


def plot_radar(summary, method_name, output_dir):

    if summary.empty or "cluster" not in summary.columns:
        return

    radar_features = [
        f for f in RADAR_FEATURES
        if f in summary.columns
    ]

    if len(radar_features) < 3:
        return

    radar_df = summary.copy()

    for c in radar_features:
        radar_df[c] = pd.to_numeric(radar_df[c], errors="coerce")

    mins = radar_df[radar_features].min()
    maxs = radar_df[radar_features].max()

    denom = (maxs - mins).replace(0, 1)

    radar_scaled = radar_df.copy()

    radar_scaled[radar_features] = (
        radar_scaled[radar_features] - mins
    ) / denom

    num_clusters = len(radar_scaled)

    ncols = 2 if num_clusters > 1 else 1
    nrows = int(np.ceil(num_clusters / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 5 * nrows),
        subplot_kw=dict(polar=True),
    )

    axes = np.array(axes).reshape(-1)

    for i, (_, row) in enumerate(radar_scaled.iterrows()):

        _make_radar(
            axes[i],
            row[radar_features],
            radar_features,
            f"Cluster {int(row['cluster'])}",
        )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(
        f"Cluster Profiles: {method_name}",
        fontsize=16,
    )

    plt.tight_layout()

    out = output_dir / f"{method_name}_radar.png"

    plt.savefig(
        out,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance_bar(
    summary,
    players,
    method_name,
    output_dir,
    top_n=25,
):

    if summary.empty or "cluster" not in summary.columns:
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE
    ]

    non_noise = players[players["cluster"] != -1]

    available = [
        c for c in feat_cols
        if c in non_noise.columns
    ]

    if not available:
        return

    # Cluster means
    cluster_means = (
        summary
        .set_index("cluster")[available]
        .apply(pd.to_numeric, errors="coerce")
    )

    # Population std from all players
    population_std = (
        non_noise[available]
        .std()
        .replace(0, np.nan)
    )

    # Importance = spread of cluster means
    # normalized by actual player variance
    importance = (
        cluster_means.std(axis=0) / population_std
    )

    importance = (
        importance
        .dropna()
        .sort_values(ascending=False)
    )

    if importance.empty:
        return

    importance = (
        importance
        .head(top_n)
        .sort_values(ascending=True)
    )

    fig_h = max(6, len(importance) * 0.35)

    fig, ax = plt.subplots(figsize=(12, fig_h))

    bars = ax.barh(
        importance.index,
        importance.values,
        alpha=0.85,
    )

    ax.set_xlabel(
        "Normalized feature importance\n(cluster separation / population std)"
    )

    ax.set_ylabel("Feature")

    ax.set_title(
        f"Feature Importance: {method_name}\n"
        f"Top {top_n} discriminating features"
    )

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

    out = output_dir / f"{method_name}_feature_importance.png"

    plt.savefig(
        out,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def plot_zscore_heatmap(
    summary,
    players,
    method_name,
    output_dir,
):

    if summary.empty or "cluster" not in summary.columns:
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE
    ]

    non_noise = players[players["cluster"] != -1]

    available = [
        c for c in feat_cols
        if c in non_noise.columns
    ]

    if not available:
        return

    global_mean = non_noise[available].mean()

    global_std = (
        non_noise[available]
        .std()
        .replace(0, np.nan)
    )

    cluster_means = summary.set_index("cluster")[available]

    z = (cluster_means - global_mean) / global_std

    z = z.dropna(axis=1, how="all")

    if z.empty:
        return

    z = z.loc[
        :,
        z.abs()
        .max(axis=0)
        .sort_values(ascending=False)
        .index
    ]

    MAX_FEATURES = 40

    z = z.iloc[:, :MAX_FEATURES]

    z_plot = z.T

    n_features = len(z_plot)
    n_clusters = len(z_plot.columns)

    fig_h = max(12, n_features * 0.42)
    fig_w = max(10, n_clusters * 2.2 + 5)

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h)
    )

    vmax = min(
        3.0,
        float(z_plot.abs().max().max())
    )

    im = ax.imshow(
        z_plot.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(n_clusters))

    ax.set_xticklabels(
        [f"Cluster {c}" for c in z_plot.columns],
        fontsize=13,
    )

    ax.set_yticks(np.arange(n_features))

    ax.set_yticklabels(
        z_plot.index,
        fontsize=11,
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")

    for row_i, feat in enumerate(z_plot.index):

        for col_i, clust in enumerate(z_plot.columns):

            val = z_plot.loc[feat, clust]

            if pd.notna(val):

                ax.text(
                    col_i,
                    row_i,
                    f"{val:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(val) > vmax * 0.55 else "black",
                )

    ax.set_xticks(
        np.arange(-0.5, n_clusters, 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, n_features, 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="lightgrey",
        linestyle="-",
        linewidth=0.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.04,
    )

    cbar.set_label(
        "Z-score\n(std devs from population mean)"
    )

    ax.set_title(
        f"Feature Z-scores by Cluster: {method_name}\n"
        f"Top {MAX_FEATURES} features by discriminating power"
    )

    plt.tight_layout()

    out = output_dir / f"{method_name}_zscore_heatmap.png"

    plt.savefig(
        out,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    PLOTS_DIR.mkdir(exist_ok=True)

    for side in SIDES:

        side_plot_dir = PLOTS_DIR / side

        side_plot_dir.mkdir(exist_ok=True)

        print(f"\n=== {side.upper()} ===")

        model_names = get_top_models(side)

        if not model_names:
            continue

        for name in model_names:

            players_path = (
                INPUT_DIR
                / side
                / f"{name}_player_clusters.csv"
            )

            if not players_path.exists():
                continue

            players = pd.read_csv(players_path)

            if players.empty:
                continue

            summary = build_cluster_summary(players)

            plot_pca_scatter(
                players,
                name,
                side_plot_dir,
            )

            plot_radar(
                summary,
                name,
                side_plot_dir,
            )

            plot_zscore_heatmap(
                summary,
                players,
                name,
                side_plot_dir,
            )

            plot_feature_importance_bar(
                summary,
                players,
                name,
                side_plot_dir,
            )

    print(f"\nSaved plots in ./{PLOTS_DIR}")


if __name__ == "__main__":
    main()
