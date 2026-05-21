from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Global font sizes — scaled up for report-quality output
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 17,
})

INPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")
SIDES = ["ct", "t"]
TOP_N_PER_ALGORITHM = 2  # best N models per (side, algorithm) to plot

RADAR_FEATURES = [
    "opening_kill_rate",
    "opening_duel_success",
    "trade_kill_rate",
    "death_traded_rate",
    "flash_assists_per_round",
    "util_damage_per_round",
    "awp_kill_share",
    "rifle_kill_share",
    "multi_kill_rate",
    "grenades_per_round",
]


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def get_top_models(side: str, top_n: int = TOP_N_PER_ALGORITHM) -> list[str]:
    """
    Read outputs/{side}/model_scores.csv and return the names of the top N
    models per algorithm, ranked by composite_score descending.
    """
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        print(f"[skip] missing {scores_path}")
        return []

    df = pd.read_csv(scores_path)
    if df.empty:
        print(f"[skip] {scores_path} is empty")
        return []

    selected = []
    for method, group in df.groupby("method"):
        top = (
            group.dropna(subset=["composite_score"])
            .sort_values("composite_score", ascending=False)
            .head(top_n)
        )
        names = top["name"].tolist()
        selected.extend(names)
        print(f"  [{side.upper()}] {method}: {names}")

    return selected


# ---------------------------------------------------------------------------
# Per-cluster summary (replaces the old summaries/ CSV dir)
# ---------------------------------------------------------------------------

def build_cluster_summary(players: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cluster mean of all numeric columns (excluding noise cluster -1).
    Returns one row per cluster with a 'cluster' column.
    """
    numeric_cols = players.select_dtypes(include="number").columns.tolist()
    exclude = {"pc1", "pc2", "cluster"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    non_noise = players[players["cluster"] != -1].copy()
    if non_noise.empty or "cluster" not in non_noise.columns:
        return pd.DataFrame()

    return non_noise.groupby("cluster")[feature_cols].mean().reset_index()


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_pca_scatter(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    required = {"player_name", "pc1", "pc2", "cluster"}
    missing = required - set(players.columns)
    if missing:
        print(f"[skip] {method_name} scatter — missing columns: {sorted(missing)}")
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
            s=80,
            zorder=3,
        )
        # Cluster legend
        handles, labels = scatter.legend_elements(prop="colors")
        cluster_ids = sorted(non_noise["cluster"].unique())
        ax.legend(handles, [f"Cluster {c}" for c in cluster_ids],
                  title="Cluster", loc="upper right")

    if not noise.empty:
        ax.scatter(
            noise["pc1"],
            noise["pc2"],
            s=140,
            marker="x",
            linewidths=2,
            color="grey",
            label="Noise / outliers",
            zorder=4,
        )

    for _, row in players.iterrows():
        ax.annotate(
            row["player_name"],
            (row["pc1"], row["pc2"]),
            fontsize=11,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA Scatter — {method_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{method_name}_pca.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


def plot_rating_box(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if "cluster" not in players.columns:
        print(f"[skip] {method_name} rating box — missing cluster column")
        return

    # rating column dropped from final CSV; fall back to adr as proxy
    rating_col = next(
        (c for c in ["adr", "damage_per_round"] if c in players.columns),
        None,
    )
    if rating_col is None:
        print(f"[skip] {method_name} rating box — no rating/adr column found")
        return

    plot_df = players[["cluster", rating_col]].copy()
    plot_df[rating_col] = pd.to_numeric(plot_df[rating_col], errors="coerce")
    plot_df = plot_df.dropna()

    non_noise = plot_df[plot_df["cluster"] != -1]
    noise = plot_df[plot_df["cluster"] == -1]

    cluster_order = sorted(non_noise["cluster"].unique())
    tick_labels = [f"Cluster {c}" for c in cluster_order]
    data = [non_noise.loc[non_noise["cluster"] == c, rating_col].tolist() for c in cluster_order]

    if not noise.empty:
        tick_labels.append("Noise")
        data.append(noise[rating_col].tolist())

    if not data:
        print(f"[skip] {method_name} rating box — no data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    box = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    for patch in box["boxes"]:
        patch.set_alpha(0.5)

    ax.set_xlabel("Cluster")
    ax.set_ylabel(rating_col)
    ax.set_title(f"ADR by Cluster — {method_name}")
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    out = output_dir / f"{method_name}_rating_box.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


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
        fontsize=12,
    )
    ax.set_yticklabels([])
    ax.set_title(title, y=1.12, fontsize=13)


def plot_radar(summary: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    if summary.empty or "cluster" not in summary.columns:
        print(f"[skip] {method_name} radar — empty or missing cluster column")
        return

    radar_features = [f for f in RADAR_FEATURES if f in summary.columns]
    if len(radar_features) < 3:
        print(f"[skip] {method_name} radar — fewer than 3 matching features")
        return

    radar_df = summary.copy()
    for c in radar_features:
        radar_df[c] = pd.to_numeric(radar_df[c], errors="coerce")

    mins = radar_df[radar_features].min()
    maxs = radar_df[radar_features].max()
    denom = (maxs - mins).replace(0, 1)
    radar_scaled = radar_df.copy()
    radar_scaled[radar_features] = (radar_scaled[radar_features] - mins) / denom

    num_clusters = len(radar_scaled)
    if num_clusters == 0:
        return

    ncols = 2 if num_clusters > 1 else 1
    nrows = int(np.ceil(num_clusters / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
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

    fig.suptitle(f"Cluster Profiles — {method_name}", fontsize=16, y=1.01)
    plt.tight_layout()
    out = output_dir / f"{method_name}_radar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def plot_stability_bar(side: str, output_dir: Path) -> None:
    """
    Horizontal bar chart of stability_mean ± stability_std for all models
    on a given side, coloured by algorithm.
    """
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        return

    df = pd.read_csv(scores_path).dropna(subset=["stability_mean"])
    if df.empty:
        return

    df = df.sort_values("stability_mean", ascending=True)

    method_colors = {"kmeans": "#4C72B0", "gmm": "#DD8452"}
    colors = [method_colors.get(m, "#888888") for m in df["method"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    ax.barh(df["name"], df["stability_mean"], xerr=df["stability_std"],
            color=colors, alpha=0.8, capsize=3)
    ax.axvline(x=0.50, linestyle="--", color="grey", linewidth=1, label="Min threshold (0.50)")
    ax.axvline(x=0.80, linestyle=":", color="green", linewidth=1, label="High tier (0.80)")
    ax.set_xlabel("Mean ARI (± 1 std)")
    ax.set_title(f"Bootstrap Stability — {side.upper()}")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=12)

    # Legend for algorithm colours
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=m) for m, c in method_colors.items()]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0][-2:],
              fontsize=12, loc="lower right")

    plt.tight_layout()
    out = output_dir / f"{side}_stability_bar.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


def plot_silhouette_vs_k(side: str, output_dir: Path) -> None:
    scores_path = INPUT_DIR / side / "model_scores.csv"
    if not scores_path.exists():
        return

    df = pd.read_csv(scores_path).dropna(subset=["silhouette", "k"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for method, group in df.groupby("method"):
        sub = group.sort_values("k")
        ax.plot(sub["k"], sub["silhouette"], marker="o", label=method)

    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(f"Silhouette vs k — {side.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{side}_silhouette_vs_k.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Cluster dominance: per-cluster mean table + z-score heatmap
# ---------------------------------------------------------------------------

# Features to exclude from dominance analysis (positional/meta, not role-defining)
_DOMINANCE_EXCLUDE = {"pc1", "pc2", "cluster", "rounds_played"}
# Std columns add noise to the mean table; keep them for the heatmap only
_STD_SUFFIX = "_std"

TOP_FEATURES_PER_CLUSTER = 8  # rows shown in the console mean table


def print_cluster_means(summary: pd.DataFrame, method_name: str) -> None:
    """
    Print a side-by-side table of per-cluster means for the most
    discriminating features (highest variance across clusters).
    Shows TOP_FEATURES_PER_CLUSTER features ranked by inter-cluster std.
    """
    if summary.empty or "cluster" not in summary.columns:
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE and not c.endswith(_STD_SUFFIX)
    ]
    if not feat_cols:
        return

    sub = summary.set_index("cluster")[feat_cols]

    # Rank features by how much they vary across clusters
    inter_std = sub.std(axis=0).sort_values(ascending=False)
    top_feats = inter_std.head(TOP_FEATURES_PER_CLUSTER).index.tolist()

    table = sub[top_feats].T
    table.index.name = "feature"

    print(f"\nCluster means — {method_name} (top {TOP_FEATURES_PER_CLUSTER} discriminating features):")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))


def plot_zscore_heatmap(
    summary: pd.DataFrame,
    players: pd.DataFrame,
    method_name: str,
    output_dir: Path,
) -> None:
    """
    Z-score heatmap: each cell is how many global std-devs above/below the
    population mean that cluster sits for each feature.

    - Rows = features (sorted by absolute max z-score so the most
      cluster-defining features float to the top)
    - Columns = clusters
    - Colour: diverging (red = above average, blue = below average)
    """
    if summary.empty or "cluster" not in summary.columns:
        print(f"[skip] {method_name} heatmap — empty summary")
        return

    feat_cols = [
        c for c in summary.columns
        if c not in _DOMINANCE_EXCLUDE
    ]
    if not feat_cols:
        return

    # Global stats from the full player population (non-noise only)
    non_noise = players[players["cluster"] != -1]
    available = [c for c in feat_cols if c in non_noise.columns]
    if not available:
        return

    global_mean = non_noise[available].mean()
    global_std = non_noise[available].std().replace(0, np.nan)

    cluster_means = summary.set_index("cluster")[available]
    z = (cluster_means - global_mean) / global_std
    z = z.dropna(axis=1, how="all")

    if z.empty:
        return

    # Sort features by max absolute z across clusters (most distinctive first)
    z = z.loc[:, z.abs().max(axis=0).sort_values(ascending=False).index]

    # Cap columns to avoid an unreadably wide plot
    MAX_FEATURES = 40
    z = z.iloc[:, :MAX_FEATURES]

    z_plot = z.T  # features as rows, clusters as columns

    fig_h = max(6, len(z_plot) * 0.35)
    fig_w = max(5, len(z_plot.columns) * 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = min(3.0, z_plot.abs().max().max())
    im = ax.imshow(z_plot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(z_plot.columns)))
    ax.set_xticklabels([f"Cluster {c}" for c in z_plot.columns], fontsize=13)
    ax.set_yticks(range(len(z_plot.index)))
    ax.set_yticklabels(z_plot.index, fontsize=12)

    # Annotate each cell with the z value
    for row_i, feat in enumerate(z_plot.index):
        for col_i, clust in enumerate(z_plot.columns):
            val = z_plot.loc[feat, clust]
            if pd.notna(val):
                ax.text(col_i, row_i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=11, color="black" if abs(val) < 1.5 else "white")

    plt.colorbar(im, ax=ax, label="Z-score (std devs from population mean)")
    ax.set_title(f"Feature Z-scores by Cluster — {method_name}\n"
                 f"(sorted by discriminating power, top {MAX_FEATURES} features)")
    plt.tight_layout()
    out = output_dir / f"{method_name}_zscore_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")




# ---------------------------------------------------------------------------
# Ground-truth comparison plots
# ---------------------------------------------------------------------------
def _is_valid_role(role: str) -> bool:
    return str(role).strip().lower() not in ("", "unknown", "nan")


def resolve_gt_label_for_plot(players: pd.DataFrame) -> pd.Series:
    """
    Mirror of cluster_players.resolve_gt_label — pick whichever of a player's
    two GT roles best matches their cluster's majority, so both roles are
    equally weighted when colouring and cross-tabulating.
    """
    from collections import Counter
    role_majority: dict[int, str] = {}
    non_noise = players[players["cluster"] != -1]
    for cid, group in non_noise.groupby("cluster"):
        counts: Counter = Counter()
        for col in ("gt_role_1", "gt_role_2"):
            if col not in group.columns:
                continue
            for r in group[col]:
                if _is_valid_role(r):
                    counts[r] += 1
        role_majority[cid] = counts.most_common(1)[0][0] if counts else ""

    def pick(row):
        cid = row.get("cluster", -1)
        if cid == -1:
            return ""
        top = role_majority.get(cid, "")
        r1 = str(row.get("gt_role_1", ""))
        r2 = str(row.get("gt_role_2", ""))
        if r1.strip().lower() == top.strip().lower() and _is_valid_role(r1):
            return r1
        if r2.strip().lower() == top.strip().lower() and _is_valid_role(r2):
            return r2
        return r1 if _is_valid_role(r1) else (r2 if _is_valid_role(r2) else "")

    return players.apply(pick, axis=1)



GT_ROLE_COLORS = {
    "AWPer":         "#e63946",
    "IGL":           "#457b9d",
    "Entry Fragger": "#f4a261",
    "Lurker":        "#2a9d8f",
    "Rifler":        "#a8dadc",
    "Coach":         "#6d6875",
    "Analyst":       "#6d6875",
    "Caster":        "#6d6875",
    "Unknown":       "#cccccc",
    "":              "#cccccc",
}


def plot_pca_gt_roles(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    """
    PCA scatter coloured by Liquipedia ground-truth role_1 instead of cluster.
    Cluster boundaries are drawn as faint convex hulls so you can see alignment.
    """
    required = {"player_name", "pc1", "pc2", "cluster", "gt_role_1"}
    if not required.issubset(players.columns):
        missing = required - set(players.columns)
        print(f"[skip] {method_name} GT scatter — missing: {sorted(missing)}")
        return

    non_noise = players[players["cluster"] != -1].copy()
    if non_noise.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw faint cluster convex hulls
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import ConvexHull
    cluster_ids = sorted(non_noise["cluster"].unique())
    hull_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_ids), 1)))
    for ci, cid in enumerate(cluster_ids):
        pts = non_noise[non_noise["cluster"] == cid][["pc1", "pc2"]].values
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
            poly = MplPolygon(pts[hull.vertices], closed=True,
                              facecolor=hull_colors[ci], alpha=0.08,
                              edgecolor=hull_colors[ci], linewidth=1.2,
                              linestyle="--", label=f"Cluster {cid} boundary")
            ax.add_patch(poly)
        except Exception:
            pass

    # Scatter coloured by GT role
    non_noise = non_noise.copy()
    non_noise["_gt_resolved"] = resolve_gt_label_for_plot(non_noise)
    for role, group in non_noise.groupby("_gt_resolved", sort=False):
        color = GT_ROLE_COLORS.get(role, "#888888")
        ax.scatter(group["pc1"], group["pc2"],
                   color=color, s=90, zorder=3, label=role, edgecolors="white", linewidths=0.5)

    # Noise players in grey
    noise = players[players["cluster"] == -1]
    if not noise.empty:
        ax.scatter(noise["pc1"], noise["pc2"], s=120, marker="x",
                   linewidths=2, color="grey", label="Noise", zorder=4)

    for _, row in players.iterrows():
        ax.annotate(row["player_name"], (row["pc1"], row["pc2"]),
                    fontsize=11, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA — Ground-Truth Roles vs Cluster Boundaries\n{method_name}")
    ax.legend(title="GT Role", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"{method_name}_pca_gt_roles.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def plot_cluster_role_heatmap(players: pd.DataFrame, method_name: str, output_dir: Path) -> None:
    """
    Heatmap: rows = clusters, columns = GT role_1.
    Cell values = player count; cells are also row-normalised (% of cluster)
    so dominant roles in each cluster stand out even when cluster sizes differ.
    """
    required = {"cluster", "gt_role_1"}
    if not required.issubset(players.columns):
        print(f"[skip] {method_name} role heatmap — missing GT columns")
        return

    df = players[players["cluster"] != -1].copy()
    df["_gt_resolved"] = resolve_gt_label_for_plot(df)
    df = df[df["_gt_resolved"].apply(_is_valid_role)]
    if df.empty:
        print(f"[skip] {method_name} role heatmap — no matched GT players")
        return

    ct = pd.crosstab(df["cluster"], df["_gt_resolved"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)  # row-normalise → fraction per cluster

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ct.columns) * 1.6 + 4),
                                            max(4, len(ct) * 0.8 + 2)))

    for ax, data, fmt, title_suffix in zip(
        axes,
        [ct_norm, ct],
        [".1%", "d"],
        ["Row % (fraction of cluster)", "Count"],
    ):
        im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=12)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels([f"Cluster {c}" for c in data.index], fontsize=12)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title_suffix}\n{method_name}", fontsize=14)

        for ri in range(data.shape[0]):
            for ci in range(data.shape[1]):
                val = data.values[ri, ci]
                txt = format(val, fmt) if fmt == ".1%" else str(int(val))
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=11, color="black" if val < 0.6 else "white")

    plt.suptitle(f"Cluster × Ground-Truth Role — {method_name}", fontsize=16, y=1.02)
    plt.tight_layout()
    out = output_dir / f"{method_name}_cluster_role_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
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

        model_names = get_top_models(side, top_n=TOP_N_PER_ALGORITHM)
        if not model_names:
            print(f"[skip] no models found for {side}")
            continue

        for name in model_names:
            players_path = INPUT_DIR / side / f"{name}_player_clusters.csv"
            if not players_path.exists():
                print(f"[skip] missing {players_path}")
                continue

            players = pd.read_csv(players_path)
            if players.empty:
                print(f"[skip] {players_path} is empty")
                continue

            summary = build_cluster_summary(players)

            print_cluster_means(summary, name)
            plot_pca_scatter(players, name, side_plot_dir)
            plot_pca_gt_roles(players, name, side_plot_dir)
            plot_cluster_role_heatmap(players, name, side_plot_dir)
            plot_rating_box(players, name, side_plot_dir)
            plot_radar(summary, name, side_plot_dir)
            plot_zscore_heatmap(summary, players, name, side_plot_dir)

        plot_stability_bar(side, side_plot_dir)
        plot_silhouette_vs_k(side, side_plot_dir)

    print(f"\nSaved plots in ./{PLOTS_DIR}")


if __name__ == "__main__":
    main()