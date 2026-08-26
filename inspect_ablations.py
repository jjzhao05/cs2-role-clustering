from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


INPUT_PATH = "output.csv"
ABLATION_RESULTS_DIR = Path("outputs/ablations")
OUTPUT_DIR = Path("outputs/ablations/cluster_inspection")
PLOTS_DIR = Path("plots/ablations/cluster_inspection")

RANDOM_STATE = 101705


def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH).fillna(0)
    df["side"] = df["side"].str.lower()
    return df[df["side"].isin(["ct", "t"])].copy()


def get_best_ablation_row(side: str, ablation: str) -> pd.Series:
    path = ABLATION_RESULTS_DIR / side / "ablation_results_best.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run ablation_clusters.py first.")

    results = pd.read_csv(path)

    match = results[results["ablation"] == ablation]

    if match.empty:
        available = sorted(results["ablation"].unique())
        raise ValueError(
            f"Ablation '{ablation}' not found for {side}. "
            f"Available: {available}"
        )

    return match.iloc[0]


def parse_feature_list(row: pd.Series) -> list[str]:
    features = str(row["features"]).split(", ")
    return [f for f in features if f]


def run_selected_ablation(side: str, ablation: str) -> pd.DataFrame:
    df = load_data()
    side_df = df[df["side"] == side].reset_index(drop=True)

    row = get_best_ablation_row(side, ablation)
    k = int(row["k"])
    features = parse_feature_list(row)

    features = [f for f in features if f in side_df.columns]

    if len(features) < 2:
        raise ValueError(f"Too few valid features for {side} {ablation}: {features}")

    X = side_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)

    out = side_df.copy()
    out["cluster"] = labels
    out["pc1"] = coords[:, 0]
    out["pc2"] = coords[:, 1]
    out["ablation"] = ablation
    out["k"] = k

    side_out_dir = OUTPUT_DIR / side
    side_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = side_out_dir / f"{ablation}_clusters.csv"
    out.to_csv(out_path, index=False)

    summary = build_cluster_summary(out, X.columns.tolist())
    summary_path = side_out_dir / f"{ablation}_cluster_summary.csv"
    summary.to_csv(summary_path, index=False)

    plot_ablation_clusters(out, side, ablation)

    print(f"[ok] cluster assignments: {out_path}")
    print(f"[ok] cluster summary:     {summary_path}")

    return out


def build_cluster_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []

    for cluster_id, group in df.groupby("cluster"):
        row = {
            "cluster": cluster_id,
            "n_players": len(group),
            "players": ", ".join(group["player_name"].astype(str).sort_values()),
        }

        for feature in features:
            row[f"{feature}_mean"] = group[feature].mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster")


def plot_ablation_clusters(df: pd.DataFrame, side: str, ablation: str) -> None:
    plot_dir = PLOTS_DIR / side
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = ax.scatter(
        df["pc1"],
        df["pc2"],
        c=df["cluster"],
        s=90,
        cmap="tab10",
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["player_name"],
            (row["pc1"], row["pc2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    handles, _ = scatter.legend_elements(prop="colors")
    cluster_ids = sorted(df["cluster"].unique())

    ax.legend(
        handles,
        [f"Cluster {c}" for c in cluster_ids],
        title="Cluster",
        loc="best",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"{side.upper()} Ablated Clusters: {ablation}")

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = plot_dir / f"{ablation}_clusters_pca.png"
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()

    print(f"[ok] PCA plot:            {out}")


def print_cluster_players(df: pd.DataFrame) -> None:
    for cluster_id, group in df.groupby("cluster"):
        players = group["player_name"].astype(str).sort_values().tolist()

        print(f"\nCluster {cluster_id} | n={len(players)}")
        print(", ".join(players))


def main() -> None:
    # Change these manually.
    side = "t"
    ablation = "no_weapon_share"

    df = run_selected_ablation(side, ablation)
    print_cluster_players(df)


if __name__ == "__main__":
    main()