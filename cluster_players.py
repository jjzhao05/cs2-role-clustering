from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan

from feature_config import CLUSTER_RANDOM_STATE, MIN_ROUNDS_PLAYED, select_model_features

INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")
K_VALUES = [3, 4, 5, 6, 7, 8]
RANDOM_STATE = CLUSTER_RANDOM_STATE

# HDBSCAN hyperparameter grid
HDBSCAN_MIN_CLUSTER_SIZES = [2, 3, 4, 5, 6, 8, 10, 12, 15]
HDBSCAN_MIN_SAMPLES = [1, 2, 3, 4, 5, 6, 7, 8]
HDBSCAN_MAX_NOISE_FRACTION = 0.30

# Stability analysis config
STABILITY_N_BOOTS = 100
STABILITY_RESAMPLE_FRAC = 0.80
STABILITY_WEIGHT = 0.40
SILHOUETTE_WEIGHT = 0.40
DB_WEIGHT = 0.20


def load_data():
    df = pd.read_csv(INPUT_PATH).fillna(0)
    required = {"player_name", "side", "rounds_played"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["side"] = df["side"].str.lower()
    df = df[df["side"].isin(["ct", "t"])]
    low_rounds = df["rounds_played"] < MIN_ROUNDS_PLAYED
    if low_rounds.any():
        print(
            f"[info] excluding {int(low_rounds.sum())} player-side profiles with "
            f"fewer than {MIN_ROUNDS_PLAYED} rounds"
        )
        df = df[~low_rounds]
    if df.empty:
        raise ValueError("No rows left after filtering.")
    return df


def get_features(df):
    return select_model_features(df)


def clean_generated_outputs(side_dir: Path) -> None:
    """Remove artifacts owned by the pipeline before writing a fresh run."""
    patterns = [
        "model_scores.csv",
        "*_player_clusters.csv",
        "*_feature_importance.csv",
        "*_gt_side_accuracy.csv",
        "*_gt_general_accuracy.csv",
        "*_gt_igl_accuracy.csv",
    ]
    for pattern in patterns:
        for path in side_dir.glob(pattern):
            path.unlink()
    shutil.rmtree(side_dir / "surrogate_labels", ignore_errors=True)


def fit_labels(X, method, k=None, mcs=None, ms=None, random_state=RANDOM_STATE):
    """Fit one clustering method and return labels, or None if it produced <2 clusters."""
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        return model.fit_predict(X)

    if method == "gmm":
        model = GaussianMixture(n_components=k, random_state=random_state, covariance_type="full")
        model.fit(X)
        return model.predict(X)

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)
        if len(set(labels) - {-1}) >= 2:
            return labels
        return None

    raise ValueError(f"Unknown method: {method}")


def compute_stability(X, full_labels, method, k=None, mcs=None, ms=None,
                       n_boots=STABILITY_N_BOOTS, resample_frac=STABILITY_RESAMPLE_FRAC,
                       random_state=RANDOM_STATE):
    """Subsampling stability including both clustered and noise assignments."""
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    sample_size = max(3, int(n * resample_frac))
    scores = []
    noise_agreements = []

    for i in range(n_boots):
        idx = rng.choice(n, size=sample_size, replace=False)
        boot_labels = fit_labels(X[idx], method, k=k, mcs=mcs, ms=ms, random_state=random_state + i)
        if boot_labels is None:
            continue

        ref = full_labels[idx]
        if len(set(ref)) < 2 or len(set(boot_labels)) < 2:
            continue

        # Treat HDBSCAN noise (-1) as an assignment that must also reproduce.
        scores.append(adjusted_rand_score(ref, boot_labels))
        if method == "hdbscan":
            noise_agreements.append(float(np.mean((ref == -1) == (boot_labels == -1))))

    if len(scores) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if noise_agreements:
        noise_mean = float(np.mean(noise_agreements))
        noise_std = float(np.std(noise_agreements))
    else:
        noise_mean = noise_std = float("nan")
    return float(np.mean(scores)), float(np.std(scores)), noise_mean, noise_std


def composite_score(sil, db, stab_mean, stab_std):
    if np.isnan(sil):
        sil_term = 0.0
    else:
        sil_term = SILHOUETTE_WEIGHT * sil

    if np.isnan(db):
        db_term = 0.0
    else:
        db_term = DB_WEIGHT * (1.0 / (1.0 + db))

    if np.isnan(stab_mean):
        stab_term = 0.0
    else:
        stab_term = STABILITY_WEIGHT * max(0.0, stab_mean - stab_std)

    return sil_term + db_term + stab_term


def score_candidate(X, side, method, name, labels, k=None, mcs=None, ms=None):
    """Validate + score one clustering result. Returns a result dict, or None if invalid."""
    mask = labels != -1
    noise_pct = (~mask).mean()

    if method == "hdbscan" and noise_pct > HDBSCAN_MAX_NOISE_FRACTION:
        print(f"  [skip] {name} — {noise_pct:.0%} noise exceeds {HDBSCAN_MAX_NOISE_FRACTION:.0%} cap")
        return None
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return None

    sil = silhouette_score(X[mask], labels[mask])
    db = davies_bouldin_score(X[mask], labels[mask])
    stab_mean, stab_std, noise_stab_mean, noise_stab_std = compute_stability(
        X, labels, method, k=k, mcs=mcs, ms=ms
    )

    if k is not None:
        cluster_count = k
    else:
        cluster_count = len(set(labels) - {-1})

    return {
        "side": side,
        "method": method,
        "name": name,
        "k": cluster_count,
        "noise_pct": round(noise_pct, 4),
        "silhouette": sil,
        "davies_bouldin": db,
        "stability_mean": stab_mean,
        "stability_std": stab_std,
        "noise_assignment_stability_mean": noise_stab_mean,
        "noise_assignment_stability_std": noise_stab_std,
        "composite_score": composite_score(sil, db, stab_mean, stab_std),
    }


def cluster_side(df, side):
    side_dir = OUTPUT_DIR / side
    side_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_outputs(side_dir)

    side_df = df[df["side"] == side].reset_index(drop=True)
    if len(side_df) < 3:
        print(f"Skipping {side}: not enough players.")
        return

    X = get_features(side_df)
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    methods, results = {}, []

    # --- KMeans + GMM ---
    for k in K_VALUES:
        if k >= len(side_df):
            continue
        for method in ("kmeans", "gmm"):
            name = f"{method}_k{k}"
            labels = fit_labels(X_scaled, method, k=k)
            methods[name] = labels
            result = score_candidate(X_scaled, side, method, name, labels, k=k)
            if result is not None:
                results.append(result)

    # --- HDBSCAN grid search ---
    for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
        if mcs > len(side_df) // 2:
            continue
        for ms in HDBSCAN_MIN_SAMPLES:
            name = f"hdbscan_mcs{mcs}_ms{ms}"
            labels = fit_labels(X_scaled, "hdbscan", mcs=mcs, ms=ms)
            if labels is None:
                continue
            methods[name] = labels
            result = score_candidate(X_scaled, side, "hdbscan", name, labels, mcs=mcs, ms=ms)
            if result is not None:
                results.append(result)

    if not results:
        print(f"No valid clustering results for {side}.")
        return

    results_df = pd.DataFrame(results).sort_values(
        ["composite_score", "name"],
        ascending=[False, True],
        kind="mergesort",
    )

    # Write top 2 per algorithm
    for method, group in results_df.groupby("method"):
        for _, row in group.head(2).iterrows():
            name = row["name"]
            output = side_df.copy()
            output["cluster"] = methods[name]
            output["pc1"], output["pc2"] = coords[:, 0], coords[:, 1]
            output.to_csv(side_dir / f"{name}_player_clusters.csv", index=False)

    results_df.to_csv(side_dir / "model_scores.csv", index=False)

    print(f"\n{side.upper()} clustering complete")
    print(f"Players: {len(side_df)}  |  Features: {X.shape[1]}  |  "
          f"Subsampling iterations: {STABILITY_N_BOOTS} "
          f"({STABILITY_RESAMPLE_FRAC:.0%} without replacement)")
    print(f"Outputs written to: {side_dir}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    cluster_side(df, "ct")
    cluster_side(df, "t")
    print("\nDone. Run evaluate_clusters.py to score against ground truth.")


if __name__ == "__main__":
    main()
