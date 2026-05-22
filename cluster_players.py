from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan

INPUT_PATH = "output.csv"
OUTPUT_DIR = Path("outputs")
K_VALUES = [3, 4, 5, 6, 7, 8]
RANDOM_STATE = 69420

# Features excluded from clustering but kept in the output CSV for plotting.
CLUSTER_EXCLUDE_FEATURES = {"adr", "kpr"}

# HDBSCAN hyperparameter grid
HDBSCAN_MIN_CLUSTER_SIZES = [2, 3, 4, 5, 6, 8, 10, 12, 15]
HDBSCAN_MIN_SAMPLES = [1, 2, 3, 4, 5]
HDBSCAN_MAX_NOISE_FRACTION = 0.30

# Stability analysis config
STABILITY_N_BOOTS = 50
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
    if df.empty:
        raise ValueError("No rows left after filtering.")
    return df


def get_features(df):
    exclude = {"player_name", "side", "rounds_played"} | CLUSTER_EXCLUDE_FEATURES
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        raise ValueError("No usable feature columns.")
    return X


def run_hdbscan(X_scaled, min_cluster_size, min_samples):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_scaled)
    if len(set(labels) - {-1}) < 2:
        return None, None
    return labels, clusterer


def _fit_labels(X_scaled, method_name: str, k: int | None,
                mcs: int | None, ms: int | None,
                random_state: int) -> np.ndarray | None:
    if method_name == "kmeans":
        return KMeans(n_clusters=k, random_state=random_state, n_init=20).fit_predict(X_scaled)

    if method_name == "gmm":
        return (
            GaussianMixture(n_components=k, random_state=random_state, covariance_type="full")
            .fit(X_scaled).predict(X_scaled)
        )

    if method_name == "hdbscan":
        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method="eom", prediction_data=True,
        ).fit_predict(X_scaled)
        return labels if len(set(labels) - {-1}) >= 2 else None

    raise ValueError(f"Unknown method: {method_name}")


def compute_stability(
    X_scaled: np.ndarray,
    full_labels: np.ndarray,
    method_name: str,
    k: int | None = None,
    mcs: int | None = None,
    ms: int | None = None,
    n_boots: int = STABILITY_N_BOOTS,
    resample_frac: float = STABILITY_RESAMPLE_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """
    Bootstrap stability via Adjusted Rand Index. Repeatedly resamples the
    dataset, refits the model, and measures how consistently clusters reappear.
    Returns (mean_ari, std_ari); returns (nan, nan) if fewer than 5 valid boots.
    """
    rng = np.random.RandomState(random_state)
    n = X_scaled.shape[0]
    sample_size = max(3, int(n * resample_frac))
    ari_scores: list[float] = []

    for boot_i in range(n_boots):
        idx = rng.choice(n, size=sample_size, replace=True)
        boot_labels = _fit_labels(X_scaled[idx], method_name, k=k, mcs=mcs, ms=ms,
                                  random_state=random_state + boot_i)
        if boot_labels is None:
            continue

        ref_labels = full_labels[idx]
        ref_valid = ref_labels != -1
        if ref_valid.sum() < 2 or len(set(ref_labels[ref_valid])) < 2:
            continue

        both_valid = ref_valid & (boot_labels != -1)
        if both_valid.sum() < 2 or len(set(ref_labels[both_valid])) < 2:
            continue

        ari_scores.append(adjusted_rand_score(ref_labels[both_valid], boot_labels[both_valid]))

    if len(ari_scores) < 5:
        return float("nan"), float("nan")
    return float(np.mean(ari_scores)), float(np.std(ari_scores))


def composite_score(silhouette: float, davies_bouldin: float,
                    stability_mean: float, stability_std: float) -> float:
    sil_term = SILHOUETTE_WEIGHT * (silhouette if not np.isnan(silhouette) else 0.0)
    db_term = DB_WEIGHT * (1.0 / (1.0 + davies_bouldin) if not np.isnan(davies_bouldin) else 0.0)
    if np.isnan(stability_mean) or np.isnan(stability_std):
        stab_score = 0.0
    else:
        stab_score = max(0.0, stability_mean - stability_std)
    return sil_term + db_term + STABILITY_WEIGHT * stab_score


def cluster_side(df, side):
    side_df = df[df["side"] == side].reset_index(drop=True)
    if len(side_df) < 3:
        print(f"Skipping {side}: not enough players.")
        return

    X = get_features(side_df)
    X_scaled = StandardScaler().fit_transform(X)
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)

    methods = {}
    results = []

    # --- KMeans + GMM ---
    for k in K_VALUES:
        if k >= len(side_df):
            continue

        for method_name, labels in [
            ("kmeans", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_scaled)),
            ("gmm", GaussianMixture(n_components=k, random_state=RANDOM_STATE, covariance_type="full").fit(X_scaled).predict(X_scaled)),
        ]:
            name = f"{method_name}_k{k}"
            methods[name] = labels
            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)
            stab_mean, stab_std = compute_stability(X_scaled, labels, method_name, k=k)
            results.append({
                "side": side, "method": method_name, "name": name, "k": k,
                "noise_pct": 0.0, "silhouette": sil, "davies_bouldin": db,
                "stability_mean": stab_mean, "stability_std": stab_std,
                "composite_score": composite_score(sil, db, stab_mean, stab_std),
            })

    # --- HDBSCAN grid search ---
    for mcs in HDBSCAN_MIN_CLUSTER_SIZES:
        for ms in HDBSCAN_MIN_SAMPLES:
            if mcs > len(side_df) // 2:
                continue
            labels, _ = run_hdbscan(X_scaled, mcs, ms)
            if labels is None:
                continue

            name = f"hdbscan_mcs{mcs}_ms{ms}"
            methods[name] = labels
            noise_mask = labels != -1
            noise_pct = (~noise_mask).mean()

            if noise_pct > HDBSCAN_MAX_NOISE_FRACTION:
                print(f"  [skip] {name} — {noise_pct:.0%} noise exceeds {HDBSCAN_MAX_NOISE_FRACTION:.0%} cap")
                continue
            if noise_mask.sum() < 2 or len(set(labels[noise_mask])) < 2:
                continue

            sil = silhouette_score(X_scaled[noise_mask], labels[noise_mask])
            db = davies_bouldin_score(X_scaled[noise_mask], labels[noise_mask])
            stab_mean, stab_std = compute_stability(X_scaled, labels, "hdbscan", mcs=mcs, ms=ms)
            results.append({
                "side": side, "method": "hdbscan", "name": name,
                "k": len(set(labels) - {-1}), "noise_pct": round(noise_pct, 4),
                "silhouette": sil, "davies_bouldin": db,
                "stability_mean": stab_mean, "stability_std": stab_std,
                "composite_score": composite_score(sil, db, stab_mean, stab_std),
            })

    if not results:
        print(f"No valid clustering results for {side}.")
        return

    results_df = pd.DataFrame(results).sort_values("composite_score", ascending=False)

    side_dir = OUTPUT_DIR / side
    side_dir.mkdir(parents=True, exist_ok=True)

    # Write top 2 per algorithm
    for method_name, group in results_df.groupby("method"):
        for _, row in group.head(2).iterrows():
            name = row["name"]
            if name not in methods:
                continue
            output = side_df.copy()
            output["cluster"] = methods[name]
            output["pc1"] = coords[:, 0]
            output["pc2"] = coords[:, 1]
            output.to_csv(side_dir / f"{name}_player_clusters.csv", index=False)

    results_df.to_csv(side_dir / "model_scores.csv", index=False)

    print(f"\n{side.upper()} clustering complete")
    print(f"Players: {len(side_df)}  |  Features: {X.shape[1]}  |  "
          f"Bootstrap iterations: {STABILITY_N_BOOTS} ({STABILITY_RESAMPLE_FRAC:.0%} resample)")
    print(f"Outputs written to: {side_dir}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    cluster_side(df, "ct")
    cluster_side(df, "t")
    print("\nDone. Run evaluate_clusters.py to score against ground truth.")


if __name__ == "__main__":
    main()