from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

INPUT_PATH = "output.csv"
GT_PATH = "liquipedia_player_roles.csv"
OUTPUT_DIR = Path("outputs/ablations")

RANDOM_STATE = 69420
K_VALUES = [3, 4, 5, 6, 7, 8]
N_BOOTS = 50
RESAMPLE_FRAC = 0.80

META_COLS = {"player_name", "side", "rounds_played"}

FEATURE_GROUPS = {
    "combat": [
        "survival_rate",
        "damage_per_round",
        "damage_taken_per_round",
        "damage_diff_per_round",
        "multi_kill_rate",
        "rifle_kill_share",
        "awp_kill_share",
    ],
    "opening_aggression": [
        "opening_kill_rate",
        "opening_death_rate",
        "opening_duel_success",
        "first_contact_rate",
        "time_near_enemy_rate",
    ],
    "trading_teamwork": [
        "trade_kill_rate",
        "death_traded_rate",
        "trade_participation",
        "assists_per_round",
        "flash_assists_per_round",
    ],
    "utility": [
        "grenades_per_round",
        "he_grenades_per_round",
        "flashbangs_per_round",
        "smokes_per_round",
        "fire_nades_per_round",
        "util_damage_per_round",
    ],
    "positioning_movement": [
        "avg_distance_to_enemy",
        "avg_distance_to_team_centroid",
        "relative_team_centroid_distance",
        "avg_distance_moved_per_round",
        "avg_distance_to_closest_teammate",
        "time_stationary_rate",
    ],
}


def load_data():
    df = pd.read_csv(INPUT_PATH).fillna(0)
    df["side"] = df["side"].str.lower()
    return df[df["side"].isin(["ct", "t"])].copy()


def load_ground_truth():
    path = Path(GT_PATH)
    if not path.exists():
        return None

    gt = pd.read_csv(path).fillna("")
    gt["_key"] = gt["player_page"].str.lower().str.strip()
    return gt


def add_ground_truth(df, gt):
    if gt is None:
        df["gt_role_1"] = ""
        df["gt_role_2"] = ""
        return df

    out = df.copy()
    out["_key"] = out["player_name"].str.lower().str.strip()

    out = out.merge(
        gt[["_key", "role_1", "role_2"]].rename(
            columns={"role_1": "gt_role_1", "role_2": "gt_role_2"}
        ),
        on="_key",
        how="left",
    )

    out = out.drop(columns=["_key"])
    out[["gt_role_1", "gt_role_2"]] = out[["gt_role_1", "gt_role_2"]].fillna("")
    return out


def valid_role(role):
    return str(role).strip().lower() not in {"", "unknown"}


def resolve_gt_labels(df, labels):
    work = df.copy()
    work["_cluster"] = labels

    cluster_majorities = {}

    for cluster_id, group in work.groupby("_cluster"):
        if cluster_id == -1:
            continue

        counts = {}

        for col in ["gt_role_1", "gt_role_2"]:
            for role in group[col]:
                if valid_role(role):
                    counts[role] = counts.get(role, 0) + 1

        if counts:
            cluster_majorities[cluster_id] = max(counts, key=counts.get)

    resolved = []

    for _, row in work.iterrows():
        cid = row["_cluster"]
        majority = cluster_majorities.get(cid)

        r1 = row.get("gt_role_1", "")
        r2 = row.get("gt_role_2", "")

        if majority is not None:
            if str(r1).strip().lower() == majority.strip().lower():
                resolved.append(r1)
                continue
            if str(r2).strip().lower() == majority.strip().lower():
                resolved.append(r2)
                continue

        if valid_role(r1):
            resolved.append(r1)
        elif valid_role(r2):
            resolved.append(r2)
        else:
            resolved.append("")

    return pd.Series(resolved, index=df.index)


def gt_metrics(df, labels):
    if "gt_role_1" not in df.columns:
        return np.nan, np.nan

    resolved = resolve_gt_labels(df, labels)

    mask = resolved.apply(valid_role) & (labels != -1)

    if mask.sum() < 2 or resolved[mask].nunique() < 2:
        return np.nan, np.nan

    encoded = LabelEncoder().fit_transform(resolved[mask])
    gt_ari = adjusted_rand_score(encoded, labels[mask])

    total_match = 0
    total_count = 0

    temp = pd.DataFrame({
        "cluster": labels[mask],
        "role": resolved[mask],
    })

    for _, group in temp.groupby("cluster"):
        dominant_count = group["role"].value_counts().max()
        total_match += dominant_count
        total_count += len(group)

    purity = total_match / total_count if total_count > 0 else np.nan

    return gt_ari, purity


def all_feature_columns(df):
    return [
        c for c in df.columns
        if c not in META_COLS
        and not c.startswith("gt_")
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique() > 1
    ]


def make_experiments(df):
    all_features = all_feature_columns(df)

    groups = {
        name: [c for c in cols if c in all_features]
        for name, cols in FEATURE_GROUPS.items()
    }

    std_features = [c for c in all_features if c.endswith("_std")]

    experiments = {}

    experiments["full"] = all_features

    for group_name, cols in groups.items():
        experiments[f"{group_name}_only"] = cols
        experiments[f"no_{group_name}"] = [
            c for c in all_features if c not in set(cols)
        ]

    experiments["consistency_only"] = std_features
    experiments["no_consistency"] = [
        c for c in all_features if c not in set(std_features)
    ]

    weapon_cols = {"awp_kill_share", "rifle_kill_share"}
    experiments["no_weapon_share"] = [
        c for c in all_features if c not in weapon_cols
    ]

    tactical_cols = (
        groups["opening_aggression"]
        + groups["trading_teamwork"]
        + groups["utility"]
        + groups["positioning_movement"]
    )
    experiments["tactics_only"] = sorted(set(tactical_cols))

    return {
        name: cols
        for name, cols in experiments.items()
        if len(cols) >= 2
    }


def bootstrap_stability(X_scaled, full_labels, k):
    rng = np.random.RandomState(RANDOM_STATE)
    n = X_scaled.shape[0]
    sample_size = max(3, int(n * RESAMPLE_FRAC))

    scores = []

    for i in range(N_BOOTS):
        idx = rng.choice(n, size=sample_size, replace=True)

        boot_labels = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE + i,
            n_init=20,
        ).fit_predict(X_scaled[idx])

        ref_labels = full_labels[idx]

        if len(set(ref_labels)) < 2 or len(set(boot_labels)) < 2:
            continue

        scores.append(adjusted_rand_score(ref_labels, boot_labels))

    if len(scores) < 5:
        return np.nan, np.nan

    return float(np.mean(scores)), float(np.std(scores))


def run_ablation_for_side(df, side, gt):
    side_df = df[df["side"] == side].reset_index(drop=True)
    side_df = add_ground_truth(side_df, gt)

    experiments = make_experiments(side_df)

    rows = []

    for ablation_name, features in experiments.items():
        X = side_df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        X = X.loc[:, X.nunique() > 1]

        if X.shape[1] < 2:
            continue

        X_scaled = StandardScaler().fit_transform(X)

        for k in K_VALUES:
            if k >= len(side_df):
                continue

            labels = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=20,
            ).fit_predict(X_scaled)

            sil = silhouette_score(X_scaled, labels)
            db = davies_bouldin_score(X_scaled, labels)

            stability_mean, stability_std = bootstrap_stability(
                X_scaled,
                labels,
                k,
            )

            gt_ari, gt_purity = gt_metrics(side_df, labels)

            composite = (
                0.40 * sil
                + 0.20 * (1 / (1 + db))
                + 0.40 * max(0, stability_mean - stability_std)
            )

            rows.append({
                "side": side,
                "ablation": ablation_name,
                "k": k,
                "n_features": X.shape[1],
                "features": ", ".join(X.columns),
                "silhouette": sil,
                "davies_bouldin": db,
                "stability_mean": stability_mean,
                "stability_std": stability_std,
                "gt_ari": gt_ari,
                "gt_purity": gt_purity,
                "composite_score": composite,
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    gt = load_ground_truth()

    all_results = []

    for side in ["ct", "t"]:
        print(f"[run] ablations for {side.upper()}")

        results = run_ablation_for_side(df, side, gt)

        side_dir = OUTPUT_DIR / side
        side_dir.mkdir(parents=True, exist_ok=True)

        results.to_csv(side_dir / "ablation_results_all.csv", index=False)

        best = (
            results.sort_values("composite_score", ascending=False)
            .groupby("ablation", as_index=False)
            .head(1)
            .sort_values("composite_score", ascending=False)
        )

        best.to_csv(side_dir / "ablation_results_best.csv", index=False)

        print(f"\nBest ablations — {side.upper()}")
        print(
            best[
                [
                    "ablation",
                    "k",
                    "n_features",
                    "silhouette",
                    "stability_mean",
                    "gt_ari",
                    "gt_purity",
                    "composite_score",
                ]
            ].to_string(index=False)
        )

        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "ablation_results_all_sides.csv", index=False)

    best_combined = (
        combined.sort_values("composite_score", ascending=False)
        .groupby(["side", "ablation"], as_index=False)
        .head(1)
        .sort_values(["side", "composite_score"], ascending=[True, False])
    )

    best_combined.to_csv(OUTPUT_DIR / "ablation_results_best_all_sides.csv", index=False)

    print(f"\n[done] results saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()