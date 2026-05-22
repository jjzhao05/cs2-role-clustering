from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

OUTPUT_DIR = Path("outputs")
GROUND_TRUTH_PATH = Path("liquipedia_player_roles.csv")
RANDOM_STATE = 69420

# Features excluded from evaluation (metadata / plot coords, not behavioral stats)
EVAL_EXCLUDE = {"player_name", "side", "cluster", "pc1", "pc2",
                "gt_role_1", "gt_role_2", "adr", "kpr"}


# ---------------------------------------------------------------------------
# Ground truth loading + joining
# ---------------------------------------------------------------------------

def load_ground_truth() -> pd.DataFrame | None:
    if not GROUND_TRUTH_PATH.exists():
        print(f"[info] ground-truth file not found at {GROUND_TRUTH_PATH} — skipping GT comparison")
        return None
    gt = pd.read_csv(GROUND_TRUTH_PATH).fillna("")
    gt.columns = gt.columns.str.strip()
    gt["_key"] = gt["player_page"].str.lower().str.strip()
    return gt


def match_ground_truth(players_df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    tmp = players_df.drop(columns=[c for c in ("gt_role_1", "gt_role_2") if c in players_df.columns], errors="ignore").copy()
    tmp["_key"] = tmp["player_name"].str.lower().str.strip()
    merged = tmp.merge(
        gt[["_key", "role_1", "role_2"]].rename(
            columns={"role_1": "gt_role_1", "role_2": "gt_role_2"}
        ),
        on="_key", how="left",
    ).drop(columns=["_key"])
    merged[["gt_role_1", "gt_role_2"]] = merged[["gt_role_1", "gt_role_2"]].fillna("")
    return merged


# ---------------------------------------------------------------------------
# GT metric helpers
# ---------------------------------------------------------------------------

def _is_valid_role(role: str) -> bool:
    return role.strip().lower() not in ("", "unknown")


def resolve_gt_label(players_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """
    For each player, pick whichever of their two GT roles best matches their
    cluster's majority role, so dual-role players are handled fairly.
    """
    df = players_df.copy()
    df["_label"] = labels

    role_counts: dict[int, pd.Series] = {}
    for cluster_id, group in df[df["_label"] != -1].groupby("_label"):
        counts: dict[str, int] = {}
        for col in ("gt_role_1", "gt_role_2"):
            if col not in group.columns:
                continue
            for role in group[col]:
                if _is_valid_role(role):
                    counts[role] = counts.get(role, 0) + 1
        role_counts[cluster_id] = pd.Series(counts)

    def pick_role(row) -> str:
        cid = row["_label"]
        if cid == -1:
            return ""
        majority = role_counts.get(cid, pd.Series(dtype=int))
        if majority.empty:
            return ""
        top_role = majority.idxmax()
        r1 = row.get("gt_role_1", "")
        r2 = row.get("gt_role_2", "")
        if r1.strip().lower() == top_role.strip().lower() and _is_valid_role(r1):
            return r1
        if r2.strip().lower() == top_role.strip().lower() and _is_valid_role(r2):
            return r2
        return r1 if _is_valid_role(r1) else (r2 if _is_valid_role(r2) else "")

    return df.apply(pick_role, axis=1)


def ground_truth_accuracy(labels: np.ndarray, players_df: pd.DataFrame) -> tuple[float | None, pd.DataFrame | None]:
    """
    Per-cluster accuracy: dominant GT role match rate per cluster + weighted overall.
    Returns (overall_accuracy, breakdown_df) or (None, None) if GT data is missing.
    """
    if "gt_role_1" not in players_df.columns:
        return None, None

    df = players_df.copy()
    df["_label"] = labels
    df["_resolved"] = resolve_gt_label(players_df, labels)

    rows = []
    total_match = total_players = 0

    for cluster_id, group in df[df["_label"] != -1].groupby("_label"):
        valid = group[group["_resolved"].str.strip().str.len() > 0]
        if valid.empty:
            continue
        dominant = valid["_resolved"].value_counts().idxmax()
        n_match = (valid["_resolved"] == dominant).sum()
        n_players = len(valid)
        rows.append({
            "cluster": cluster_id,
            "dominant_role": dominant,
            "n_players": n_players,
            "n_match": n_match,
            "accuracy": round(n_match / n_players, 4),
        })
        total_match += n_match
        total_players += n_players

    if not rows or total_players == 0:
        return None, None

    return round(total_match / total_players, 4), pd.DataFrame(rows).sort_values("cluster")


def ground_truth_ari(labels: np.ndarray, players_df: pd.DataFrame) -> float | None:
    """
    Adjusted Rand Index between cluster labels and resolved GT roles.
    """
    if "gt_role_1" not in players_df.columns:
        return None

    resolved = resolve_gt_label(players_df, labels)
    mask = (
        (labels != -1)
        & resolved.str.strip().str.len().gt(0)
        & resolved.str.strip().str.lower().ne("unknown")
    )
    if mask.sum() < 2 or resolved[mask].nunique() < 2:
        return None

    le = LabelEncoder()
    gt_encoded = le.fit_transform(resolved[mask])
    return float(adjusted_rand_score(gt_encoded, labels[mask]))


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(df: pd.DataFrame, labels: np.ndarray) -> pd.Series | None:
    X = df.drop(columns=[c for c in EVAL_EXCLUDE if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        return None

    valid = labels != -1
    if len(set(labels[valid])) < 2 or valid.sum() < 10:
        return None

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X.values[valid], labels[valid])
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Stability tier printer
# ---------------------------------------------------------------------------

def _print_stability_tiers(results_df: pd.DataFrame) -> None:
    stab = (
        results_df[["name", "stability_mean", "stability_std"]]
        .dropna()
        .sort_values("stability_mean", ascending=False)
    )
    if stab.empty:
        return

    def tier(v):
        if v >= 0.80: return "HIGH  ✓"
        if v >= 0.50: return "MEDIUM ~"
        return "LOW   ✗"

    print("\nStability tiers (bootstrap ARI):")
    for _, row in stab.iterrows():
        print(f"  {row['name']:<30}  mean={row['stability_mean']:.3f}  "
              f"std={row['stability_std']:.3f}  [{tier(row['stability_mean'])}]")



PLOTS_DIR = Path("plots")
XGB_RANDOM_STATE = 42
XGB_TEST_SIZE = 0.25


def plot_xgb_classification_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    side: str,
) -> None:
    """
    Train an XGBoost classifier on cluster labels, evaluate on a held-out
    test set, and save the classification report as a formatted PNG.
    """
    X = df.drop(columns=[c for c in EVAL_EXCLUDE if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        return

    valid = labels != -1
    X_valid = X.values[valid]
    y_valid = labels[valid]

    if len(set(y_valid)) < 2 or len(y_valid) < 10:
        return

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_valid)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_enc,
            test_size=XGB_TEST_SIZE,
            random_state=XGB_RANDOM_STATE,
            stratify=y_enc,
        )
    except ValueError:
        return

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=XGB_RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    cluster_ids = encoder.classes_
    target_names = [f"Cluster {c}" for c in cluster_ids]

    report = classification_report(
        y_test, preds,
        target_names=target_names,
        output_dict=True,
    )

    # Build display table
    rows = []
    for label in target_names:
        r = report[label]
        rows.append({
            "Cluster": label,
            "Precision": f"{r['precision']:.2f}",
            "Recall": f"{r['recall']:.2f}",
            "F1-Score": f"{r['f1-score']:.2f}",
            "Support": int(r["support"]),
        })

    # Spacer then summary rows
    for summary_key, display_label in [
        ("accuracy", "Accuracy"),
        ("macro avg", "Macro Avg"),
        ("weighted avg", "Weighted Avg"),
    ]:
        if summary_key == "accuracy":
            rows.append({
                "Cluster": display_label,
                "Precision": "",
                "Recall": "",
                "F1-Score": f"{report['accuracy']:.2f}",
                "Support": int(sum(r["support"] for r in [report[t] for t in target_names])),
            })
        else:
            r = report[summary_key]
            rows.append({
                "Cluster": display_label,
                "Precision": f"{r['precision']:.2f}",
                "Recall": f"{r['recall']:.2f}",
                "F1-Score": f"{r['f1-score']:.2f}",
                "Support": int(r["support"]),
            })

    table_df = pd.DataFrame(rows)
    col_labels = list(table_df.columns)
    cell_text = table_df.values.tolist()

    n_rows = len(table_df)
    fig_h = max(3.0, 0.45 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.axis("off")

    # Alternate row colors
    row_colors = []
    divider_idx = len(target_names)  # where summary rows start
    for i in range(n_rows):
        if i >= divider_idx:
            row_colors.append(["#dce8f5"] * len(col_labels))
        elif i % 2 == 0:
            row_colors.append(["#f7f7f7"] * len(col_labels))
        else:
            row_colors.append(["#ffffff"] * len(col_labels))

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.6)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"XGBoost Classification Report — {side.upper()} {name}",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()

    plots_side_dir = PLOTS_DIR / side
    plots_side_dir.mkdir(parents=True, exist_ok=True)
    out = plots_side_dir / f"{name}_xgb_report.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")

# ---------------------------------------------------------------------------
# Per-side evaluation
# ---------------------------------------------------------------------------

def evaluate_side(side: str, gt: pd.DataFrame | None) -> None:
    side_dir = OUTPUT_DIR / side
    scores_path = side_dir / "model_scores.csv"

    if not scores_path.exists():
        print(f"[skip] {side}: model_scores.csv not found — run cluster_players.py first.")
        return

    results_df = pd.read_csv(scores_path)

    # Add GT columns if not already present
    for col in ("gt_ari", "gt_accuracy"):
        if col not in results_df.columns:
            results_df[col] = float("nan")

    cluster_files = list(side_dir.glob("*_player_clusters.csv"))
    if not cluster_files:
        print(f"[skip] {side}: no cluster CSVs found.")
        return

    # --- Compute GT metrics + feature importance for every cluster CSV ---
    for cluster_path in sorted(cluster_files):
        name = cluster_path.stem.replace("_player_clusters", "")
        df = pd.read_csv(cluster_path)

        if "cluster" not in df.columns:
            continue
        labels = df["cluster"].to_numpy()

        if gt is not None:
            df = match_ground_truth(df, gt)
            gt_ari_val = ground_truth_ari(labels, df)
            gt_acc, gt_breakdown = ground_truth_accuracy(labels, df)

            if gt_ari_val is not None:
                results_df.loc[results_df["name"] == name, "gt_ari"] = gt_ari_val
            if gt_acc is not None:
                results_df.loc[results_df["name"] == name, "gt_accuracy"] = gt_acc
            if gt_breakdown is not None:
                gt_breakdown.to_csv(side_dir / f"{name}_gt_accuracy.csv", index=False)

        importance = compute_feature_importance(df, labels)
        if importance is not None:
            importance.to_frame("importance").to_csv(
                side_dir / f"{name}_feature_importance.csv"
            )

    results_df.to_csv(scores_path, index=False)

    # --- Best model selection: exclude HDBSCAN ---
    MIN_STABILITY = 0.50
    non_hdbscan = results_df[results_df["method"] != "hdbscan"]
    eligible = non_hdbscan[
        non_hdbscan["stability_mean"].isna()
        | (non_hdbscan["stability_mean"] >= MIN_STABILITY)
    ]
    if eligible.empty:
        print(f"[warn] all non-HDBSCAN models below stability threshold; falling back.")
        eligible = results_df
    best_models = eligible.head(2)

    # --- Console summary ---
    print(f"\n{'='*60}")
    print(f"{side.upper()} EVALUATION SUMMARY")
    print(f"{'='*60}")

    display_cols = ["name", "k", "silhouette", "davies_bouldin",
                    "stability_mean", "stability_std", "composite_score",
                    "gt_ari", "gt_accuracy"]
    print("\nBest 2 models (non-HDBSCAN, ranked by composite score):")
    print(best_models[display_cols].to_string(index=False))

    if gt is not None:
        for _, row in best_models.iterrows():
            name = row["name"]
            breakdown_path = side_dir / f"{name}_gt_accuracy.csv"
            if breakdown_path.exists():
                breakdown = pd.read_csv(breakdown_path)
                acc = row.get("gt_accuracy", float("nan"))
                ari = row.get("gt_ari", float("nan"))
                print(f"\nGT accuracy — {side.upper()} {name}  "
                      f"(overall={acc:.1%}  ARI={ari:.4f}):")
                print(breakdown.to_string(index=False))

    _print_stability_tiers(results_df)

    for _, row in best_models.iterrows():
        name = row["name"]
        imp_path = side_dir / f"{name}_feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            print(f"\nFeature importance — {side.upper()} {name}:")
            print(imp.head(15).to_string())

        # --- XGBoost classification report PNG ---
        cluster_path = side_dir / f"{name}_player_clusters.csv"
        if cluster_path.exists():
            df_plot = pd.read_csv(cluster_path)
            if gt is not None:
                df_plot = match_ground_truth(df_plot, gt)
            plot_xgb_classification_report(
                df_plot,
                df_plot["cluster"].to_numpy(),
                name,
                side,
            )


def main():
    gt = load_ground_truth()
    evaluate_side("ct", gt)
    evaluate_side("t", gt)


if __name__ == "__main__":
    main()