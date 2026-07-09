from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")

                                                                                              
GROUND_TRUTH_CANDIDATES = [
    Path("Roles.csv"),
    Path("roles.csv"),
]

RANDOM_STATE = 69420
XGB_RANDOM_STATE = 42
XGB_TEST_SIZE = 0.25

                                                                         
EVAL_EXCLUDE = {
    "player_name", "side", "cluster", "pc1", "pc2",
    "gt_role_1", "gt_role_2", "gt_role_source",
    "gt_side_role", "gt_general_role", "gt_general_role_raw", "gt_igl_status",
    "gt_ct_role", "gt_t_role", "gt_match_key", "adr", "kpr",
}
                                                                  

def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _name_key(value) -> str:
    return _clean_text(value).lower()


def _canonical_general_role(role: str) -> str:

    role = _clean_text(role)
    low = role.lower()
    if not role or low == "unknown":
        return ""
    if "igl" in low:
        return "IGL"
    if "awp" in low:
        return "AWPer"
    return role


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    attempts = [
        {"encoding": "utf-16", "sep": "\t"},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "utf-8-sig", "sep": "\t"},
        {"encoding": "utf-8", "sep": "\t"},
    ]
    errors: list[str] = []
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs).fillna("")
            df.columns = df.columns.str.strip()
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            errors.append(f"{kwargs}: {exc}")
    raise ValueError(f"Could not read {path}. Tried: {' | '.join(errors)}")


def _prepare_roles_csv(gt: pd.DataFrame) -> pd.DataFrame:
    required = {"Name", "Role", "CT Role", "T Role"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Roles.csv is missing required columns: {sorted(missing)}")

    out = pd.DataFrame()
    out["player_page"] = gt["Name"].map(_clean_text)
    out["_key"] = gt["Name"].map(_name_key)
    out["gt_general_role_raw"] = gt["Role"].map(_clean_text)
    out["gt_general_role"] = gt["Role"].map(_canonical_general_role)
    out["gt_ct_role"] = gt["CT Role"].map(_clean_text)
    out["gt_t_role"] = gt["T Role"].map(_clean_text)
    out["gt_igl_status"] = np.where(
        out["gt_general_role_raw"].str.contains("IGL", case=False, na=False),
        "IGL",
        "Non-IGL",
    )
    out = out[out["_key"].str.len() > 0].drop_duplicates("_key", keep="first")
    return out


def _candidate_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    for base in [Path.cwd(), script_dir]:
        for path in GROUND_TRUTH_CANDIDATES:
            candidate = base / path
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def load_ground_truth() -> pd.DataFrame | None:
    for path in _candidate_paths():
        if not path.exists():
            continue

        raw = _read_csv_flexible(path)
        columns = set(raw.columns)
        required = {"Name", "Role", "CT Role", "T Role"}
        if not required.issubset(columns):
            print(f"[warn] {path} is missing required columns {sorted(required - columns)} - skipping")
            continue

        try:
            gt = _prepare_roles_csv(raw)
        except ValueError as exc:
            print(f"[warn] could not use {path}: {exc}")
            continue

        print(f"[info] loaded ground truth: {path} ({len(gt)} players)")
        return gt

    searched = ", ".join(str(p) for p in _candidate_paths())
    print(f"[info] no Roles.csv found. Searched: {searched}")
    print("[info] skipping GT comparison")
    return None


def match_ground_truth(players_df: pd.DataFrame, gt: pd.DataFrame, side: str, mode: str = "side") -> pd.DataFrame:

    stale_cols = [
        c for c in (
            "gt_role_1", "gt_role_2", "gt_role_source",
            "gt_side_role", "gt_general_role", "gt_general_role_raw",
            "gt_igl_status", "gt_ct_role", "gt_t_role", "gt_match_key",
        )
        if c in players_df.columns
    ]
    tmp = players_df.drop(columns=stale_cols, errors="ignore").copy()
    tmp["_key"] = tmp["player_name"].map(_name_key)

    merged = tmp.merge(gt, on="_key", how="left")

    side = side.lower().strip()
    if side == "ct":
        side_col = "gt_ct_role"
        side_source = "CT Role"
    elif side == "t":
        side_col = "gt_t_role"
        side_source = "T Role"
    else:
        raise ValueError(f"side must be 'ct' or 't', got {side!r}")

    merged["gt_side_role"] = merged[side_col].fillna("").map(_clean_text)
    merged["gt_general_role"] = merged["gt_general_role"].fillna("").map(_clean_text)
    merged["gt_general_role_raw"] = merged["gt_general_role_raw"].fillna("").map(_clean_text)
    merged["gt_igl_status"] = merged["gt_igl_status"].fillna("").map(_clean_text)

    if mode == "side":
        merged["gt_role_1"] = merged["gt_side_role"]
        merged["gt_role_2"] = ""
        merged["gt_role_source"] = side_source
    elif mode == "general":
        merged["gt_role_1"] = merged["gt_general_role"]
        merged["gt_role_2"] = ""
        merged["gt_role_source"] = "Role (IGL-* collapsed to IGL)"
    elif mode == "igl":
        merged["gt_role_1"] = merged["gt_igl_status"]
        merged["gt_role_2"] = ""
        merged["gt_role_source"] = "IGL binary"
    else:
        raise ValueError("mode must be one of: side, general, igl")

    merged["gt_match_key"] = merged["_key"]
    merged = merged.drop(columns=["_key"], errors="ignore")
    merged[["gt_role_1", "gt_role_2"]] = merged[["gt_role_1", "gt_role_2"]].fillna("")
    return merged
                                                                          

def _is_valid_role(role: str) -> bool:
    return _clean_text(role).lower() not in ("", "unknown", "nan")


def resolve_gt_label(players_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:

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
                    role = _clean_text(role)
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
        r1 = _clean_text(row.get("gt_role_1", ""))
        r2 = _clean_text(row.get("gt_role_2", ""))

        if _is_valid_role(r1) and r1.lower() == str(top_role).lower():
            return r1
        if _is_valid_role(r2) and r2.lower() == str(top_role).lower():
            return r2
        if _is_valid_role(r1):
            return r1
        if _is_valid_role(r2):
            return r2
        return ""

    return df.apply(pick_role, axis=1)


def ground_truth_accuracy(labels: np.ndarray, players_df: pd.DataFrame) -> tuple[float | None, pd.DataFrame | None]:
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
        n_match = int((valid["_resolved"] == dominant).sum())
        n_players = int(len(valid))
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


def ground_truth_match_stats(players_df: pd.DataFrame) -> tuple[int, float]:
    if "gt_role_1" not in players_df.columns:
        return 0, 0.0
    matched = players_df["gt_role_1"].map(_is_valid_role)
    n_matched = int(matched.sum())
    rate = n_matched / len(players_df) if len(players_df) else 0.0
    return n_matched, rate
                                                                          

def compute_feature_importance(df: pd.DataFrame, labels: np.ndarray) -> pd.Series | None:
    X = df.drop(columns=[c for c in EVAL_EXCLUDE if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        return None

    valid = labels != -1
    if len(set(labels[valid])) < 2 or valid.sum() < 10:
        return None

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X.values[valid], labels[valid])
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                                                                         

METHOD_COLORS = {
    "kmeans": "#4C72B0",
    "gmm": "#DD8452",
    "hdbscan": "#8C8C8C",
}


def plot_stability_tiers(results_df: pd.DataFrame, side: str) -> None:

    stab = (
        results_df[["name", "method", "stability_mean", "stability_std"]]
        .dropna(subset=["stability_mean"])
        .sort_values("stability_mean", ascending=False)
        .reset_index(drop=True)
    )
    if stab.empty:
        return

    fig_height = max(4.0, 0.3 * len(stab) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y_pos = np.arange(len(stab))
    colors = [METHOD_COLORS.get(m, "#999999") for m in stab["method"]]

    ax.barh(
        y_pos, stab["stability_mean"], xerr=stab["stability_std"],
        color=colors, error_kw={"ecolor": "black", "elinewidth": 1, "capsize": 3},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stab["name"], fontsize=8)
    ax.invert_yaxis()                                

    ax.set_xlabel("Mean ARI (± 1 std)")
    ax.set_title(f"Bootstrap Stability — {side.upper()}")

    x_max = (stab["stability_mean"] + stab["stability_std"]).max()
    ax.set_xlim(0, max(1.05, x_max * 1.05))

    ax.axvline(0.50, color="black", linestyle="--", linewidth=1)
    ax.axvline(0.80, color="black", linestyle=":", linewidth=1)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    method_handles = [
        Patch(color=METHOD_COLORS[m], label=m)
        for m in ("kmeans", "gmm")
        if m in stab["method"].unique()
    ]
    threshold_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Min threshold (0.50)"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1, label="High tier (0.80)"),
    ]
    ax.legend(handles=method_handles + threshold_handles, loc="lower right", fontsize=8)

    plt.tight_layout()

    plots_side_dir = PLOTS_DIR / side
    plots_side_dir.mkdir(parents=True, exist_ok=True)
    out = plots_side_dir / "stability_tiers.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def plot_composite_scores(results_df: pd.DataFrame, side: str) -> None:

    scored = (
        results_df[["name", "method", "composite_score"]]
        .dropna(subset=["composite_score"])
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    if scored.empty:
        return

    fig_height = max(4.0, 0.3 * len(scored) + 1.5)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    y_pos = np.arange(len(scored))
    colors = [METHOD_COLORS.get(m, "#999999") for m in scored["method"]]

    ax.barh(y_pos, scored["composite_score"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scored["name"], fontsize=8)
    ax.invert_yaxis()                            

    ax.set_xlabel("Composite Score")
    ax.set_title(f"Composite Score — {side.upper()}")

    x_max = scored["composite_score"].max()
    ax.set_xlim(0, max(0.1, x_max * 1.1))

    median = scored["composite_score"].median()
    ax.axvline(median, color="black", linestyle="--", linewidth=1)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    method_handles = [
        Patch(color=METHOD_COLORS[m], label=m)
        for m in ("kmeans", "gmm")
        if m in scored["method"].unique()
    ]
    threshold_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1, label=f"Median ({median:.3f})"),
    ]
    ax.legend(handles=method_handles + threshold_handles, loc="lower right", fontsize=8)

    plt.tight_layout()

    plots_side_dir = PLOTS_DIR / side
    plots_side_dir.mkdir(parents=True, exist_ok=True)
    out = plots_side_dir / "composite_scores.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


def _print_stability_tiers(results_df: pd.DataFrame) -> None:
    stab = (
        results_df[["name", "stability_mean", "stability_std"]]
        .dropna()
        .sort_values("stability_mean", ascending=False)
    )
    if stab.empty:
        return

    def tier(v):
        if v >= 0.80:
            return "HIGH"
        if v >= 0.50:
            return "MEDIUM"
        return "LOW"

    print("\nStability tiers (bootstrap ARI):")
    for _, row in stab.iterrows():
        print(
            f"  {row['name']:<30}  mean={row['stability_mean']:.3f}  "
            f"std={row['stability_std']:.3f}  [{tier(row['stability_mean'])}]"
        )
                                                                

def plot_xgb_classification_report(
    df: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    side: str,
) -> None:

    X = df.drop(columns=[c for c in EVAL_EXCLUDE if c in df.columns], errors="ignore")
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
            X_valid,
            y_enc,
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
        y_test,
        preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

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
                "Support": int(sum(report[t]["support"] for t in target_names)),
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

    row_colors = []
    divider_idx = len(target_names)
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

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"XGBoost Classification Report - {side.upper()} {name}",
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


ROLE_COLORS = {
    "AWPer": "#D55E00",
    "Flex": "#E69F00",
    "Lurker": "#0072B2",
    "Spacetaker": "#009E73",
    "Rotator": "#009E73",
    "Anchor": "#0072B2",
    "Mixed": "#CC79A7",
    "No GT": "#C7C7C7",
}

IGL_HIGHLIGHT_COLOR = "#00BFFF"

def plot_cluster_vs_ground_truth(
    df_with_gt: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    side: str,
) -> None:

    if "gt_role_1" not in df_with_gt.columns:
        return
    if "pc1" not in df_with_gt.columns or "pc2" not in df_with_gt.columns:
        return

    valid = labels != -1
    if valid.sum() == 0:
        return

    plot_cols = ["player_name", "pc1", "pc2", "gt_role_1", "gt_igl_status"]
    plot_cols = [c for c in plot_cols if c in df_with_gt.columns]
    plot_df = df_with_gt.loc[valid, plot_cols].copy()
    plot_df["cluster"] = labels[valid]

    gt_role = plot_df["gt_role_1"].map(_clean_text)
    has_role = gt_role.map(_is_valid_role)
    plot_df["has_gt"] = has_role
    plot_df["gt_role"] = gt_role.where(has_role, "No GT")

    if "gt_igl_status" in plot_df.columns:
        plot_df["is_igl"] = plot_df["gt_igl_status"].fillna("").astype(str).str.strip().eq("IGL")
    else:
        plot_df["is_igl"] = False

    plots_side_dir = PLOTS_DIR / side
    plots_side_dir.mkdir(parents=True, exist_ok=True)

    def _save_one_plot(plot_data: pd.DataFrame, clean: bool) -> None:
        if plot_data.empty:
            print(f"[skip] {side} {name}: no rows available for {'clean ' if clean else ''}PCA-vs-GT plot")
            return

        fig, ax = plt.subplots(figsize=(11, 8.5))

        cluster_ids = sorted(plot_data["cluster"].unique())
        cluster_palette = plt.cm.tab10.colors
        for i, cluster_id in enumerate(cluster_ids):
            points = plot_data.loc[plot_data["cluster"] == cluster_id, ["pc1", "pc2"]].to_numpy()
            color = cluster_palette[i % len(cluster_palette)]
            label = f"Cluster {cluster_id} boundary"

            if len(points) >= 3:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                boundary = Polygon(
                    hull_points, closed=True,
                    facecolor=color, edgecolor=color,
                    alpha=0.25, linewidth=1.0, zorder=1,
                    label=label,
                )
                ax.add_patch(boundary)
            else:
                ax.scatter([], [], color=color, alpha=0.3, s=100, marker="s", label=label)

        role_order = sorted(r for r in plot_data["gt_role"].unique() if r != "No GT")
        if "No GT" in plot_data["gt_role"].values:
            role_order.append("No GT")

        for role in role_order:
            subset = plot_data[plot_data["gt_role"] == role]
            color = ROLE_COLORS.get(role, "#999999")
            ax.scatter(
                subset["pc1"], subset["pc2"],
                color=color, edgecolor="white", linewidth=0.4,
                s=45, zorder=3, label=role, alpha=0.75,
            )

        igl_subset = plot_data[plot_data["is_igl"]]
        if not igl_subset.empty:
            ax.scatter(
                igl_subset["pc1"], igl_subset["pc2"],
                facecolors="none", edgecolors=IGL_HIGHLIGHT_COLOR,
                linewidth=0.75, s=120, zorder=5, label="IGL highlight", alpha = 0.6
            )

        for _, row in plot_data.iterrows():
            ax.annotate(
                row["player_name"],
                (row["pc1"], row["pc2"]),
                fontsize=6, xytext=(3, 3), textcoords="offset points", zorder=4,
            )

        n_matched = int(plot_data["has_gt"].sum())
        n_total = int(len(plot_data))
        match_pct = n_matched / n_total if n_total else 0.0
        clean_suffix = " Clean" if clean else ""

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"PCA - Ground-Truth Roles vs Cluster Boundaries{clean_suffix}\n"
            f"{side.upper()} {name}  ({n_matched}/{n_total} matched, {match_pct:.0%})"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(title="GT Role", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
        plt.tight_layout()

        suffix = "_clean" if clean else ""
        out = plots_side_dir / f"{name}_cluster_vs_gt{suffix}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[ok] {out}")

    _save_one_plot(plot_df, clean=False)
    _save_one_plot(plot_df[plot_df["has_gt"]].copy(), clean=True)                                                                            

def _ensure_result_cols(results_df: pd.DataFrame, gt: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["gt_ari", "gt_accuracy", "gt_matched_n", "gt_match_rate"]
    if gt is not None:
        cols += [
            "gt_side_ari", "gt_side_accuracy",
            "gt_general_ari", "gt_general_accuracy",
            "gt_igl_ari", "gt_igl_accuracy",
        ]
    for col in cols:
        if col not in results_df.columns:
            results_df[col] = float("nan")
    return results_df


def _store_metric(
    results_df: pd.DataFrame,
    model_name: str,
    prefix: str,
    labels: np.ndarray,
    df_with_gt: pd.DataFrame,
    side_dir: Path,
) -> None:
    ari = ground_truth_ari(labels, df_with_gt)
    acc, breakdown = ground_truth_accuracy(labels, df_with_gt)
    matched_n, match_rate = ground_truth_match_stats(df_with_gt)

    if ari is not None:
        results_df.loc[results_df["name"] == model_name, f"{prefix}_ari"] = ari
    if acc is not None:
        results_df.loc[results_df["name"] == model_name, f"{prefix}_accuracy"] = acc
    results_df.loc[results_df["name"] == model_name, "gt_matched_n"] = matched_n
    results_df.loc[results_df["name"] == model_name, "gt_match_rate"] = match_rate

    if breakdown is not None:
        breakdown.to_csv(side_dir / f"{model_name}_{prefix}_accuracy.csv", index=False)


def evaluate_side(side: str, gt: pd.DataFrame | None) -> None:
    side_dir = OUTPUT_DIR / side
    scores_path = side_dir / "model_scores.csv"

    if not scores_path.exists():
        print(f"[skip] {side}: model_scores.csv not found. Run cluster_players.py first.")
        return

    results_df = pd.read_csv(scores_path)
    results_df = _ensure_result_cols(results_df, gt)

    cluster_files = list(side_dir.glob("*_player_clusters.csv"))
    if not cluster_files:
        print(f"[skip] {side}: no cluster CSVs found.")
        return

    for cluster_path in sorted(cluster_files):
        name = cluster_path.stem.replace("_player_clusters", "")
        df = pd.read_csv(cluster_path)

        if "cluster" not in df.columns:
            continue
        labels = df["cluster"].to_numpy()

        df_for_models = df
        if gt is not None:
            df_side = match_ground_truth(df, gt, side=side, mode="side")
            _store_metric(results_df, name, "gt_side", labels, df_side, side_dir)

                                                                                                   
            results_df.loc[results_df["name"] == name, "gt_ari"] = results_df.loc[
                results_df["name"] == name, "gt_side_ari"
            ].values
            results_df.loc[results_df["name"] == name, "gt_accuracy"] = results_df.loc[
                results_df["name"] == name, "gt_side_accuracy"
            ].values

            df_general = match_ground_truth(df, gt, side=side, mode="general")
            _store_metric(results_df, name, "gt_general", labels, df_general, side_dir)

            df_igl = match_ground_truth(df, gt, side=side, mode="igl")
            _store_metric(results_df, name, "gt_igl", labels, df_igl, side_dir)

            df_for_models = df_side

        importance = compute_feature_importance(df_for_models, labels)
        if importance is not None:
            importance.to_frame("importance").to_csv(side_dir / f"{name}_feature_importance.csv")

    results_df.to_csv(scores_path, index=False)
                                                                                        
    min_stability = 0.50
    non_hdbscan = results_df[results_df["method"] != "hdbscan"]
    eligible = non_hdbscan[
        non_hdbscan["stability_mean"].isna()
        | (non_hdbscan["stability_mean"] >= min_stability)
    ]
    if eligible.empty:
        print("[warn] all non-HDBSCAN models below stability threshold; falling back.")
        eligible = results_df
    best_models = eligible.sort_values("composite_score", ascending=False).head(2)

    print(f"\n{'=' * 60}")
    print(f"{side.upper()} EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    display_cols = [
        "name", "k", "silhouette", "davies_bouldin",
        "stability_mean", "stability_std", "composite_score",
    ]
    if gt is not None:
        display_cols += [
            "gt_side_ari", "gt_side_accuracy",
            "gt_general_ari", "gt_general_accuracy",
            "gt_igl_ari", "gt_igl_accuracy",
            "gt_matched_n", "gt_match_rate",
        ]
    display_cols = [c for c in display_cols if c in best_models.columns]

    print("\nBest 2 models (non-HDBSCAN, ranked by composite score):")
    print(best_models[display_cols].to_string(index=False))

    if gt is not None:
        print("\nGT note: gt_side_* uses CT Role for CT and T Role for T.")
        print("GT note: gt_general_* uses Role with IGL-* collapsed to IGL.")
        for _, row in best_models.iterrows():
            name = row["name"]
            for prefix, label in [
                ("gt_side", "side-specific"),
                ("gt_general", "general role"),
                ("gt_igl", "IGL binary"),
            ]:
                breakdown_path = side_dir / f"{name}_{prefix}_accuracy.csv"
                if breakdown_path.exists():
                    breakdown = pd.read_csv(breakdown_path)
                    acc = row.get(f"{prefix}_accuracy", float("nan"))
                    ari = row.get(f"{prefix}_ari", float("nan"))
                    print(
                        f"\nGT accuracy - {side.upper()} {name} ({label}; "
                        f"overall={acc:.1%}  ARI={ari:.4f}):"
                    )
                    print(breakdown.to_string(index=False))

    if gt is not None and "gt_match_rate" in results_df.columns:
        match_rates = results_df["gt_match_rate"].dropna()
        if not match_rates.empty and match_rates.min() < 0.90:
            print(
                f"\n[warn] only {match_rates.min():.1%} of {side.upper()} cluster rows matched "
                "the ground-truth file. Interpret GT metrics as matched-sample metrics."
            )

    _print_stability_tiers(results_df)
    plot_stability_tiers(results_df, side)
    plot_composite_scores(results_df, side)

    for _, row in best_models.iterrows():
        name = row["name"]
        imp_path = side_dir / f"{name}_feature_importance.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            print(f"\nFeature importance - {side.upper()} {name}:")
            print(imp.head(15).to_string())

        cluster_path = side_dir / f"{name}_player_clusters.csv"
        if cluster_path.exists():
            df_plot = pd.read_csv(cluster_path)
            if gt is not None:
                df_plot = match_ground_truth(df_plot, gt, side=side, mode="side")
            plot_xgb_classification_report(
                df_plot,
                df_plot["cluster"].to_numpy(),
                name,
                side,
            )
            plot_cluster_vs_ground_truth(
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