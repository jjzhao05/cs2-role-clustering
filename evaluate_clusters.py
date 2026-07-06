from pathlib import Path
import numpy as np
import pandas as pd
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
    Path("Roles.csv")
]

RANDOM_STATE = 69420
XGB_RANDOM_STATE = 42
XGB_TEST_SIZE = 0.25

SHOW_UNMATCHED_GT_POINTS = False
INCLUDE_UNMATCHED_PLAYERS_IN_CLUSTER_HULLS = False

SHOW_IGL_MARKERS = True
JITTER_POINTS = True
PCA_JITTER_STD = 0.04
PCA_JITTER_RANDOM_STATE = 7

EVAL_EXCLUDE = {
    "player_name", "side", "cluster", "pc1", "pc2",
    "gt_role_1", "gt_role_2", "gt_role_source",
    "gt_side_role", "gt_general_role", "gt_general_role_raw", "gt_igl_status",
    "gt_ct_role", "gt_t_role", "gt_match_key", "adr", "kpr",
}

# ---------------------------------------------------------------------------
# Ground truth loading + joining
# ---------------------------------------------------------------------------

def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _name_key(value) -> str:
    return _clean_text(value).lower()


def _explicit_role(role: str) -> str:
    """Return the role label exactly as it appears in Roles.csv, except blanks/unknowns."""
    role = _clean_text(role)
    if role.lower() in ("", "unknown", "nan"):
        return ""
    return role


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read either Roles.csv or the older Liquipedia CSV without hard-coding one dialect."""
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
    out["gt_general_role"] = gt["Role"].map(_explicit_role)
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
    """Search both the current working directory and the script directory."""
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
    """
    Join ground-truth labels (from Roles.csv) to a player-cluster dataframe.

    mode='side'    -> CT uses CT Role, T uses T Role. This is the main metric.
    mode='general' -> uses the explicit general Role column exactly as written in Roles.csv.
    mode='igl'     -> binary IGL vs Non-IGL diagnostic.
    """
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
        merged["gt_role_source"] = "Role (explicit Roles.csv label)"
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


# ---------------------------------------------------------------------------
# GT metric helpers
# ---------------------------------------------------------------------------

def _is_valid_role(role: str) -> bool:
    return _clean_text(role).lower() not in ("", "unknown", "nan")


def resolve_gt_label(players_df: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """
    For each player, pick whichever of their two GT roles best matches their
    cluster's majority role, so dual-role players are handled fairly.
    For Roles.csv side/general/igl modes, gt_role_2 is intentionally blank.
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
    """Adjusted Rand Index between cluster labels and resolved GT roles."""
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


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# XGBoost validation plots
# ---------------------------------------------------------------------------

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



EXPLICIT_ROLE_COLORS = {
    # Shared / weapon specialist
    "AWPer": "#D81B60",       # strong magenta-red
    "IGL-AWPer": "#8E063B",   # dark magenta-red

    # General Role labels from Roles.csv
    "Opener": "#E69F00",      # amber
    "Closer": "#0072B2",      # blue
    "Linchpin": "#009E73",    # teal-green
    "IGL-Opener": "#56B4E9",  # sky blue
    "IGL-Closer": "#6A3D9A",  # purple

    # CT Role labels from Roles.csv
    "Anchor": "#0072B2",      # blue
    "Rotator": "#009E73",     # teal-green
    "Mixed": "#6A3D9A",       # purple

    # T Role labels from Roles.csv
    "Spacetaker": "#E69F00",  # amber/orange
    "Lurker": "#6A3D9A",      # purple
    "Flex": "#009E73",        # teal-green

    # Missing ground truth
    "No GT": "#A0A0A0",       # neutral gray
}

CLUSTER_BOUNDARY_COLORS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#54A24B",  # green
    "#B279A2",  # mauve
    "#9D755D",  # brown
    "#72B7B2",  # teal
    "#EECA3B",  # yellow
    "#8F8F8F",  # gray
]

FALLBACK_ROLE_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#6A3D9A",
    "#56B4E9",
    "#D81B60",
    "#8A8A8A",
]


def _role_color(role: str, idx: int) -> str:
    if role in EXPLICIT_ROLE_COLORS:
        return EXPLICIT_ROLE_COLORS[role]
    return FALLBACK_ROLE_COLORS[idx % len(FALLBACK_ROLE_COLORS)]

def plot_cluster_vs_ground_truth(
    df_with_gt: pd.DataFrame,
    labels: np.ndarray,
    name: str,
    side: str,
) -> None:
    """
    PCA scatter plot: each point is a player, colored by ground-truth role,
    with a translucent convex-hull boundary drawn around each cluster.

    By default, players missing ground-truth labels are hidden from the plot.
    The quantitative GT metrics are also matched-subset metrics, so this keeps
    the main report figure visually aligned with the reported evaluation.

    Set SHOW_UNMATCHED_GT_POINTS=True for diagnostic coverage plots.
    """
    if "gt_role_1" not in df_with_gt.columns:
        return
    if "pc1" not in df_with_gt.columns or "pc2" not in df_with_gt.columns:
        return

    valid = labels != -1
    if valid.sum() == 0:
        return

    optional_cols = [
        c for c in ("gt_igl_status", "gt_general_role_raw", "gt_general_role")
        if c in df_with_gt.columns
    ]
    plot_cols = ["player_name", "pc1", "pc2", "gt_role_1"] + optional_cols
    plot_df = df_with_gt.loc[valid, plot_cols].copy()
    plot_df["cluster"] = labels[valid]

    if "gt_igl_status" in plot_df.columns:
        plot_df["is_igl"] = plot_df["gt_igl_status"].map(_clean_text).str.lower().eq("igl")
    elif "gt_general_role_raw" in plot_df.columns:
        plot_df["is_igl"] = plot_df["gt_general_role_raw"].map(_clean_text).str.contains(
            "IGL", case=False, na=False
        )
    elif "gt_general_role" in plot_df.columns:
        plot_df["is_igl"] = plot_df["gt_general_role"].map(_clean_text).str.contains(
            "IGL", case=False, na=False
        )
    else:
        plot_df["is_igl"] = False

    gt_role = plot_df["gt_role_1"].map(_clean_text)
    has_role = gt_role.map(_is_valid_role)
    n_matched = int(has_role.sum())
    n_total = int(len(has_role))
    match_pct = n_matched / n_total if n_total else 0.0

    plot_df["gt_role"] = gt_role.where(has_role, "No GT")

    if SHOW_UNMATCHED_GT_POINTS:
        display_df = plot_df.copy()
    else:
        display_df = plot_df[plot_df["gt_role"] != "No GT"].copy()

    if display_df.empty:
        print(f"[skip] {side.upper()} {name}: no matched ground-truth rows to plot")
        return

    # Apply tiny jitter only to rendered point/text positions.
    if JITTER_POINTS:
        rng = np.random.default_rng(PCA_JITTER_RANDOM_STATE)
        display_df["pc1_plot"] = display_df["pc1"] + rng.normal(0, PCA_JITTER_STD, len(display_df))
        display_df["pc2_plot"] = display_df["pc2"] + rng.normal(0, PCA_JITTER_STD, len(display_df))
    else:
        display_df["pc1_plot"] = display_df["pc1"]
        display_df["pc2_plot"] = display_df["pc2"]

    fig, ax = plt.subplots(figsize=(12, 8.5))

    # Translucent convex-hull boundary per cluster
    hull_df = plot_df if INCLUDE_UNMATCHED_PLAYERS_IN_CLUSTER_HULLS else display_df
    cluster_ids = sorted(hull_df["cluster"].unique())
    cluster_palette = CLUSTER_BOUNDARY_COLORS
    for i, cluster_id in enumerate(cluster_ids):
        points = hull_df.loc[hull_df["cluster"] == cluster_id, ["pc1", "pc2"]].to_numpy()
        color = cluster_palette[i % len(cluster_palette)]
        label = f"Cluster {cluster_id} boundary"

        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            boundary = Polygon(
                hull_points, closed=True,
                facecolor=color, edgecolor=color,
                alpha=0.1, linewidth=1.8, zorder=1,
                label=label,
            )
            ax.add_patch(boundary)


    # Points colored by ground-truth role
    role_order = sorted(r for r in display_df["gt_role"].unique() if r != "No GT")
    if SHOW_UNMATCHED_GT_POINTS and "No GT" in display_df["gt_role"].values:
        role_order.append("No GT")

    for role in role_order:
        subset = display_df[display_df["gt_role"] == role]
        color = _role_color(role, role_order.index(role))
        ax.scatter(
            subset["pc1_plot"], subset["pc2_plot"],
            color=color, edgecolor="white", linewidth=0.8,
            s=58, zorder=3, label=role,
        )

    #  ring around IGLs, while preserving role color
    if SHOW_IGL_MARKERS and "is_igl" in display_df.columns:
        igl_df = display_df[display_df["is_igl"]].copy()
        if not igl_df.empty:
            ax.scatter(
                igl_df["pc1_plot"], igl_df["pc2_plot"],
                facecolors="none", edgecolors="#1D72CC", linewidth=1.8,
                s=112, marker="o", zorder=5, label="IGL"
            )

    # Player-name labels
    for _, row in display_df.iterrows():
        ax.annotate(
            row["player_name"],
            (row["pc1_plot"], row["pc2_plot"]),
            fontsize=5.5, alpha=0.8, xytext=(3, 3), textcoords="offset points", zorder=4,
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    igl_count = int(display_df.get("is_igl", pd.Series(False, index=display_df.index)).sum())
    igl_note = f"; {igl_count} IGLs marked" if SHOW_IGL_MARKERS and igl_count else ""

    jitter_note = "; points lightly jittered" if JITTER_POINTS else ""
    ax.set_title(
        f"PCA - Explicit Role Labels + IGL Overlay vs Cluster Boundaries\n"
        f"{side.upper()} {name}  ({n_matched}/{n_total} GT matched, {match_pct:.0%}; "
        f"plotting {len(display_df)} labeled players{igl_note}{jitter_note})"
    )
    ax.grid(True, alpha=0.4)
    ax.legend(title="Explicit Role Label", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=9, frameon=True)
    plt.tight_layout()

    plots_side_dir = PLOTS_DIR / side
    plots_side_dir.mkdir(parents=True, exist_ok=True)
    out = plots_side_dir / f"{name}_cluster_vs_gt.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[ok] {out}")


# ---------------------------------------------------------------------------
# Per-side evaluation
# ---------------------------------------------------------------------------

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

            # Backward-compatible columns: gt_ari / gt_accuracy are the main side-specific metrics.
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

    # Best model selection: exclude HDBSCAN and require minimum stability when possible.
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
        print("GT note: gt_general_* uses the explicit Role label from Roles.csv.")
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