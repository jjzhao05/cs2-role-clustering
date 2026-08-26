from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox
from scipy.spatial import ConvexHull
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from feature_config import select_model_features

OUTPUT_DIR = Path("outputs")
PLOTS_DIR = Path("plots")

                                                                                              
GROUND_TRUTH_CANDIDATES = [
    Path("Roles.csv"),
    Path("roles.csv"),
]

PLAYER_PROFILE_CANDIDATES = [
    Path("output.csv"),
]

# Naming every point makes the PCA plots unreadable, so only this many players
# are annotated, picked once and shared by both sides. Players are taken from
# each cluster of the reference models in turn so that every cluster on both
# plots carries at least LABEL_MIN_PER_CLUSTER names, and LABEL_INCLUDE /
# LABEL_EXCLUDE are manual overrides applied on top. All names are lowercase.
LABELED_PLAYER_COUNT = 30
LABEL_MIN_PER_CLUSTER = 3
LABEL_FONT_SIZE = 14
LABEL_REFERENCE_MODELS = {"ct": "kmeans_k3", "t": "kmeans_k4"}
LABEL_INCLUDE = ("karrigan",)
LABEL_EXCLUDE = ("twistzz",)

RANDOM_STATE = 101705

                                                                         
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


@lru_cache(maxsize=1)
def labeled_player_keys() -> frozenset:
    """Name keys of the players annotated on the PCA plots.

    Prominence is taken from the expert file's big-event rating, averaged over
    the two sides, and restricted to players who appear on both the CT and T
    side of the dataset. The set is computed once and reused by both sides, so
    the CT and T plots annotate the same players. Returns an empty set when the
    inputs are missing, which leaves every point labeled as before.
    """
    profile_path = next(
        (
            base / candidate
            for base in (Path.cwd(), Path(__file__).resolve().parent)
            for candidate in PLAYER_PROFILE_CANDIDATES
            if (base / candidate).is_file()
        ),
        None,
    )
    gt_path = next((path for path in _candidate_paths() if path.exists()), None)
    if profile_path is None or gt_path is None:
        return frozenset()

    profiles = pd.read_csv(profile_path)
    if not {"player_name", "side"}.issubset(profiles.columns):
        return frozenset()
    sides_per_player = profiles.assign(_key=profiles["player_name"].map(_name_key)).groupby("_key")["side"].nunique()
    on_both_sides = set(sides_per_player[sides_per_player >= 2].index)
    if not on_both_sides:
        return frozenset()

    raw = _read_csv_flexible(gt_path)
    if "Name" not in raw.columns:
        return frozenset()
    for pair in (
        ("CT BigEventsCS2 Rating", "T BigEventsCS2 Rating"),
        ("CT Last12 Rating", "T Last12 Rating"),
    ):
        if all(column in raw.columns for column in pair):
            rating = raw[list(pair)].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            break
    else:
        return frozenset()

    ranked = pd.DataFrame({"_key": raw["Name"].map(_name_key), "rating": rating})
    ranked = ranked[ranked["_key"].isin(on_both_sides)].dropna(subset=["rating"])
    ranked = ranked.sort_values("rating", ascending=False).drop_duplicates("_key")
    rating_by_key = dict(zip(ranked["_key"], ranked["rating"]))

    cells = _label_reference_cells(rating_by_key)
    if not cells:
        by_rating = list(ranked["_key"])
    else:
        by_rating = _round_robin_by_cell(cells, LABELED_PLAYER_COUNT)

    picked = [key for key in by_rating[:LABELED_PLAYER_COUNT] if key not in LABEL_EXCLUDE]
    for key in LABEL_INCLUDE:
        if key in rating_by_key and key not in picked:
            picked.append(key)
    picked = picked[:LABELED_PLAYER_COUNT]
    picked = _enforce_min_per_cell(picked, cells, rating_by_key)
    return frozenset(picked)


def _label_offsets() -> tuple:
    """Candidate label positions, in points, ordered from nearest ring outward."""
    offsets = [(9, 9), (9, -11), (-9, 9), (-9, -11)]
    for radius in (26, 44, 64, 88, 116, 148, 185, 230, 280, 330):
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (0.72, 0.72), (0.72, -0.72), (-0.72, 0.72), (-0.72, -0.72),
            (0.94, 0.34), (0.94, -0.34), (-0.94, 0.34), (-0.94, -0.34),
            (0.34, 0.94), (0.34, -0.94), (-0.34, 0.94), (-0.34, -0.94),
        ):
            offsets.append((round(dx * radius), round(dy * radius)))
    return tuple(offsets)


_LABEL_OFFSETS = _label_offsets()
_MARKER_PAD = 6.0


def _box_overlap_area(first: Bbox, second: Bbox) -> float:
    width = min(first.x1, second.x1) - max(first.x0, second.x0)
    height = min(first.y1, second.y1) - max(first.y0, second.y0)
    return width * height if width > 0 and height > 0 else 0.0


def _annotate_without_overlap(fig, ax, labels, points=None, fontsize: float = LABEL_FONT_SIZE) -> None:
    """Draw player names so they cover neither each other nor the markers.

    Every candidate offset is scored on how much it would overlap the markers
    and the names already placed, how far outside the axes it would fall, and
    how far it sits from its own point. Covering an unrelated marker is
    penalized far more than covering another label, since a name sitting on
    top of the wrong dot is actively misleading rather than just crowded. The
    first candidate that collides with nothing wins outright; otherwise the
    least-bad one is used, so a crowded region degrades gracefully instead of
    stacking names on one spot. Every name gets a white halo so it stays
    legible over the coloured cluster hulls and over any marker it partially
    covers, and every name gets a thin arrow back to its own point so which
    marker it names is never ambiguous.
    """
    if not labels:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    occupied = []
    if points is not None and len(points):
        for px, py in ax.transData.transform(points):
            occupied.append(Bbox.from_extents(
                px - _MARKER_PAD, py - _MARKER_PAD, px + _MARKER_PAD, py + _MARKER_PAD,
            ))

    axes_box = ax.get_window_extent(renderer=renderer)

    def _make(text: str, x: float, y: float, dx: float, dy: float):
        return ax.annotate(
            text, (x, y),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, zorder=7,
            ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
            va="bottom" if dy > 0 else ("top" if dy < 0 else "center"),
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white", edgecolor="none", alpha=0.78,
            ),
            arrowprops=dict(
                arrowstyle="->", linewidth=0.9, color="0.3",
                shrinkA=0, shrinkB=_MARKER_PAD,
                mutation_scale=10,
            ),
        )

    placed = []
    for text, x, y in sorted(labels, key=lambda item: (-item[2], item[1])):
        best_offset = None
        best_score = None
        for dx, dy in _LABEL_OFFSETS:
            artist = _make(text, x, y, dx, dy)
            box = artist.get_window_extent(renderer=renderer).expanded(1.05, 1.3)
            artist.remove()

            collision = sum(_box_overlap_area(box, other) for other in occupied) * 4.0
            collision += sum(_box_overlap_area(box, other) for other in placed) * 4.5
            outside = (
                max(0.0, axes_box.x0 - box.x0) + max(0.0, box.x1 - axes_box.x1)
                + max(0.0, axes_box.y0 - box.y0) + max(0.0, box.y1 - axes_box.y1)
            )
            score = collision + outside * 60.0 + (abs(dx) + abs(dy)) * 1.4
            if best_score is None or score < best_score:
                best_offset, best_score = (dx, dy), score
            if collision == 0.0 and outside == 0.0:
                break

        artist = _make(text, x, y, *best_offset)
        placed.append(artist.get_window_extent(renderer=renderer).expanded(1.05, 1.3))


def _label_reference_cells(rating_by_key: dict) -> dict:
    """Rated players grouped by (side, cluster) of the reference model, best first."""
    cells: dict = {}
    for side, model in LABEL_REFERENCE_MODELS.items():
        path = OUTPUT_DIR / side / f"{model}_player_clusters.csv"
        if not path.is_file():
            return {}
        clusters = pd.read_csv(path)
        if not {"player_name", "cluster"}.issubset(clusters.columns):
            return {}
        clusters["_key"] = clusters["player_name"].map(_name_key)
        for cluster_id, group in clusters[clusters["cluster"] != -1].groupby("cluster"):
            members = [key for key in group["_key"] if key in rating_by_key]
            cells[(side, int(cluster_id))] = sorted(members, key=lambda k: -rating_by_key[k])
    return cells


def _round_robin_by_cell(cells: dict, limit: int) -> list:
    """Take the best remaining player from each cluster in turn, both sides at once."""
    remaining = {cell: list(members) for cell, members in cells.items()}
    picked: list = []
    while len(picked) < limit:
        progressed = False
        for cell in sorted(remaining):
            while remaining[cell]:
                key = remaining[cell].pop(0)
                if key in picked:
                    continue
                picked.append(key)
                progressed = True
                break
            if len(picked) >= limit:
                break
        if not progressed:
            break
    return picked


def _enforce_min_per_cell(picked: list, cells: dict, rating_by_key: dict) -> list:
    """Top up any cluster below the floor, trading against the most crowded one."""
    if not cells:
        return picked
    picked = list(picked)

    def counts() -> dict:
        return {cell: sum(1 for key in picked if key in set(members)) for cell, members in cells.items()}

    for cell, members in sorted(cells.items()):
        while counts()[cell] < min(LABEL_MIN_PER_CLUSTER, len(members)):
            addition = next((key for key in members if key not in picked), None)
            if addition is None:
                break
            current = counts()
            crowded = max(current, key=lambda c: (current[c], c))
            droppable = [
                key for key in picked
                if key not in LABEL_INCLUDE
                and key in set(cells[crowded])
                and all(
                    current[other] - 1 >= min(LABEL_MIN_PER_CLUSTER, len(cells[other]))
                    for other in cells
                    if key in set(cells[other])
                )
            ]
            if droppable:
                picked.remove(min(droppable, key=lambda k: rating_by_key.get(k, 0.0)))
            picked.append(addition)
    return picked


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
            r1 = _clean_text(row.get("gt_role_1", ""))
            r2 = _clean_text(row.get("gt_role_2", ""))
            return r1 if _is_valid_role(r1) else r2 if _is_valid_role(r2) else ""

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


def ground_truth_accuracy(
    labels: np.ndarray,
    players_df: pd.DataFrame,
) -> tuple[dict[str, float] | None, pd.DataFrame | None]:
    if "gt_role_1" not in players_df.columns:
        return None, None

    df = players_df.copy()
    df["_label"] = labels
    df["_resolved"] = resolve_gt_label(players_df, labels)

    rows = []
    total_match = total_players = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for cluster_id, group in df.groupby("_label"):
        valid = group[group["_resolved"].str.strip().str.len() > 0]
        if valid.empty:
            continue
        n_players = int(len(valid))
        if cluster_id == -1:
            dominant = "Noise (unassigned)"
            n_match = 0
        else:
            dominant = valid["_resolved"].value_counts().idxmax()
            n_match = int((valid["_resolved"] == dominant).sum())
        rows.append({
            "cluster": cluster_id,
            "dominant_role": dominant,
            "n_players": n_players,
            "n_match": n_match,
            "accuracy": round(n_match / n_players, 4),
        })
        total_match += n_match
        total_players += n_players
        y_true.extend(valid["_resolved"].tolist())
        y_pred.extend([dominant] * n_players)

    if not rows or total_players == 0:
        return None, None

    role_counts = pd.Series(y_true).value_counts()
    scores = {
        "accuracy": round(total_match / total_players, 4),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "baseline_accuracy": float(role_counts.max() / total_players),
    }
    return scores, pd.DataFrame(rows).sort_values("cluster")


def ground_truth_ari(labels: np.ndarray, players_df: pd.DataFrame) -> float | None:
    if "gt_role_1" not in players_df.columns:
        return None

    resolved = resolve_gt_label(players_df, labels)
    mask = (
        resolved.str.strip().str.len().gt(0)
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
    try:
        X = select_model_features(df)
    except ValueError:
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
    columns = [
        "name", "method", "k", "noise_pct", "stability_mean",
        "stability_std", "composite_score",
    ]
    stab = results_df[columns].dropna(subset=["stability_mean"]).copy()
    if stab.empty:
        return

    # The HDBSCAN grid contains many equivalent two-cluster solutions. Showing
    # every row makes the chart too tall to use in the report, so retain one
    # representative: the most stable result, with composite score as a tie-break.
    non_hdbscan = stab[stab["method"] != "hdbscan"]
    hdbscan = (
        stab[stab["method"] == "hdbscan"]
        .sort_values(
            ["stability_mean", "composite_score", "name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        .head(1)
    )
    omitted_hdbscan = max(0, int((stab["method"] == "hdbscan").sum()) - len(hdbscan))
    stab = (
        pd.concat([non_hdbscan, hdbscan], ignore_index=True)
        .sort_values(["stability_mean", "name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )

    eligible = non_hdbscan[
        non_hdbscan["stability_mean"].isna()
        | (non_hdbscan["stability_mean"] >= 0.50)
    ]
    if eligible.empty:
        eligible = non_hdbscan
    selected_name = None
    if not eligible.empty:
        selected_name = eligible.sort_values(
            ["composite_score", "name"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]["name"]

    def display_name(row: pd.Series) -> str:
        k = int(row["k"])
        if row["method"] == "hdbscan":
            return f"HDBSCAN representative (k={k}, {row['noise_pct']:.0%} noise)"
        return f"{row['method'].upper()} k={k}"

    stab["display_name"] = stab.apply(display_name, axis=1)
    stab["selected"] = stab["name"].eq(selected_name)

    fig_height = max(5.5, 0.48 * len(stab) + 2.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    ax.axvspan(0.00, 0.50, color="#FDECEC", alpha=0.75, zorder=0)
    ax.axvspan(0.50, 0.80, color="#FFF4CC", alpha=0.75, zorder=0)
    ax.axvspan(0.80, 1.00, color="#E8F5E9", alpha=0.75, zorder=0)

    y_pos = np.arange(len(stab))
    colors = [METHOD_COLORS.get(m, "#999999") for m in stab["method"]]
    edgecolors = ["#111111" if selected else "none" for selected in stab["selected"]]
    linewidths = [2.0 if selected else 0.0 for selected in stab["selected"]]

    ax.barh(
        y_pos, stab["stability_mean"], xerr=stab["stability_std"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
        error_kw={"ecolor": "#222222", "elinewidth": 1.2, "capsize": 3},
        zorder=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stab["display_name"], fontsize=10)
    for label, selected in zip(ax.get_yticklabels(), stab["selected"]):
        if selected:
            label.set_fontweight("bold")
    ax.invert_yaxis()                                

    ax.set_xlabel("Mean stability ARI (+/- 1 SD)")
    ax.set_title(f"{side.upper()} Model Stability from 80% Subsampling", pad=22)
    ax.set_xlim(0, 1.12)

    ax.axvline(0.50, color="#444444", linestyle="--", linewidth=1.2, zorder=3)
    ax.axvline(0.80, color="#444444", linestyle=":", linewidth=1.2, zorder=3)
    ax.grid(axis="x", color="white", linewidth=1, alpha=0.9, zorder=1)

    tier_transform = ax.get_xaxis_transform()
    ax.text(0.25, 1.01, "Low", ha="center", va="bottom", transform=tier_transform, fontsize=9)
    ax.text(0.65, 1.01, "Moderate", ha="center", va="bottom", transform=tier_transform, fontsize=9)
    ax.text(0.90, 1.01, "High", ha="center", va="bottom", transform=tier_transform, fontsize=9)

    for y, row in stab.iterrows():
        ax.text(
            0.025,
            y,
            f"{row['stability_mean']:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if row["selected"] else "normal",
            color="white",
            zorder=4,
        )

    from matplotlib.patches import Patch
    method_handles = [
        Patch(color=METHOD_COLORS[m], label=m.upper())
        for m in ("kmeans", "gmm", "hdbscan")
        if m in stab["method"].unique()
    ]
    selected_handle = Patch(facecolor="white", edgecolor="#111111", linewidth=2, label="Selected model")
    ax.legend(handles=method_handles + [selected_handle], loc="lower right", fontsize=9)

    if omitted_hdbscan:
        fig.text(
            0.01,
            0.01,
            f"One representative HDBSCAN result is shown; {omitted_hdbscan} grid rows are omitted.",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#555555",
        )

    plt.tight_layout(rect=(0, 0.035, 1, 1))

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
    ax.set_title(f"Composite Score: {side.upper()}")

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

    print("\nStability tiers (subsampling ARI):")
    for _, row in stab.iterrows():
        print(
            f"  {row['name']:<30}  mean={row['stability_mean']:.3f}  "
            f"std={row['stability_std']:.3f}  [{tier(row['stability_mean'])}]"
        )
                                                                

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

IGL_HIGHLIGHT_COLOR = "#0077FF"

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

        clean_suffix = " Clean" if clean else ""

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(
            f"PCA - Ground-Truth Roles vs Cluster Boundaries{clean_suffix}\n"
            f"{side.upper()} {name}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(title="GT Role", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)
        plt.tight_layout()

        labeled = labeled_player_keys()
        to_label = [
            (str(row["player_name"]), float(row["pc1"]), float(row["pc2"]))
            for _, row in plot_data.iterrows()
            if not labeled or _name_key(row["player_name"]) in labeled
        ]
        _annotate_without_overlap(
            fig, ax, to_label,
            points=plot_data[["pc1", "pc2"]].to_numpy(),
        )

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
        for prefix in ("gt_side", "gt_general", "gt_igl"):
            cols += [
                f"{prefix}_ari",
                f"{prefix}_accuracy",
                f"{prefix}_balanced_accuracy",
                f"{prefix}_macro_f1",
                f"{prefix}_baseline_accuracy",
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
    quality, breakdown = ground_truth_accuracy(labels, df_with_gt)
    matched_n, match_rate = ground_truth_match_stats(df_with_gt)

    if ari is not None:
        results_df.loc[results_df["name"] == model_name, f"{prefix}_ari"] = ari
    if quality is not None:
        for metric, value in quality.items():
            results_df.loc[
                results_df["name"] == model_name,
                f"{prefix}_{metric}",
            ] = value
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

    side_plot_dir = PLOTS_DIR / side
    side_plot_dir.mkdir(parents=True, exist_ok=True)
    generated_plot_patterns = [
        "stability_tiers.png",
        "composite_scores.png",
        "*_cluster_vs_gt.png",
        "*_cluster_vs_gt_clean.png",
        "*_pca.png",
        "*_radar.png",
        "*_zscore_heatmap.png",
        "*_feature_importance.png",
    ]
    for pattern in generated_plot_patterns:
        for path in side_plot_dir.glob(pattern):
            try:
                path.unlink()
            except PermissionError:
                print(f"[warn] could not delete {path} (open in another program?) - overwriting instead")

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
    best_models = eligible.sort_values(
        ["composite_score", "name"],
        ascending=[False, True],
        kind="mergesort",
    ).head(2)

    print(f"\n{'=' * 60}")
    print(f"{side.upper()} EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    display_cols = [
        "name", "k", "silhouette", "davies_bouldin",
        "stability_mean", "stability_std", "composite_score",
    ]
    if gt is not None:
        display_cols += [
            "gt_side_ari", "gt_side_accuracy", "gt_side_baseline_accuracy",
            "gt_general_ari", "gt_general_accuracy", "gt_general_baseline_accuracy",
            "gt_igl_ari", "gt_igl_accuracy", "gt_igl_balanced_accuracy",
            "gt_igl_macro_f1", "gt_igl_baseline_accuracy",
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
                    balanced = row.get(f"{prefix}_balanced_accuracy", float("nan"))
                    macro_f1 = row.get(f"{prefix}_macro_f1", float("nan"))
                    baseline = row.get(f"{prefix}_baseline_accuracy", float("nan"))
                    ari = row.get(f"{prefix}_ari", float("nan"))
                    print(
                        f"\nGT accuracy - {side.upper()} {name} ({label}; "
                        f"overall={acc:.1%}  baseline={baseline:.1%}  "
                        f"balanced={balanced:.1%}  macro-F1={macro_f1:.3f}  ARI={ari:.4f}):"
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
