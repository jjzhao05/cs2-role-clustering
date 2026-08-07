from collections.abc import Iterable

import pandas as pd


MIN_ROUNDS_PLAYED = 100
CLUSTER_RANDOM_STATE = 101705
ABLATION_K = {"ct": 3, "t": 4}


# These columns identify a row, describe its exposure, or are intentionally
# excluded performance summaries. They must never enter clustering or a
# surrogate model trained to reproduce cluster assignments.
MODEL_EXCLUDE_COLUMNS = {
    "player_name",
    "side",
    "rounds_played",
    "demo_count",
    "adr",
    "kdr",
    # Exact derived combinations already represented by their component rates.
    "damage_diff_per_round",
    "trade_participation",
    "opening_duel_attempt_rate",
    "first_contact_participation_rate",
    "grenades_per_round",
}

RESULT_ARTIFACT_COLUMNS = {
    "cluster",
    "pc1",
    "pc2",
    "predicted_cluster",
    "gt_role_1",
    "gt_role_2",
    "gt_role_source",
    "gt_side_role",
    "gt_general_role",
    "gt_general_role_raw",
    "gt_igl_status",
    "gt_ct_role",
    "gt_t_role",
    "gt_match_key",
}


def select_model_features(
    df: pd.DataFrame,
    extra_exclude: Iterable[str] = (),
) -> pd.DataFrame:
    """Return the numeric, nonconstant feature matrix used across the project."""
    exclude = MODEL_EXCLUDE_COLUMNS | RESULT_ARTIFACT_COLUMNS | set(extra_exclude)
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.loc[:, X.nunique() > 1]
    if X.empty:
        raise ValueError("No usable feature columns.")
    return X
