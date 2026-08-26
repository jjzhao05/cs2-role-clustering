import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="CS2 Role Discovery",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent


INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
BORDER = "rgba(11,11,11,0.10)"

ROLE_COLOR = {
    "AWPer": "#2a78d6",
    "Rotator": "#eb6834",
    "Anchor": "#1baf7a",
    "Lurker": "#4a3aa7",
    "Spacetaker": "#eda100",
    "Spacetaker (Aggressive)": "#eda100",
    "Spacetaker (IGL)": "#e87ba4",
    "Noise (unassigned)": "#a9a79d",
}
ROLE_SYMBOL = {
    "AWPer": "circle",
    "Rotator": "circle",
    "Anchor": "circle",
    "Lurker": "circle",
    "Spacetaker": "circle",
    "Spacetaker (Aggressive)": "circle",
    "Spacetaker (IGL)": "circle",
}
METHOD_COLOR = {"kmeans": "#2a78d6", "gmm": "#eb6834", "hdbscan": "#1baf7a"}

# ISO 3166-1 alpha-2 codes for the countries in Roles.csv, used to build flag emoji.
COUNTRY_CODE = {
    "Argentina": "AR", "Belarus": "BY", "Bosnia and Herzegovina": "BA", "Brazil": "BR",
    "Bulgaria": "BG", "Canada": "CA", "Chile": "CL", "China": "CN", "Czech Republic": "CZ",
    "Denmark": "DK", "Estonia": "EE", "Finland": "FI", "France": "FR", "Germany": "DE",
    "Guatemala": "GT", "Hungary": "HU", "Israel": "IL", "Kazakhstan": "KZ", "Kosovo": "XK",
    "Latvia": "LV", "Lithuania": "LT", "Mongolia": "MN", "Poland": "PL", "Romania": "RO",
    "Russia": "RU", "Slovakia": "SK", "South Africa": "ZA", "Spain": "ES", "Sweden": "SE",
    "Turkey": "TR", "Ukraine": "UA", "United Kingdom": "GB", "United States": "US",
    "Uruguay": "UY",
}
SEQ_BLUE = "#2a78d6"
MUTED_BAR = "#c3c2b7"

CHART_FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY)
PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
    "showAxisDragHandles": False,
}

# Best-scoring configuration per method per side (top composite_score in model_scores.csv).
MODELS = {
    "kmeans": {"ct": "kmeans_k3", "t": "kmeans_k4"},
    "gmm": {"ct": "gmm_k3", "t": "gmm_k4"},
    "hdbscan": {"ct": "hdbscan_mcs10_ms2", "t": "hdbscan_mcs10_ms1"},
}
METHOD_LABEL = {"kmeans": "KMeans", "gmm": "Gaussian Mixture", "hdbscan": "HDBSCAN"}
BASE_MODEL = MODELS["kmeans"]  # model used by the Player Explorer
SIDE_LABEL = {"ct": "CT Side", "t": "T Side"}
SIDE_AVG_LABEL = {"ct": "CT-side average", "t": "T-side average"}

# Feature -> stat category, following the project README's feature-group
# taxonomy (Combat / Opening Duels / Trading / Utility / Positioning /
# Movement), plus a Consistency group for the across-demo std-dev features
# (matches the "consistency_only" ablation group in the pipeline).
CATEGORY_ORDER = [
    "Combat", "Opening Duels", "Trading", "Utility", "Positioning", "Movement", "Consistency",
]
FEATURE_CATEGORY = {
    "kpr": "Combat", "survival_rate": "Combat", "damage_per_round": "Combat",
    "damage_taken_per_round": "Combat", "multi_kill_rate": "Combat",
    "rifle_kill_share": "Combat", "awp_kill_share": "Combat", "assists_per_round": "Combat",
    "opening_kill_rate": "Opening Duels", "opening_death_rate": "Opening Duels",
    "opening_duel_success": "Opening Duels", "first_contact_rate": "Opening Duels",
    "first_contact_received_rate": "Opening Duels",
    "trade_kill_rate": "Trading", "death_traded_rate": "Trading",
    "util_damage_per_round": "Utility", "flash_assists_per_round": "Utility",
    "he_grenades_per_round": "Utility", "flashbangs_per_round": "Utility",
    "smokes_per_round": "Utility", "fire_nades_per_round": "Utility", "decoys_per_round": "Utility",
    "avg_distance_to_enemy": "Positioning", "avg_distance_to_team_centroid": "Positioning",
    "relative_team_centroid_distance": "Positioning", "avg_distance_to_closest_teammate": "Positioning",
    "time_near_enemy_rate": "Positioning",
    "avg_distance_moved_per_round": "Movement", "time_stationary_rate": "Movement",
    "adr_std": "Consistency", "avg_distance_to_enemy_std": "Consistency",
    "avg_distance_to_team_centroid_std": "Consistency", "relative_team_centroid_distance_std": "Consistency",
    "avg_distance_moved_per_round_std": "Consistency", "avg_distance_to_closest_teammate_std": "Consistency",
    "time_near_enemy_rate_std": "Consistency", "time_stationary_rate_std": "Consistency",
    "kast_std": "Consistency", "impact_std": "Consistency", "rating_std": "Consistency",
}


def feature_category(feature: str) -> str:
    return FEATURE_CATEGORY.get(feature, "Other")


def sort_by_category(features: list) -> list:
    """Stable sort: category (in CATEGORY_ORDER), then original (importance) order within it."""
    return sorted(
        features,
        key=lambda f: (
            CATEGORY_ORDER.index(feature_category(f)) if feature_category(f) in CATEGORY_ORDER else len(CATEGORY_ORDER),
            features.index(f),
        ),
    )

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAGE}; }}
    .block-container {{ padding-top: 2rem; max-width: 1200px; }}
    div[data-testid="stMetric"] {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    .role-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }}
    .finding-line {{ color: {INK_SECONDARY}; font-size: 0.95rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_roles() -> pd.DataFrame:
    """Roles.csv is UTF-16, tab-separated (expert reference annotations)."""
    path = ROOT / "Roles.csv"
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    df = df.rename(
        columns={
            "Name": "player_name",
            "Team": "team",
            "Country": "country",
            "CT Role": "ct_expert_role",
            "T Role": "t_expert_role",
            "Role": "general_expert_role",
        }
    )
    keep = ["player_name", "team", "country", "ct_expert_role", "t_expert_role", "general_expert_role"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["_merge_key"] = df["player_name"].astype(str).str.strip().str.lower()
    df["is_igl"] = df["general_expert_role"].astype(str).str.contains("IGL", na=False)
    return df


@st.cache_data
def load_side_accuracy_labels(side: str, model_name: str) -> dict:
    """Build a human role-name-per-cluster mapping from the gt_side_accuracy csv,
    disambiguating duplicate dominant_role names (e.g. two 'Spacetaker' clusters
    on T side) using the gt_igl_accuracy csv."""
    side_path = ROOT / "outputs" / side / f"{model_name}_gt_side_accuracy.csv"
    igl_path = ROOT / "outputs" / side / f"{model_name}_gt_igl_accuracy.csv"
    side_acc = pd.read_csv(side_path)
    igl_acc = pd.read_csv(igl_path)
    counts = side_acc["dominant_role"].value_counts()
    labels = {}
    for _, row in side_acc.iterrows():
        role = row["dominant_role"]
        if counts.get(role, 0) > 1:
            igl_row = igl_acc[igl_acc["cluster"] == row["cluster"]]
            is_igl_cluster = (not igl_row.empty) and igl_row.iloc[0]["dominant_role"] == "IGL"
            role_label = f"{role} (IGL)" if is_igl_cluster else f"{role} (Aggressive)"
        else:
            role_label = role
        labels[int(row["cluster"])] = role_label
    return labels


@st.cache_data
def load_player_clusters(side: str, model_name: str = "") -> pd.DataFrame:
    model_name = model_name or BASE_MODEL[side]
    path = ROOT / "outputs" / side / f"{model_name}_player_clusters.csv"
    df = pd.read_csv(path)
    labels = load_side_accuracy_labels(side, model_name)
    df["role"] = df["cluster"].map(labels).fillna("Noise (unassigned)")

    roles = load_roles()
    df["_merge_key"] = df["player_name"].astype(str).str.strip().str.lower()
    df = df.merge(roles.drop(columns=["player_name"]), on="_merge_key", how="left")

    expert_col = "ct_expert_role" if side == "ct" else "t_expert_role"
    df["expert_role"] = df[expert_col]
    # "Mixed" (CT) / "Flex" (T) mean the expert annotation itself didn't commit this
    # player to one role, so they are excluded from the match/mismatch comparison
    # rather than being counted as an automatic mismatch.
    is_ambiguous_label = df["expert_role"].isin(["Mixed", "Flex"])
    df["role_match"] = (df["role"].str.split(" (", regex=False).str[0] == df["expert_role"]).astype("boolean")
    df.loc[is_ambiguous_label, "role_match"] = pd.NA
    # HDBSCAN leaves some players unassigned; they have no role to compare.
    df.loc[df["role"] == "Noise (unassigned)", "role_match"] = pd.NA
    return df


@st.cache_data
def load_feature_importance(side: str, model_name: str = "") -> pd.DataFrame:
    model_name = model_name or BASE_MODEL[side]
    path = ROOT / "outputs" / side / f"{model_name}_feature_importance.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "feature"})
    df = df[df["feature"] != "player_page"]
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


@st.cache_data
def load_model_scores(side: str) -> pd.DataFrame:
    path = ROOT / "outputs" / side / "model_scores.csv"
    return pd.read_csv(path)


@st.cache_data
def load_ablations() -> pd.DataFrame:
    path = ROOT / "outputs" / "ablations" / "ablation_results_best_all_sides.csv"
    return pd.read_csv(path)


@st.cache_data
def load_gt_accuracy(side: str, kind: str, model_name: str = "") -> pd.DataFrame:
    model_name = model_name or BASE_MODEL[side]
    path = ROOT / "outputs" / side / f"{model_name}_gt_{kind}_accuracy.csv"
    df = pd.read_csv(path)
    labels = load_side_accuracy_labels(side, model_name)
    df["role"] = df["cluster"].map(labels)
    return df


try:
    _load_error = None
    for _s in ("ct", "t"):
        load_player_clusters(_s)
except Exception as exc:  # pragma: no cover - surfaced in the UI
    _load_error = exc

if _load_error is not None:
    st.error(
        "Could not load the precomputed pipeline outputs. Make sure you are "
        "running `streamlit run app.py` from the cs2-role-classifier project "
        f"root (expected outputs/, output.csv, Roles.csv next to app.py).\n\n{_load_error}"
    )
    st.stop()

NON_FEATURE_COLS = {
    "player_name", "side", "rounds_played", "demo_count", "cluster", "pc1", "pc2",
    "role", "_merge_key", "team", "country", "ct_expert_role", "t_expert_role",
    "general_expert_role", "is_igl", "expert_role", "role_match",
}


def feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def role_color(role: str) -> str:
    return ROLE_COLOR.get(role, INK_MUTED)


def role_symbol(role: str) -> str:
    return ROLE_SYMBOL.get(role, "circle")


def styled_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=CHART_FONT,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        dragmode=False,  # disable click-drag box/lasso zoom, hover + legend clicks stay active
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=INK_MUTED))
    return fig


st.title("Unsupervised Role Discovery in Professional CS2")
st.markdown(
    "<div class='finding-line'>Can professional Counter-Strike 2 player roles be discovered "
    "from behavioral and positional gameplay statistics alone, without labels?</div>",
    unsafe_allow_html=True,
)
st.write("")

st.markdown(
    "<div class='finding-line'>"
    "<b>Main finding:</b> AWPers are highly separable on both sides. CT riflers split into a "
    "meaningful Rotator/Anchor structure driven mainly by positioning. T-side Lurkers separate "
    "cleanly, while Spacetakers divide into an aggressive group and a utility-heavy, IGL group."
    "</div>",
    unsafe_allow_html=True,
)
st.divider()

tab_player, tab_pipeline = st.tabs(["Player Explorer", "Clusters, Models & Ablations"])

with tab_pipeline:
    tab_clusters, tab_models, tab_ablation = st.tabs(
        ["Clusters & Roles", "Model Comparison", "Feature Ablations"]
    )


def country_flag_html(country: str) -> str:
    """Flag image for a country. Windows browsers do not draw flag emoji, so this
    uses flagcdn.com and falls back to an empty box if the image cannot load."""
    code = COUNTRY_CODE.get(country)
    if not code:
        return ""
    # Rendered as an <img> with object-fit so flags whose native aspect ratio
    # doesn't match the 24x18 badge box (e.g. Russia/Ukraine at 3:2) are cropped
    # symmetrically instead of being squashed/tiled the way a CSS background-image
    # with background-size:cover was (bottom stripe bleeding into the top).
    return (
        f'<img src="https://flagcdn.com/h24/{code.lower()}.png" alt="{country}" '
        f'style="display:inline-block;width:24px;height:18px;object-fit:cover;'
        f'object-position:center;border-radius:2px;border:1px solid {BORDER};'
        f'vertical-align:middle;" onerror="this.style.visibility=\'hidden\'" />'
    )


def player_info_html(player: str) -> str:
    """Player name + team/country badges, shown once above both side panels."""
    roles = load_roles()
    # Demo-derived names are lowercased, expert names are not (ZywOo vs zywoo),
    # so match on the same case-folded key used for the cluster/roles merge.
    row = roles[roles["_merge_key"] == str(player).strip().lower()]
    badge_style = (
        f"display:inline-flex;align-items:center;gap:8px;background:{PAGE};"
        f"border:1px solid {BORDER};border-radius:8px;padding:5px 12px;"
        f"font-size:1rem;color:{INK_PRIMARY};white-space:nowrap;"
    )
    badges = ""
    if not row.empty:
        team = row.iloc[0]["team"]
        country = row.iloc[0]["country"]
        if isinstance(team, str):
            badges += f"<span style='{badge_style}'><b>{team}</b></span>"
        if isinstance(country, str):
            badges += f"<span style='{badge_style}'>{country_flag_html(country)}<b>{country}</b></span>"
    return (
        "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:12px 0 6px;'>"
        f"<span style='color:{INK_PRIMARY};font-size:1.5rem;font-weight:700;'>{player}</span>"
        f"{badges}</div>"
    )


def render_player_side(side: str, player: str, show_no_label_caption: bool = True, show_chart_captions: bool = True) -> None:
    """One side's panel for the selected player: role card, feature profile, PCA map."""
    df = load_player_clusters(side)
    match_rows = df[df["player_name"] == player]
    if match_rows.empty:
        st.markdown(
            f"""
            <div class='role-card'>
                <div style='font-size:0.85rem;color:{INK_MUTED};text-transform:uppercase;letter-spacing:0.04em;'>
                    {SIDE_LABEL[side]} role assignment
                </div>
                <div style='font-size:1.1rem;font-weight:600;color:{INK_MUTED};'>No {SIDE_LABEL[side]} data</div>
                <div style='color:{INK_SECONDARY};font-size:0.9rem;'>{player} was not clustered on this side.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    row = match_rows.iloc[0]
    role = row["role"]
    expert_role = row.get("expert_role", None)

    st.markdown(
        f"""
        <div class='role-card'>
            <div style='font-size:0.85rem;color:{INK_MUTED};text-transform:uppercase;letter-spacing:0.04em;'>
                {SIDE_LABEL[side]} role assignment
            </div>
            <div style='font-size:1.6rem;font-weight:700;color:{role_color(role)};'>{role}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if isinstance(expert_role, str):
        if expert_role in ("Mixed", "Flex"):
            st.caption(
                f"Expert reference label: **{expert_role}**. "
                "Not scored: the expert label itself is ambiguous."
            )
        else:
            match = "matches" if row["role_match"] else "differs from"
            st.caption(f"Expert reference label: **{expert_role}**. Cluster assignment {match} the expert label.")
    elif show_no_label_caption:
        st.caption("No expert reference label available for this player (not in the curated annotation set).")

    avg_label = SIDE_AVG_LABEL[side]
    st.markdown(f"**Top-10 feature z-scores vs. {avg_label}**")
    fi = load_feature_importance(side)
    top_feats = sort_by_category([f for f in fi["feature"].head(10).tolist() if f in df.columns])
    mu = df[top_feats].mean()
    sigma = df[top_feats].std().replace(0, np.nan)
    z = ((row[top_feats].astype(float) - mu) / sigma).fillna(0.0)
    # Reverse so the first category (Combat, etc.) reads top-to-bottom on a horizontal bar chart.
    z = z.iloc[::-1]
    labels = [f"{feature_category(f)} · {f.replace('_', ' ')}" for f in z.index]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=z.values,
            y=labels,
            orientation="h",
            marker_color=[role_color(role) if v >= 0 else MUTED_BAR for v in z.values],
            hovertemplate="%{y}: %{x:.2f} std vs " + avg_label + "<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=INK_MUTED, line_width=1)
    fig.update_layout(xaxis_title=f"Std. devs from {avg_label}", yaxis_title=None)
    fig.update_yaxes(tickfont=dict(color=INK_MUTED, size=10), automargin=True)
    st.plotly_chart(styled_fig(fig, height=380), width='stretch', config=PLOTLY_CONFIG, key=f"profile_{side}")
    if show_chart_captions:
        st.caption(
            "Top 10 features by that side's model importance, grouped by stat category. "
            "Positive means above that side's average."
        )

    st.markdown("**PCA cluster map with this player highlighted**")
    fig2 = go.Figure()
    for r, sub in df.groupby("role"):
        others = sub[sub["player_name"] != player]
        fig2.add_trace(
            go.Scatter(
                x=others["pc1"], y=others["pc2"], mode="markers", name=r,
                marker=dict(color=role_color(r), size=7, symbol=role_symbol(r), opacity=0.35, line=dict(width=0)),
                text=others["player_name"],
                hovertemplate="%{text} (" + r + ")<extra></extra>",
            )
        )
    fig2.add_trace(
        go.Scatter(
            x=match_rows["pc1"], y=match_rows["pc2"], mode="markers+text", name=player,
            marker=dict(color=role_color(role), size=16, symbol=role_symbol(role), line=dict(width=2, color=INK_PRIMARY)),
            text=[player], textposition="top center", showlegend=False,
            hovertemplate=f"{player} ({role})<extra></extra>",
        )
    )
    fig2.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(styled_fig(fig2, height=360), width='stretch', config=PLOTLY_CONFIG, key=f"pca_{side}")
    if show_chart_captions:
        st.caption("PCA projection of all clustered features, with this player highlighted.")


with tab_player:
    players = sorted(
        set(load_player_clusters("ct")["player_name"].dropna())
        | set(load_player_clusters("t")["player_name"].dropna())
    )

    if st.session_state.get("player_select") not in players:
        st.session_state["player_select"] = "device" if "device" in players else players[0]

    def _pick_random_player():
        st.session_state["player_select"] = random.choice(players)

    sel_col, btn_col, _spacer = st.columns([2, 1, 3], vertical_alignment="bottom")
    with sel_col:
        player = st.selectbox("Player", players, key="player_select")
    with btn_col:
        st.button("Random", on_click=_pick_random_player, width='stretch')

    st.markdown(player_info_html(player), unsafe_allow_html=True)

    st.write("")

    def _expert_role(side: str):
        df = load_player_clusters(side)
        match = df[df["player_name"] == player]
        if match.empty:
            return None
        return match.iloc[0].get("expert_role", None)

    both_unlabeled = not isinstance(_expert_role("ct"), str) and not isinstance(_expert_role("t"), str)
    if both_unlabeled:
        st.caption("No expert reference label available for this player (not in the curated annotation set).")

    ct_col, t_col = st.columns(2)
    with ct_col:
        render_player_side("ct", player, show_no_label_caption=not both_unlabeled)
    with t_col:
        render_player_side("t", player, show_no_label_caption=not both_unlabeled, show_chart_captions=False)


with tab_clusters:
    ctl_side, ctl_method = st.columns([1, 2])
    with ctl_side:
        side = st.radio("Side", ["ct", "t"], format_func=lambda s: SIDE_LABEL[s], horizontal=True, key="cluster_side")
    with ctl_method:
        method = st.radio(
            "Model", list(MODELS), format_func=lambda m: METHOD_LABEL[m], horizontal=True, key="cluster_method"
        )

    model_name = MODELS[method][side]
    df = load_player_clusters(side, model_name)
    order = df.groupby("role")["role"].count().sort_values(ascending=False).index.tolist()

    if method == "hdbscan":
        _mcs, _ms = re.match(r"hdbscan_mcs(\d+)_ms(\d+)", model_name).groups()
        config_label = f"{method} min cluster size {_mcs}, min samples {_ms}"
    else:
        config_label = f"{method} k = {len(order)}"

    scores = load_model_scores(side)
    score_row = scores[scores["name"] == model_name]
    if not score_row.empty:
        s = score_row.iloc[0]
        st.caption(
            f"{config_label} · silhouette {s['silhouette']:.3f} · stability {s['stability_mean']:.3f} · "
            f"composite {s['composite_score']:.3f} · external ARI {s['gt_ari']:.3f}"
        )
    if method == "hdbscan":
        st.caption(
            "HDBSCAN chooses its own cluster count and leaves low-density players unassigned "
            "(shown as a noise group). It converges to a broad AWPer/everyone-else split."
        )

    cols = st.columns(len(order))
    role_gt_side = load_gt_accuracy(side, "side", model_name)
    for i, r in enumerate(order):
        sub = df[df["role"] == r]
        gt_row = role_gt_side[role_gt_side["role"] == r]
        if r == "Noise (unassigned)" or gt_row.empty:
            acc_line = "not scored"
        else:
            acc_line = f"{gt_row.iloc[0]['accuracy'] * 100:.0f}% match vs. expert labels"
        with cols[i]:
            st.markdown(
                f"""
                <div class='role-card'>
                    <div style='font-weight:700;color:{role_color(r)};font-size:1.05rem;'>{r}</div>
                    <div style='color:{INK_SECONDARY};font-size:0.9rem;'>{len(sub)} players</div>
                    <div style='color:{INK_MUTED};font-size:0.85rem;'>{acc_line}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([3, 2])

    with left:
        st.subheader("PCA cluster map")
        fig = go.Figure()
        for r, sub in df.groupby("role"):
            fig.add_trace(
                go.Scatter(
                    x=sub["pc1"], y=sub["pc2"], mode="markers", name=r,
                    marker=dict(color=role_color(r), size=8, symbol=role_symbol(r), opacity=0.75, line=dict(width=1, color=SURFACE)),
                    text=sub["player_name"],
                    hovertemplate="%{text} (" + r + ")<extra></extra>",
                )
            )
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(styled_fig(fig, height=460), width='stretch', config=PLOTLY_CONFIG, key="cluster_pca")

    with right:
        st.subheader("Top-12 feature importances")
        fi = load_feature_importance(side, model_name)
        top = fi.head(12).sort_values("importance")
        fig3 = go.Figure(
            go.Bar(
                x=top["importance"], y=[f.replace("_", " ") for f in top["feature"]],
                orientation="h", marker_color=SEQ_BLUE,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig3.update_layout(xaxis_title="Feature importance", yaxis_title=None)
        st.plotly_chart(styled_fig(fig3, height=460), width='stretch', config=PLOTLY_CONFIG, key="cluster_importance")

    st.write("")
    avg_label = SIDE_AVG_LABEL[side]
    st.subheader(f"Feature z-score heatmap vs. {avg_label}")
    fi_full = load_feature_importance(side, model_name)
    heat_feats = fi_full["feature"].head(14).tolist()
    heat_feats = [f for f in heat_feats if f in df.columns]
    mu = df[heat_feats].mean()
    sigma = df[heat_feats].std().replace(0, np.nan)
    z = (df[heat_feats] - mu) / sigma
    z["role"] = df["role"]
    zmean = z.groupby("role")[heat_feats].mean()
    zmean = zmean.loc[order]

    fig4 = go.Figure(
        data=go.Heatmap(
            z=zmean.values,
            x=[f.replace("_", " ") for f in heat_feats],
            y=zmean.index.tolist(),
            colorscale=[[0, "#0d366b"], [0.5, SURFACE], [1, "#e34948"]],
            zmid=0,
            colorbar=dict(title="z-score"),
            text=zmean.round(2).values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=11, color=INK_PRIMARY),
            hovertemplate="%{y}/%{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig4.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(styled_fig(fig4, height=340), width='stretch', config=PLOTLY_CONFIG, key="cluster_heatmap")
    st.caption(f"Blue = below the {avg_label}, red = above. Built from the same per-player features used for clustering.")


with tab_models:
    st.subheader("Composite score and accuracy by method")
    st.caption(
        "Composite score blends internal cluster quality (silhouette, Davies-Bouldin) "
        "and subsampling stability. "
        "The best-scoring configuration per method is shown for each side. These are the same configurations "
        "you can inspect cluster-by-cluster in the Clusters & Roles tab."
    )

    rows = []
    for side in ("ct", "t"):
        scores = load_model_scores(side)
        best_per_method = scores.sort_values("composite_score", ascending=False).groupby("method").first().reset_index()
        best_per_method["side"] = side
        rows.append(best_per_method)
    all_best = pd.concat(rows, ignore_index=True)

    def method_bar(sub: pd.DataFrame, column: str, title: str, axis_title: str, x_max: float, key: str) -> None:
        sub = sub.sort_values(column, ascending=True)
        fig = go.Figure(
            go.Bar(
                x=sub[column], y=sub["method"].str.upper(),
                orientation="h",
                marker_color=[METHOD_COLOR.get(m, INK_MUTED) for m in sub["method"]],
                text=[f"{v:.3f}" for v in sub[column]],
                textposition="outside",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        )
        fig.update_layout(title=title, xaxis_title=axis_title, yaxis_title=None, xaxis_range=[0, x_max])
        st.plotly_chart(styled_fig(fig, height=280), width='stretch', config=PLOTLY_CONFIG, key=key)

    col_a, col_b = st.columns(2)
    for col, side in zip((col_a, col_b), ("ct", "t")):
        with col:
            sub = all_best[all_best["side"] == side]
            method_bar(
                sub, "composite_score", f"{SIDE_LABEL[side]}: composite score by method",
                "Composite score", 0.6, f"models_composite_{side}",
            )
            method_bar(
                sub, "gt_accuracy", f"{SIDE_LABEL[side]}: accuracy vs. expert labels",
                "External accuracy", 1.0, f"models_accuracy_{side}",
            )

    st.caption(
        "Composite score is internal (no labels used). Accuracy is external: the share of expert-labelled "
        "players falling in a cluster whose dominant role matches their label."
    )

    st.write("")
    st.subheader("Model metrics for the best config per method")
    display_cols = [
        "side", "method", "name", "k", "silhouette", "davies_bouldin",
        "stability_mean", "composite_score", "gt_ari", "gt_accuracy",
    ]
    display_cols = [c for c in display_cols if c in all_best.columns]
    table = all_best[display_cols].sort_values(["side", "composite_score"], ascending=[True, False])
    table = table.rename(columns={
        "side": "Side", "method": "Method", "name": "Config", "k": "k",
        "silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin",
        "stability_mean": "Stability", "composite_score": "Composite score",
        "gt_ari": "External ARI", "gt_accuracy": "External accuracy",
    })
    st.dataframe(
        table.style.format({
            "Silhouette": "{:.3f}", "Davies-Bouldin": "{:.3f}", "Stability": "{:.3f}",
            "Composite score": "{:.3f}", "External ARI": "{:.3f}", "External accuracy": "{:.1%}",
        }),
        width='stretch', hide_index=True,
    )
    st.caption(
        "KMeans produced the most interpretable, best-scoring role structures on both sides. "
        "GMM was less stable across subsamples. HDBSCAN converged to broad two-cluster solutions "
        "that are very stable but too coarse to separate individual roles."
    )


with tab_ablation:
    st.subheader("Composite score by feature-group ablation")
    st.caption(
        "Each bar is a full re-clustering with one feature group removed (or isolated), "
        "holding k fixed at the side's best value. 'full' is the baseline model using every feature group."
    )

    side = st.radio("Side", ["ct", "t"], format_func=lambda s: SIDE_LABEL[s], horizontal=True, key="ablation_side")
    ab = load_ablations()
    sub = ab[ab["side"] == side].sort_values("composite_score", ascending=True)

    colors = ["#0d366b" if a == "full" else SEQ_BLUE for a in sub["ablation"]]
    fig = go.Figure(
        go.Bar(
            x=sub["composite_score"], y=sub["ablation"].str.replace("_", " "),
            orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Composite score", yaxis_title=None)
    st.plotly_chart(styled_fig(fig, height=520), width='stretch', config=PLOTLY_CONFIG, key="ablation_chart")
    st.caption("Dark bar = the full-feature baseline. Bars below it mean that ablation hurt overall cluster quality. Bars above mean it helped.")

    st.write("")
    st.subheader("Ablation metrics table")
    show_cols = ["ablation", "n_features", "silhouette", "stability_mean", "gt_ari", "gt_purity", "composite_score"]
    show_cols = [c for c in show_cols if c in sub.columns]
    detail = sub[show_cols].sort_values("composite_score", ascending=False).rename(columns={
        "ablation": "Ablation", "n_features": "# Features", "silhouette": "Silhouette",
        "stability_mean": "Stability", "gt_ari": "External ARI", "gt_purity": "External purity",
        "composite_score": "Composite score",
    })
    st.dataframe(
        detail.style.format({
            "Silhouette": "{:.3f}", "Stability": "{:.3f}", "External ARI": "{:.3f}",
            "External purity": "{:.1%}", "Composite score": "{:.3f}",
        }),
        width='stretch', hide_index=True,
    )

    if side == "ct":
        st.caption(
            "CT side: removing positioning/movement features causes the largest drop in external agreement. "
            "The Rotator/Anchor split is fundamentally a spatial-behavior split."
        )
    else:
        st.caption(
            "T side: no single feature group reproduces the full role structure. Combat/weapon features drive "
            "raw separation, while positioning, contact, and utility features each contribute part of the picture."
        )

st.divider()
st.caption(
    "Data: 463 parsed professional CS2 demos across four S-tier events, 214 players. "
    "External reference labels from NER0cs's Positions Database (HLTV.org). "
    "This dashboard reads only precomputed pipeline outputs. No demo parsing or model fitting runs here."
)
