from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
from awpy import Demo
from awpy.stats import adr, calculate_trades, kast, rating

# Constants and configuration

NEAR_ENEMY_RADIUS = 750.0
STATIONARY_STEP_DISTANCE = 1.0

# Number of worker processes for parallel demo parsing.
# Defaults to all logical CPU cores; lower this if memory is tight
# (each worker loads a full demo into memory simultaneously).
WORKERS = 6
    
TEAM_ALIASES = {
    "t": "t",
    "terrorist": "t",
    "terrorists": "t",
    "ct": "ct",
    "counterterrorist": "ct",
    "counterterrorists": "ct",
    "counter-terrorist": "ct",
    "counter-terrorists": "ct",
}

RIFLES = {"ak47", "m4a1", "m4a1_silencer", "aug", "sg556", "galilar", "famas"}
AWP = {"awp"}
PISTOLS = {
    "glock", "usp_silencer", "hkp2000", "p250", "deagle",
    "elite", "tec9", "fiveseven", "cz75a", "revolver"
}

SMGS = {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"}
SHOTGUNS = {"nova", "xm1014", "mag7", "sawedoff"}
SNIPERS = {"ssg08", "scar20", "g3sg1"}
UTIL_DAMAGE_WEAPONS = {"hegrenade", "inferno", "molotov", "incgrenade"}
GRENADE_WEAPONS = {
    "weapon_hegrenade": "he_grenades",
    "weapon_flashbang": "flashbangs",
    "weapon_smokegrenade": "smokes",
    "weapon_molotov": "fire_nades",
    "weapon_incgrenade": "fire_nades",
    "weapon_decoy": "decoys",
}

SIDE_KEYS = ["player_name", "side"]

RATE_COLUMNS_TO_DROP = {
    "kills",
    "deaths",
    "assists",
    "flash_assists",
    "opening_kills",
    "opening_deaths",
    "trade_kills",
    "traded_deaths",
    "rifle_kills",
    "awp_kills",
    "multi_kill_rounds",
    "damage",
    "damage_taken",
    "util_damage",
    "grenades_thrown",
    "he_grenades_thrown",
    "flashbangs_thrown",
    "smokes_thrown",
    "fire_nades_thrown",
    "decoys_thrown",
    "awpy_damage",
    "awpy_rounds",
    "kast_rounds",
}

FINAL_COLUMNS_TO_DROP = {
    "kast",
    "impact",
    "rating",
    "kdr",
    "dpr",
}

POSITION_FEATURES = [
    "avg_distance_to_enemy",
    "avg_distance_to_team_centroid",
    "relative_team_centroid_distance",
    "avg_distance_moved_per_round",
    "avg_distance_to_closest_teammate",
    "time_near_enemy_rate",
    "first_contact_rate",
    "time_stationary_rate",
]


def pick_col(df: pl.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of {list(candidates)} exist. Available: {df.columns}")
    return None


def empty(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def norm_side(col_name: str) -> pl.Expr:
    return pl.col(col_name).cast(pl.Utf8).str.to_lowercase().replace_strict(TEAM_ALIASES, default=None)


def norm_name(expr: pl.Expr) -> pl.Expr:
    """Strip leading spaces and normalise NAF-FLY → NAF."""
    return (
        expr.cast(pl.Utf8)
        .str.replace(r"^ +", "")
        .str.replace(r"^NAF-FLY$", "NAF")
    )


def rate(num: str, den: str, out: str) -> pl.Expr:
    return pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(0.0).alias(out)


def safe_table(demo: Demo, names: Sequence[str]) -> pl.DataFrame:
    for name in names:
        value = getattr(demo, name, None)
        if value is None:
            continue
        if isinstance(value, pl.DataFrame):
            return value
        try:
            return pl.DataFrame(value)
        except Exception:
            pass
    return pl.DataFrame()


def resolve_input_path(input_path: str | None, test_mode: bool) -> Path:
    if test_mode:
        path = Path("demos_test")
        if not path.exists():
            raise FileNotFoundError(
                "Test mode expects a folder named 'demos_test' in the working directory."
            )
        return path

    if input_path is None:
        raise ValueError("Provide an input demo path unless using --test.")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    return path


def iter_demo_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".dem":
            raise ValueError(f"Input file is not a .dem file: {path}")
        return [path]
    return sorted(path.rglob("*.dem"))


def select_player_side(
    df: pl.DataFrame,
    name_candidates: Sequence[str],
    side_candidates: Sequence[str],
) -> pl.DataFrame:
    name_col = pick_col(df, name_candidates, required=False)
    side_col = pick_col(df, side_candidates, required=False)
    if name_col is None or side_col is None:
        return empty({"player_name": pl.Utf8, "side": pl.Utf8})

    return (
        df.select(
            norm_name(pl.col(name_col)).alias("player_name"),
            norm_side(side_col).alias("side"),
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
    )


def dist3_expr(ax: str, ay: str, az: str, bx: str, by: str, bz: str) -> pl.Expr:
    return (
        ((pl.col(ax) - pl.col(bx)) ** 2)
        + ((pl.col(ay) - pl.col(by)) ** 2)
        + ((pl.col(az) - pl.col(bz)) ** 2)
    ).sqrt()


def build_position_stats(ticks: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "player_name": pl.Utf8,
        "side": pl.Utf8,
        "avg_distance_to_enemy": pl.Float64,
        "avg_distance_to_team_centroid": pl.Float64,
        "relative_team_centroid_distance": pl.Float64,
        "avg_distance_moved_per_round": pl.Float64,
        "avg_distance_to_closest_teammate": pl.Float64,
        "time_near_enemy_rate": pl.Float64,
        "first_contact_rate": pl.Float64,
        "time_stationary_rate": pl.Float64,
        "opening_displacement_20s": pl.Float64,
    }

    if ticks.is_empty():
        return empty(schema)

    player = pick_col(ticks, ["player_name", "name", "player"], required=False)
    side = pick_col(ticks, ["side", "player_side", "team_name"], required=False)
    tick = pick_col(ticks, ["tick", "game_tick"], required=False)
    round_col = pick_col(ticks, ["round_num", "round_number", "round", "round_index"], required=False)
    x = pick_col(ticks, ["x", "X"], required=False)
    y = pick_col(ticks, ["y", "Y"], required=False)
    z = pick_col(ticks, ["z", "Z"], required=False)

    if not all([player, side, tick, round_col, x, y, z]):
        return empty(schema)

    pos = (
        ticks.select(
            norm_name(pl.col(player)).alias("player_name"),
            norm_side(side).alias("side"),
            pl.col(round_col).cast(pl.Int64).alias("round_id"),
            pl.col(tick).cast(pl.Int64).alias("tick"),
            pl.col(x).cast(pl.Float64).alias("x"),
            pl.col(y).cast(pl.Float64).alias("y"),
            pl.col(z).cast(pl.Float64).alias("z"),
        )
        .filter(
            pl.col("player_name").is_not_null()
            & pl.col("side").is_in(["ct", "t"])
            & pl.col("x").is_not_null()
            & pl.col("y").is_not_null()
            & pl.col("z").is_not_null()
        )
    )

    if pos.is_empty():
        return empty(schema)

    enemies = pos.rename({
        "player_name": "enemy_name",
        "side": "enemy_side",
        "x": "enemy_x",
        "y": "enemy_y",
        "z": "enemy_z",
    })

    enemy_tick = (
        pos.join(enemies, on=["round_id", "tick"], how="inner")
        .filter(pl.col("side") != pl.col("enemy_side"))
        .with_columns(
            dist3_expr("x", "y", "z", "enemy_x", "enemy_y", "enemy_z").alias("enemy_dist")
        )
        .group_by(["player_name", "side", "round_id", "tick"])
        .agg(
            pl.mean("enemy_dist").alias("avg_tick_enemy_dist"),
            pl.min("enemy_dist").alias("nearest_enemy_dist"),
        )
    )

    enemy_features = (
        enemy_tick.with_columns(
            (pl.col("nearest_enemy_dist") <= NEAR_ENEMY_RADIUS)
            .cast(pl.Float64)
            .alias("near_enemy")
        )
        .group_by(SIDE_KEYS)
        .agg(
            pl.mean("avg_tick_enemy_dist").alias("avg_distance_to_enemy"),
            pl.mean("near_enemy").alias("time_near_enemy_rate"),
        )
    )

    first_contact = (
        enemy_tick.with_columns(
            pl.min("nearest_enemy_dist")
            .over(["round_id", "tick", "side"])
            .alias("team_min_enemy_dist")
        )
        .with_columns(
            (pl.col("nearest_enemy_dist") == pl.col("team_min_enemy_dist"))
            .cast(pl.Float64)
            .alias("is_first_contact")
        )
        .group_by(SIDE_KEYS)
        .agg(pl.mean("is_first_contact").alias("first_contact_rate"))
    )

    team_centroid = (
        pos.group_by(["round_id", "tick", "side"])
        .agg(
            pl.mean("x").alias("team_x"),
            pl.mean("y").alias("team_y"),
            pl.mean("z").alias("team_z"),
        )
    )

    centroid_tick = (
        pos.join(team_centroid, on=["round_id", "tick", "side"], how="left")
        .with_columns(
            dist3_expr("x", "y", "z", "team_x", "team_y", "team_z").alias("centroid_dist")
        )
    )

    team_avg_centroid = (
        centroid_tick.group_by(["round_id", "tick", "side"])
        .agg(pl.mean("centroid_dist").alias("team_avg_centroid_dist"))
    )

    centroid_features = (
        centroid_tick.join(team_avg_centroid, on=["round_id", "tick", "side"], how="left")
        .with_columns(
            pl.when(pl.col("team_avg_centroid_dist") > 0)
            .then(pl.col("centroid_dist") / pl.col("team_avg_centroid_dist"))
            .otherwise(0.0)
            .alias("relative_centroid_dist")
        )
        .group_by(SIDE_KEYS)
        .agg(
            pl.mean("centroid_dist").alias("avg_distance_to_team_centroid"),
            pl.mean("relative_centroid_dist").alias("relative_team_centroid_distance"),
        )
    )

    teammates = pos.rename({
        "player_name": "teammate_name",
        "x": "teammate_x",
        "y": "teammate_y",
        "z": "teammate_z",
    })

    teammate_features = (
        pos.join(teammates, on=["round_id", "tick", "side"], how="inner")
        .filter(pl.col("player_name") != pl.col("teammate_name"))
        .with_columns(
            dist3_expr("x", "y", "z", "teammate_x", "teammate_y", "teammate_z")
            .alias("teammate_dist")
        )
        .group_by(["player_name", "side", "round_id", "tick"])
        .agg(pl.min("teammate_dist").alias("closest_teammate_dist"))
        .group_by(SIDE_KEYS)
        .agg(pl.mean("closest_teammate_dist").alias("avg_distance_to_closest_teammate"))
    )

    movement_tick = (
        pos.sort(["player_name", "side", "round_id", "tick"])
        .with_columns(
            pl.col("x").shift(1).over(["player_name", "side", "round_id"]).alias("prev_x"),
            pl.col("y").shift(1).over(["player_name", "side", "round_id"]).alias("prev_y"),
            pl.col("z").shift(1).over(["player_name", "side", "round_id"]).alias("prev_z"),
        )
        .with_columns(
            pl.when(pl.col("prev_x").is_not_null())
            .then(dist3_expr("x", "y", "z", "prev_x", "prev_y", "prev_z"))
            .otherwise(0.0)
            .alias("step_dist")
        )
    )

    movement_features = (
        movement_tick.group_by(SIDE_KEYS + ["round_id"])
        .agg(
            pl.sum("step_dist").alias("round_distance_moved"),
            ((pl.col("step_dist") <= STATIONARY_STEP_DISTANCE).cast(pl.Float64))
            .mean()
            .alias("round_stationary_rate"),
        )
        .group_by(SIDE_KEYS)
        .agg(
            pl.mean("round_distance_moved").alias("avg_distance_moved_per_round"),
            pl.mean("round_stationary_rate").alias("time_stationary_rate"),
        )
    )


    out = pos.select(SIDE_KEYS).unique()

    for features in [
        enemy_features,
        first_contact,
        centroid_features,
        teammate_features,
        movement_features,
    ]:
        out = out.join(features, on=SIDE_KEYS, how="left")

    return out.fill_null(0).fill_nan(0)


def build_builtin_stats(demo: Demo) -> pl.DataFrame:
    pieces: list[pl.DataFrame] = []

    adr_df = adr(demo)
    if not adr_df.is_empty():
        pieces.append(
            adr_df.select(
                norm_name(pl.col("name")).alias("player_name"),
                norm_side("side").alias("side"),
                pl.col("n_rounds").cast(pl.Int64).alias("rounds_played"),
                pl.col("dmg").cast(pl.Float64).alias("awpy_damage"),
                pl.col("adr").cast(pl.Float64).alias("adr"),
            )
        )

    kast_df = kast(demo, trade_length_in_seconds=5.0)
    if not kast_df.is_empty():
        pieces.append(
            kast_df.select(
                norm_name(pl.col("name")).alias("player_name"),
                norm_side("side").alias("side"),
                pl.col("kast").cast(pl.Float64).alias("kast"),
            )
        )

    rating_df = rating(demo)
    if not rating_df.is_empty():
        pieces.append(
            rating_df.select(
                norm_name(pl.col("name")).alias("player_name"),
                norm_side("side").alias("side"),
                pl.col("impact").cast(pl.Float64).alias("impact"),
                pl.col("rating").cast(pl.Float64).alias("rating"),
            )
        )

    if not pieces:
        return empty(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "rounds_played": pl.Int64,
                "awpy_damage": pl.Float64,
                "adr": pl.Float64,
                "kast": pl.Float64,
                "impact": pl.Float64,
                "rating": pl.Float64,
            }
        )

    return (
        pl.concat(pieces, how="diagonal_relaxed")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(SIDE_KEYS)
        .agg(
            pl.max("rounds_played").alias("rounds_played"),
            pl.max("awpy_damage").alias("awpy_damage"),
            pl.max("adr").alias("adr"),
            pl.max("kast").alias("kast"),
            pl.max("impact").alias("impact"),
            pl.max("rating").alias("rating"),
        )
    )


def build_base(kills: pl.DataFrame, damages: pl.DataFrame, builtin: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    pieces = []

    if not builtin.is_empty():
        pieces.append(builtin.select(SIDE_KEYS))

    if not positions.is_empty():
        pieces.append(positions.select(SIDE_KEYS))

    if not kills.is_empty():
        pieces.extend(
            [
                select_player_side(kills, ["attacker_name", "killer_name"], ["attacker_side", "killer_side"]),
                select_player_side(kills, ["victim_name", "player_name"], ["victim_side", "player_side"]),
                select_player_side(kills, ["assister_name", "assister"], ["assister_side"]),
            ]
        )

    if not damages.is_empty():
        pieces.extend(
            [
                select_player_side(damages, ["attacker_name", "attacker"], ["attacker_side"]),
                select_player_side(damages, ["victim_name", "victim"], ["victim_side"]),
            ]
        )

    pieces = [p for p in pieces if not p.is_empty()]
    if not pieces:
        return empty({"player_name": pl.Utf8, "side": pl.Utf8})

    return pl.concat(pieces, how="vertical").unique()


def build_kill_stats(kills: pl.DataFrame) -> pl.DataFrame:
    if kills.is_empty():
        return empty(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "kills": pl.Int64,
                "deaths": pl.Int64,
                "assists": pl.Int64,
                "flash_assists": pl.Int64,
                "rifle_kills": pl.Int64,
                "awp_kills": pl.Int64,
            }
        )

    attacker = pick_col(kills, ["attacker_name", "killer_name"])
    attacker_side = pick_col(kills, ["attacker_side", "killer_side"])
    victim = pick_col(kills, ["victim_name", "player_name"])
    victim_side = pick_col(kills, ["victim_side", "player_side"])
    assister = pick_col(kills, ["assister_name", "assister"], required=False)
    assister_side = pick_col(kills, ["assister_side"], required=False)
    assisted_flash = pick_col(kills, ["assistedflash"], required=False)
    weapon = pick_col(kills, ["weapon", "weapon_name", "weapon_class"], required=False)

    pieces = [
        kills.select(
            norm_name(pl.col(attacker)).alias("player_name"),
            norm_side(attacker_side).alias("side"),
            pl.lit(1, dtype=pl.Int64).alias("kills"),
            pl.lit(0, dtype=pl.Int64).alias("deaths"),
            pl.lit(0, dtype=pl.Int64).alias("assists"),
            pl.lit(0, dtype=pl.Int64).alias("flash_assists"),
            (
                pl.col(weapon).cast(pl.Utf8).str.to_lowercase().is_in(list(RIFLES)).cast(pl.Int64)
                if weapon
                else pl.lit(0)
            ).alias("rifle_kills"),
            (
                pl.col(weapon).cast(pl.Utf8).str.to_lowercase().is_in(list(AWP)).cast(pl.Int64)
                if weapon
                else pl.lit(0)
            ).alias("awp_kills"),
        ),
        kills.select(
            norm_name(pl.col(victim)).alias("player_name"),
            norm_side(victim_side).alias("side"),
            pl.lit(0, dtype=pl.Int64).alias("kills"),
            pl.lit(1, dtype=pl.Int64).alias("deaths"),
            pl.lit(0, dtype=pl.Int64).alias("assists"),
            pl.lit(0, dtype=pl.Int64).alias("flash_assists"),
            pl.lit(0, dtype=pl.Int64).alias("rifle_kills"),
            pl.lit(0, dtype=pl.Int64).alias("awp_kills"),
        ),
    ]

    if assister and assister_side:
        pieces.append(
            kills.select(
                norm_name(pl.col(assister)).alias("player_name"),
                norm_side(assister_side).alias("side"),
                pl.lit(0, dtype=pl.Int64).alias("kills"),
                pl.lit(0, dtype=pl.Int64).alias("deaths"),
                pl.lit(1, dtype=pl.Int64).alias("assists"),
                (
                    pl.col(assisted_flash).fill_null(False).cast(pl.Boolean).cast(pl.Int64)
                    if assisted_flash
                    else pl.lit(0)
                ).alias("flash_assists"),
                pl.lit(0, dtype=pl.Int64).alias("rifle_kills"),
                pl.lit(0, dtype=pl.Int64).alias("awp_kills"),
            )
        )

    return (
        pl.concat(pieces, how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(SIDE_KEYS)
        .sum()
    )


def build_opening_and_multi_stats(kills: pl.DataFrame) -> pl.DataFrame:
    if kills.is_empty():
        return empty(
            {
                "player_name": pl.Utf8,
                "side": pl.Utf8,
                "opening_kills": pl.Int64,
                "opening_deaths": pl.Int64,
                "multi_kill_rounds": pl.Int64,
            }
        )

    round_col = pick_col(kills, ["round_num", "round_number", "round", "round_index"])
    tick_col = pick_col(kills, ["tick", "game_tick", "event_tick"], required=False)
    attacker = pick_col(kills, ["attacker_name", "killer_name"])
    attacker_side = pick_col(kills, ["attacker_side", "killer_side"])
    victim = pick_col(kills, ["victim_name", "player_name"])
    victim_side = pick_col(kills, ["victim_side", "player_side"])

    first_kills = (
        kills.sort([round_col, tick_col]).unique(subset=[round_col], keep="first")
        if tick_col
        else kills.unique(subset=[round_col], keep="first")
    )

    openings = pl.concat(
        [
            first_kills.select(
                norm_name(pl.col(attacker)).alias("player_name"),
                norm_side(attacker_side).alias("side"),
                pl.lit(1).alias("opening_kills"),
                pl.lit(0).alias("opening_deaths"),
            ),
            first_kills.select(
                norm_name(pl.col(victim)).alias("player_name"),
                norm_side(victim_side).alias("side"),
                pl.lit(0).alias("opening_kills"),
                pl.lit(1).alias("opening_deaths"),
            ),
        ],
        how="vertical",
    ).group_by(SIDE_KEYS).sum()

    multi = (
        kills.select(
            norm_name(pl.col(attacker)).alias("player_name"),
            norm_side(attacker_side).alias("side"),
            pl.col(round_col).alias("round_id"),
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(SIDE_KEYS + ["round_id"])
        .agg(pl.len().alias("kills_in_round"))
        .group_by(SIDE_KEYS)
        .agg((pl.col("kills_in_round") >= 2).sum().alias("multi_kill_rounds"))
    )

    return openings.join(multi, on=SIDE_KEYS, how="full", coalesce=True).fill_null(0)


def build_trade_stats(kills: pl.DataFrame) -> pl.DataFrame:
    schema = {"player_name": pl.Utf8, "side": pl.Utf8, "trade_kills": pl.Int64, "traded_deaths": pl.Int64}
    required = {"round_num", "tick", "attacker_name", "attacker_side", "victim_name", "victim_side", "was_traded"}
    if kills.is_empty() or not required <= set(kills.columns):
        return empty(schema)

    df = (
        kills.select(
            pl.col("round_num").cast(pl.Int64),
            pl.col("tick").cast(pl.Int64),
            pl.col("attacker_name").cast(pl.Utf8),
            norm_side("attacker_side").alias("attacker_side"),
            pl.col("victim_name").cast(pl.Utf8),
            norm_side("victim_side").alias("victim_side"),
            pl.col("was_traded").fill_null(False).cast(pl.Boolean),
        )
        .with_row_index("kill_id")
        .filter(
            pl.col("attacker_name").is_not_null()
            & pl.col("victim_name").is_not_null()
            & pl.col("attacker_side").is_in(["ct", "t"])
            & pl.col("victim_side").is_in(["ct", "t"])
        )
    )

    traded = df.filter(pl.col("was_traded"))

    traded_deaths = traded.select(
        pl.col("victim_name").alias("player_name"),
        pl.col("victim_side").alias("side"),
        pl.lit(0).alias("trade_kills"),
        pl.lit(1).alias("traded_deaths"),
    )

    trade_kills = (
        traded.select(
            pl.col("kill_id").alias("orig_id"),
            pl.col("round_num"),
            pl.col("tick").alias("orig_tick"),
            pl.col("attacker_name").alias("orig_attacker"),
            pl.col("attacker_side").alias("orig_attacker_side"),
            pl.col("victim_side").alias("orig_victim_side"),
        )
        .join(
            df.select(
                pl.col("kill_id"),
                pl.col("round_num"),
                pl.col("tick").alias("trade_tick"),
                pl.col("attacker_name").alias("player_name"),
                pl.col("attacker_side").alias("side"),
                pl.col("victim_name").alias("trade_victim"),
                pl.col("victim_side").alias("trade_victim_side"),
            ),
            on="round_num",
        )
        .filter(pl.col("trade_tick") > pl.col("orig_tick"))
        .filter(pl.col("trade_victim") == pl.col("orig_attacker"))
        .filter(pl.col("side") == pl.col("orig_victim_side"))
        .filter(pl.col("trade_victim_side") == pl.col("orig_attacker_side"))
        .sort(["orig_id", "trade_tick", "kill_id"])
        .unique(subset=["orig_id"], keep="first")
        .select(
            "player_name",
            "side",
            pl.lit(1).alias("trade_kills"),
            pl.lit(0).alias("traded_deaths"),
        )
    )

    pieces = [p for p in [traded_deaths, trade_kills] if not p.is_empty()]
    if not pieces:
        return empty(schema)

    return pl.concat(pieces, how="vertical").group_by(SIDE_KEYS).sum()


def build_damage_stats(damages: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "player_name": pl.Utf8,
        "side": pl.Utf8,
        "damage": pl.Float64,
        "damage_taken": pl.Float64,
        "util_damage": pl.Float64,
    }
    if damages.is_empty():
        return empty(schema)

    attacker = pick_col(damages, ["attacker_name", "attacker"], required=False)
    attacker_side = pick_col(damages, ["attacker_side"], required=False)
    victim = pick_col(damages, ["victim_name", "victim"], required=False)
    victim_side = pick_col(damages, ["victim_side"], required=False)
    dmg = pick_col(damages, ["dmg_health", "hp_damage", "health_damage", "damage", "damage_health"], required=False)
    weapon = pick_col(damages, ["weapon", "weapon_name"], required=False)

    pieces = []

    if attacker and attacker_side and dmg:
        pieces.append(
            damages.select(
                norm_name(pl.col(attacker)).alias("player_name"),
                norm_side(attacker_side).alias("side"),
                pl.col(dmg).cast(pl.Float64).fill_null(0.0).alias("damage"),
                pl.lit(0.0).alias("damage_taken"),
                (
                    pl.when(pl.col(weapon).cast(pl.Utf8).str.to_lowercase().is_in(list(UTIL_DAMAGE_WEAPONS)))
                    .then(pl.col(dmg).cast(pl.Float64).fill_null(0.0))
                    .otherwise(0.0)
                    if weapon
                    else pl.lit(0.0)
                ).alias("util_damage"),
            )
        )

    if victim and victim_side and dmg:
        pieces.append(
            damages.select(
                norm_name(pl.col(victim)).alias("player_name"),
                norm_side(victim_side).alias("side"),
                pl.lit(0.0).alias("damage"),
                pl.col(dmg).cast(pl.Float64).fill_null(0.0).alias("damage_taken"),
                pl.lit(0.0).alias("util_damage"),
            )
        )

    pieces = [p for p in pieces if not p.is_empty()]
    if not pieces:
        return empty(schema)

    return (
        pl.concat(pieces, how="vertical")
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .group_by(SIDE_KEYS)
        .sum()
    )


def build_grenade_stats(shots: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "player_name": pl.Utf8,
        "side": pl.Utf8,
        "grenades_thrown": pl.Int64,
        "he_grenades_thrown": pl.Int64,
        "flashbangs_thrown": pl.Int64,
        "smokes_thrown": pl.Int64,
        "fire_nades_thrown": pl.Int64,
        "decoys_thrown": pl.Int64,
    }
    if shots.is_empty():
        return empty(schema)

    player = pick_col(shots, ["player_name", "player", "name"], required=False)
    side = pick_col(shots, ["player_side", "side"], required=False)
    weapon = pick_col(shots, ["weapon"], required=False)
    if not player or not side or not weapon:
        return empty(schema)

    df = (
        shots.select(
            norm_name(pl.col(player)).alias("player_name"),
            norm_side(side).alias("side"),
            pl.col(weapon).cast(pl.Utf8).str.to_lowercase().alias("weapon"),
        )
        .filter(pl.col("player_name").is_not_null() & pl.col("side").is_in(["ct", "t"]))
        .filter(pl.col("weapon").is_in(list(GRENADE_WEAPONS)))
    )

    return df.group_by(SIDE_KEYS).agg(
        pl.len().alias("grenades_thrown"),
        (pl.col("weapon") == "weapon_hegrenade").sum().alias("he_grenades_thrown"),
        (pl.col("weapon") == "weapon_flashbang").sum().alias("flashbangs_thrown"),
        (pl.col("weapon") == "weapon_smokegrenade").sum().alias("smokes_thrown"),
        pl.col("weapon").is_in(["weapon_molotov", "weapon_incgrenade"]).sum().alias("fire_nades_thrown"),
        (pl.col("weapon") == "weapon_decoy").sum().alias("decoys_thrown"),
    )


def add_rates(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.col(c).fill_null(0) for c in df.columns if c not in SIDE_KEYS)
    cols = set(df.columns)
    exprs: list[pl.Expr] = []

    for raw, out in [
        ("kills", "kpr"),
        ("deaths", "dpr"),
        ("assists", "assists_per_round"),
        ("flash_assists", "flash_assists_per_round"),
        ("multi_kill_rounds", "multi_kill_rate"),
        ("damage", "damage_per_round"),
        ("damage_taken", "damage_taken_per_round"),
        ("util_damage", "util_damage_per_round"),
        ("grenades_thrown", "grenades_per_round"),
        ("he_grenades_thrown", "he_grenades_per_round"),
        ("flashbangs_thrown", "flashbangs_per_round"),
        ("smokes_thrown", "smokes_per_round"),
        ("fire_nades_thrown", "fire_nades_per_round"),
        ("decoys_thrown", "decoys_per_round"),
    ]:
        if {raw, "rounds_played"} <= cols:
            exprs.append(rate(raw, "rounds_played", out))

    if {"kills", "deaths"} <= cols:
        exprs.append(rate("kills", "deaths", "kdr"))

    if {"rounds_played", "deaths"} <= cols:
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("rounds_played") - pl.col("deaths")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .clip(lower_bound=0.0)
            .alias("survival_rate")
        )

    if {"opening_kills", "kills"} <= cols:
        exprs.append(rate("opening_kills", "kills", "opening_kill_rate"))
    if {"opening_deaths", "deaths"} <= cols:
        exprs.append(rate("opening_deaths", "deaths", "opening_death_rate"))
    if {"opening_kills", "opening_deaths"} <= cols:
        attempts = pl.col("opening_kills") + pl.col("opening_deaths")
        exprs.append(
            pl.when(attempts > 0)
            .then(pl.col("opening_kills") / attempts)
            .otherwise(0.0)
            .alias("opening_duel_success")
        )

    if {"trade_kills", "kills"} <= cols:
        exprs.append(rate("trade_kills", "kills", "trade_kill_rate"))
    if {"traded_deaths", "deaths"} <= cols:
        exprs.append(rate("traded_deaths", "deaths", "death_traded_rate"))
    if {"trade_kills", "traded_deaths", "rounds_played"} <= cols:
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("trade_kills") + pl.col("traded_deaths")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .alias("trade_participation")
        )

    if {"awp_kills", "kills"} <= cols:
        exprs.append(rate("awp_kills", "kills", "awp_kill_share"))
    if {"rifle_kills", "kills"} <= cols:
        exprs.append(rate("rifle_kills", "kills", "rifle_kill_share"))
    if {"damage", "damage_taken", "rounds_played"} <= cols:
        exprs.append(
            pl.when(pl.col("rounds_played") > 0)
            .then((pl.col("damage") - pl.col("damage_taken")) / pl.col("rounds_played"))
            .otherwise(0.0)
            .alias("damage_diff_per_round")
        )

    if exprs:
        df = df.with_columns(exprs)

    return df.fill_null(0).fill_nan(0)


def keep_rate_features_only(df: pl.DataFrame) -> pl.DataFrame:
    drop_cols = [c for c in RATE_COLUMNS_TO_DROP if c in df.columns]
    return df.drop(drop_cols).fill_null(0).fill_nan(0)


def drop_unwanted_final_columns(df: pl.DataFrame) -> pl.DataFrame:
    drop_cols = [c for c in FINAL_COLUMNS_TO_DROP if c in df.columns]
    return df.drop(drop_cols).fill_null(0).fill_nan(0)


def parse_single_demo(demo_path: Path) -> pl.DataFrame:
    demo = Demo(str(demo_path))
    demo.parse()

    kills = calculate_trades(demo)
    shots = safe_table(demo, ["shots", "shots_df"])
    damages = safe_table(demo, ["damages", "damages_df"])
    ticks = safe_table(demo, ["ticks", "ticks_df", "player_ticks", "player_ticks_df"])

    builtin = build_builtin_stats(demo)
    positions = build_position_stats(ticks)

    if (
        kills.is_empty()
        and shots.is_empty()
        and damages.is_empty()
        and builtin.is_empty()
        and positions.is_empty()
    ):
        raise RuntimeError("No usable event tables found in demo.")

    out = build_base(kills, damages, builtin, positions)

    for features in [
        builtin,
        build_kill_stats(kills),
        build_trade_stats(kills),
        build_opening_and_multi_stats(kills),
        build_damage_stats(damages),
        build_grenade_stats(shots),
        positions,
    ]:
        out = out.join(features, on=SIDE_KEYS, how="left")

    return add_rates(out)

def combine_demo_results(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    frames = list(frames)
    if not frames:
        raise RuntimeError("No demos parsed successfully.")

    # Drop players that appeared in the demo but never played a round
    # (spectators, coaches, late-joiners, etc.)
    frames = [f.filter(pl.col("rounds_played") > 0) if "rounds_played" in f.columns else f for f in frames]
    frames = [f for f in frames if not f.is_empty()]

    if not frames:
        raise RuntimeError("No demos parsed successfully.")

    # Compute rates on each per-demo frame first so std is taken over
    # per-demo rates (e.g. kpr per demo), not raw counts.
    frames = [add_rates(f) for f in frames]

    df = pl.concat(frames, how="diagonal_relaxed")
    work = df

    weighted_cols = ["adr", "kast", "impact", "rating"] + POSITION_FEATURES
    for col in weighted_cols:
        if {col, "rounds_played"} <= set(work.columns):
            work = work.with_columns(
                (pl.col(col).fill_null(0.0) * pl.col("rounds_played").fill_null(0)).alias(f"_{col}_weighted")
            )

    sum_cols = [
        c
        for c in work.columns
        if c not in SIDE_KEYS
        and c not in {"adr", "kast", "impact", "rating", *POSITION_FEATURES}
        and not c.startswith("_")
    ]

    # Columns to compute std over: weighted/rate cols (most informative for
    # consistency) and position features. Excludes raw counts and rounds_played.
    std_cols = [c for c in weighted_cols if c in work.columns]

    agg_exprs = [pl.sum(c).alias(c) for c in sum_cols]
    agg_exprs.extend(
        pl.sum(f"_{c}_weighted").alias(f"_{c}_weighted")
        for c in weighted_cols
        if f"_{c}_weighted" in work.columns
    )
    # Std aggregations — _std suffix marks them clearly as variance features.
    agg_exprs.extend(
        pl.std(c).fill_null(0.0).alias(f"{c}_std")
        for c in std_cols
    )

    combined = work.group_by(SIDE_KEYS).agg(agg_exprs)

    exprs: list[pl.Expr] = []

    for col in weighted_cols:
        weighted = f"_{col}_weighted"
        if {weighted, "rounds_played"} <= set(combined.columns):
            exprs.append(rate(weighted, "rounds_played", col))

    if exprs:
        combined = combined.with_columns(exprs)

    combined = combined.drop([c for c in combined.columns if c.startswith("_")])
    return keep_rate_features_only(add_rates(combined))


def sort_output_columns(df: pl.DataFrame) -> pl.DataFrame:
    metric_order = [
        "adr",
        "kpr",
        "survival_rate",
        "damage_per_round",
        "damage_taken_per_round",
        "damage_diff_per_round",
        "util_damage_per_round",
        "assists_per_round",
        "flash_assists_per_round",
        "trade_kill_rate",
        "death_traded_rate",
        "trade_participation",
        "opening_kill_rate",
        "opening_death_rate",
        "opening_duel_success",
        "multi_kill_rate",
        "rifle_kill_share",
        "awp_kill_share",
        "grenades_per_round",
        "he_grenades_per_round",
        "flashbangs_per_round",
        "smokes_per_round",
        "fire_nades_per_round",
        "decoys_per_round",
        "avg_distance_to_enemy",
        "avg_distance_to_team_centroid",
        "relative_team_centroid_distance",
        "avg_distance_moved_per_round",
        "avg_distance_to_closest_teammate",
        "time_near_enemy_rate",
        "first_contact_rate",
        "time_stationary_rate",
    ]

    # Interleave _std columns right after their base column so the CSV reads
    # naturally: adr, adr_std, kpr, kpr_std, ...
    ordered: list[str] = ["player_name", "side", "rounds_played"]
    for col in metric_order:
        if col in df.columns:
            ordered.append(col)
        std_col = f"{col}_std"
        if std_col in df.columns:
            ordered.append(std_col)

    existing = [c for c in ordered if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df.select(existing + remaining).sort(["player_name", "side"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", nargs="?", default=None)
    parser.add_argument("output_csv", nargs="?", default=None)
    parser.add_argument("--test", dest="test_output_csv", default=None)
    args = parser.parse_args()

    test_mode = args.test_output_csv is not None
    output_csv = args.test_output_csv if test_mode else args.output_csv

    if output_csv is None:
        raise ValueError("You must provide an output CSV path, or use --test <output_csv>.")

    input_path = resolve_input_path(args.input_path, test_mode)
    demo_files = iter_demo_files(input_path)

    print(f"[info] input path: {input_path.resolve()}")
    print(f"[info] found {len(demo_files)} demos to process")

    frames: list[pl.DataFrame] = []
    success_count = 0
    total = len(demo_files)

    print(f"[info] spawning {WORKERS} worker process(es)")

    # Each demo is parsed in its own process — fully independent, no shared
    # state — so ProcessPoolExecutor sidesteps the GIL and gets real
    # parallelism for the CPU/IO-bound awpy parsing work.
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        future_to_path = {
            pool.submit(parse_single_demo, demo_path): demo_path
            for demo_path in demo_files
        }

        for future in as_completed(future_to_path):
            demo_path = future_to_path[future]
            print(f"[{success_count + 1}/{total}] finished: {demo_path.name}")
            try:
                frames.append(future.result())
                success_count += 1
                print(f"[ok] {demo_path.name}")
            except Exception as exc:
                print(f"[skip] {demo_path.name}: {exc}")

    if not frames:
        raise RuntimeError("No demos parsed successfully.")

    final_df = combine_demo_results(frames)
    final_df = sort_output_columns(final_df)
    final_df = drop_unwanted_final_columns(final_df)
    final_df = final_df.fill_null(0).fill_nan(0)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(output_path)

    print(f"[done] parsed {success_count}/{len(demo_files)} demos successfully")
    print(f"[done] output written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()