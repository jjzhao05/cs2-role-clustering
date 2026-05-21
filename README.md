# CS2 Role Classifier

## Motivation

I have followed professional Counter-Strike for years and have always been interested in how teams structure themselves around player roles. Analysts and fans often describe players as entries, lurkers, anchors, support players, or many other roles, but many of these labels were informal and difficult to define quantitatively. This project started as a way to explore whether those role archetypes could emerge naturally from gameplay data rather than being manually assigned. It also gave me an opportunity to work with demo parsing, feature engineering, clustering methods, and large-scale match data from an esport I genuinely enjoy watching and following.

## Background

Counter-Strike 2 is a competitive, team-based first-person shooter where two teams of five, Terrorists (T) and Counter-Terrorists (CT), compete in round-based matches. The T side aims to plant a bomb at a designated site and defend it until detonation; the CT side aims to prevent the plant or defuse the bomb. At the highest level, the game involves tactical positioning, utility usage (grenades, flashbangs, smokes, molotovs), economic decision-making, and mechanical skill.

Although all players share the same mechanics, professional teams develop distinct roles. Some are easily defined:

- **AWPer**: uses the AWP sniper rifle to hold angles, punish peeks, and control space
- **IGL** (In-Game Leader): makes tactical calls and directs the team mid-round

Others are far less well-defined: entry fraggers, lurkers, anchors, support riflers. The goal of this project is not to force players into rigid labels, but to test whether these archetypes emerge naturally from the data.

## Pipeline

```
demos/
  └─ *.dem
       │
       ▼
 file_unzipper.py     Extract .rar archives from Downloads, deduplicate .dem files
       │
       ▼
 demo_parser.py       Parse demos → per-player, per-side feature CSV
       │
       ▼
 cluster_players.py   Run KMeans, GMM, HDBSCAN
       │
       ▼
 plotter.py           PCA scatter plots, radar charts, silhouette vs k chart
```

Run the full pipeline end-to-end:
```bash
python main.py
```

Or run individual stages:
```bash
python file_unzipper.py
python demo_parser.py <demos_dir> output.csv
python cluster_players.py
python plotter.py
```

## Methods

Three clustering algorithms are evaluated per side (CT and T separately):

| Method | Notes |
|---|---|
| **KMeans** | k ∈ [2, 8], n_init=20 |
| **GMM** | k ∈ [2, 8], full covariance |
| **HDBSCAN** | grid search over min_cluster_size × min_samples; noise points excluded from scoring |

Models are ranked by silhouette score then Davies-Bouldin index. The top 3 models per method per side are plotted.

Feature importance is printed after clustering using a `RandomForestClassifier` trained to predict cluster labels.

PCA is computed once per side and reused across all models for consistent 2D visualization.

## Data Schema

Output is in **long format**: one row per player per side. All rate features are normalized by rounds played.

### Identifiers

| Column | Description |
|---|---|
| `player_name` | Player identifier |
| `side` | `ct` or `t` |
| `rounds_played` | Rounds included for this player-side |

### Combat

| Column | Description |
|---|---|
| `adr` | Average damage per round |
| `kpr` | Kills per round |
| `survival_rate` | Fraction of rounds survived |
| `damage_per_round` | Mean damage dealt |
| `damage_taken_per_round` | Mean damage received |
| `damage_diff_per_round` | Net damage per round |
| `assists_per_round` | Assists per round |
| `multi_kill_rate` | Fraction of rounds with 2+ kills |
| `rifle_kill_share` | Fraction of kills with rifles |
| `awp_kill_share` | Fraction of kills with AWP |

### Opening Duels

| Column | Description |
|---|---|
| `opening_kill_rate` | Fraction of rounds with an opening kill |
| `opening_death_rate` | Fraction of rounds with an opening death |
| `opening_duel_success` | Win rate in opening duels |
| `first_contact_rate` | Fraction of rounds as first to engage enemy |

### Trading

A trade kill is defined as a kill that occurs within 5 seconds of a teammate's death, against the player who made that kill.

| Column | Description |
|---|---|
| `trade_kill_rate` | Fraction of kills that are trades |
| `death_traded_rate` | Fraction of deaths traded by teammates |
| `trade_participation` | Combined trade involvement rate |

### Utility

| Column | Description |
|---|---|
| `grenades_per_round` | Total grenades per round |
| `he_grenades_per_round` | HE grenades per round |
| `flashbangs_per_round` | Flashbangs per round |
| `smokes_per_round` | Smokes per round |
| `fire_nades_per_round` | Molotovs/incendiaries per round |
| `decoys_per_round` | Decoys per round |
| `flash_assists_per_round` | Flash assists per round |
| `util_damage_per_round` | Damage dealt via utility |

### Positioning and Movement

| Column | Description |
|---|---|
| `avg_distance_to_enemy` | Mean distance to nearest enemy |
| `avg_distance_to_team_centroid` | Mean distance to team centroid |
| `relative_team_centroid_distance` | Distance to centroid, normalized by team spread |
| `avg_distance_moved_per_round` | Total distance traveled per round |
| `avg_distance_to_closest_teammate` | Mean distance to nearest teammate |
| `time_near_enemy_rate` | Fraction of ticks within 750 units of an enemy |
| `time_stationary_rate` | Fraction of ticks with movement ≤ 1 unit |

### Consistency (std columns)

For players with multiple demos, a `{feature}_std` column is computed alongside each rate feature, capturing game-to-game variance. A player with high `awp_kill_share` but high `awp_kill_share_std` may be a situational or secondary AWPer; low std may indicate a dedicated role.

Std columns are only non-zero when more than one demo is parsed.

## Results

- AWPers form an extremely stable and separable cluster across every model.
- T-side IGLs emerge surprisingly well through utility usage and teammate proximity.
- CT-side roles are substantially less separable than T-side roles.
- Positional metrics are among the strongest predictors of role structure.

## Dataset

Professional CS2 match demos from HLTV.org, parsed using [`awpy`](https://github.com/pnxenopoulos/awpy).

Currently using 320 players from 176 demos from **IEM Atlanta 2026** and **PGL Astana 2026**, with planned expansion to include the **IEM Cologne Major 2026** and **CS Asia Championship 2026**. 
This time period was selected because these were high level events took place after the IEM Cologne Major roster locks, meaning team rosters were expected to remain stable until the conclusion of the Major. This should provide more consistent data regarding player roles.

## Dependencies

```bash
pip install awpy polars pandas scikit-learn hdbscan xgboost matplotlib
```

