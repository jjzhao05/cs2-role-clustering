# CS2 Role Clustering

Unsupervised machine learning project that discovers professional Counter-Strike 2 player archetypes from match demo data.

The pipeline parses raw `.dem` files, engineers behavioral and positional features, clusters players using multiple algorithms, and evaluates cluster quality using stability analysis and external role labels from Liquipedia.

Major changes currently happening, so code and report will not match!!

[Read the full project report](Report.pdf)

## Dataset

- 176 professional match demos
- 320 professional players
- IEM Atlanta 2026 and PGL Astana 2026

## Methods

- Feature engineering from combat, utility, trading, movement, and positioning data
- KMeans, Gaussian Mixture Models (GMM), and HDBSCAN
- Bootstrap stability analysis using Adjusted Rand Index (ARI)
- PCA-based visualization
- Random Forest and XGBoost for cluster interpretation

## Key Findings

- AWPers form an extremely stable and highly separable cluster across all models.
- T-side roles are substantially more separable than CT-side roles.
- T-side IGLs emerge as a distinct cluster despite no direct leadership labels in the data.
- Positional features are among the strongest predictors of role structure.

## Pipeline

```text
.dem files
    ↓
demo_parser.py
    ↓
feature dataset
    ↓
cluster_players.py
    ↓
evaluate_clusters.py
    ↓
plotter.py
```

## Run

Run the full pipeline:

```bash
python main.py
```

Or run stages individually:

```bash
python demo_parser.py <demos_dir> output.csv
python cluster_players.py
python evaluate_clusters.py
python plotter.py
```
