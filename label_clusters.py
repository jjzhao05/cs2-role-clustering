from pathlib import Path
import shutil

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder

from feature_config import select_model_features


INPUT_DIR = Path("outputs")
RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 5


def get_features(df):
    return select_model_features(df)


def train_label_model(path):
    df = pd.read_csv(path)

    if "cluster" not in df.columns:
        raise ValueError(f"No cluster column found in {path}")

    X = get_features(df)
    y = df["cluster"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if len(set(y_encoded)) < 2:
        print(f"Skipping {path.name}: only one cluster.")
        return

    class_counts = pd.Series(y_encoded).value_counts()

    n_splits = min(CV_SPLITS, int(class_counts.min()))
    if n_splits < 2:
        print(f"Skipping {path.name}: at least one cluster has fewer than 2 players.")
        print("Cluster counts:")
        print(pd.Series(y).value_counts().sort_index())
        return

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    X_values = X.to_numpy()
    votes = np.zeros((len(y_encoded), len(encoder.classes_)), dtype=np.int64)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_values, y_encoded)):
        fold_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE + fold,
        )
        fold_model.fit(X_values[train_idx], y_encoded[train_idx])
        fold_preds = fold_model.predict(X_values[test_idx]).astype(int)
        votes[test_idx, fold_preds] += 1

    preds = votes.argmax(axis=1)

    print(f"\nModel trained for: {path}")
    print(classification_report(
        y_encoded,
        preds,
        target_names=[str(c) for c in encoder.classes_],
    ))

    # Fit one full-data surrogate only for descriptive feature importance.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_values, y_encoded)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_dir = path.parent / "surrogate_labels"
    output_dir.mkdir(exist_ok=True)

    importance.to_csv(
        output_dir / f"{path.stem}_feature_importance.csv",
        index=False,
    )

    labeled = df.copy()
    labeled["predicted_cluster"] = encoder.inverse_transform(preds)

    labeled.to_csv(
        output_dir / f"{path.stem}_predicted_labels.csv",
        index=False,
    )

    print("Top label-driving features:")
    print(importance.head(15))


def main():
    cluster_files = list(INPUT_DIR.glob("*/*_player_clusters.csv"))

    if not cluster_files:
        raise ValueError("No cluster assignment files found.")

    for side_dir in {path.parent for path in cluster_files}:
        shutil.rmtree(side_dir / "surrogate_labels", ignore_errors=True)

    for path in sorted(cluster_files):
        train_label_model(path)


if __name__ == "__main__":
    main()
