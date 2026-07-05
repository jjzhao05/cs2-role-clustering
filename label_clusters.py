from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


INPUT_DIR = Path("outputs")
RANDOM_STATE = 42
TEST_SIZE = 0.25


def get_features(df):
    exclude = {
        "player_name",
        "side",
        "cluster",
        "pc1",
        "pc2",
    }

    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = X.loc[:, X.nunique() > 1]

    if X.empty:
        raise ValueError("No usable feature columns.")

    return X


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

    if class_counts.min() < 2:
        print(f"Skipping {path.name}: at least one cluster has fewer than 2 players.")
        print("Cluster counts:")
        print(pd.Series(y).value_counts().sort_index())
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

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

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(f"\nModel trained for: {path}")
    print(classification_report(
        y_test,
        preds,
        target_names=[str(c) for c in encoder.classes_],
    ))

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_dir = path.parent / "xgboost_labels"
    output_dir.mkdir(exist_ok=True)

    importance.to_csv(
        output_dir / f"{path.stem}_feature_importance.csv",
        index=False,
    )

    labeled = df.copy()
    labeled["predicted_cluster"] = encoder.inverse_transform(
        model.predict(X)
    )

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

    for path in cluster_files:
        train_label_model(path)


if __name__ == "__main__":
    main()