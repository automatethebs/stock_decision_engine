"""Trains and compares classification models on pooled multi-stock data, using a
time-based split (no shuffling) to avoid future data leakage. Saves the best model.

Model selection is by EDGE OVER THE MAJORITY-CLASS BASELINE, not raw accuracy — raw
accuracy is misleading here because it rises with a wider deadband purely from growing
class imbalance, not from any real skill. See README "Known limitations" for the
sweep/walk-forward evidence behind this."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

from features import FEATURE_COLUMNS
from pooled import load_pooled_dataset as _load_pooled_dataset

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

HORIZON = 30
DEADBAND = 0.03  # 3% deadband: |move| < 3% over the horizon is "neutral" and dropped
TEST_FRACTION = 0.2  # last 20% of each stock's timeline held out, time-ordered


def load_pooled_dataset():
    """Pooled feature dataset for all stocks — disk-cached by pooled.py (invalidated
    automatically when any data CSV or the horizon/deadband config changes)."""
    return _load_pooled_dataset(horizon=HORIZON, deadband=DEADBAND)


def time_based_split(df: pd.DataFrame):
    """Splits each stock's rows by time (last TEST_FRACTION as test), then combines.
    Never shuffles — order within a stock stays chronological."""
    train_parts, test_parts = [], []
    for stock, group in df.groupby("stock"):
        group = group.sort_index()
        split_idx = int(len(group) * (1 - TEST_FRACTION))
        train_parts.append(group.iloc[:split_idx])
        test_parts.append(group.iloc[split_idx:])
    return pd.concat(train_parts), pd.concat(test_parts)


def evaluate(name, model, X_test, y_test, baseline):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    edge = acc - baseline
    print(f"{name:20s} accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} edge_over_baseline={edge:+.4f}")
    return edge


def main():
    df = load_pooled_dataset()
    train_df, test_df = time_based_split(df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target"].astype(int)
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target"].astype(int)
    baseline = y_test.value_counts(normalize=True).max()

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # CalibratedClassifierCV internally cross-validates on the training set only —
    # no test-set leakage — and turns overconfident tree/linear scores into probabilities
    # that better match observed frequencies.
    models = {
        "logistic_regression": CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), method="sigmoid", cv=5,
        ).fit(X_train_scaled, y_train),
        "random_forest": CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0), method="isotonic", cv=5,
        ).fit(X_train, y_train),
        "xgboost": CalibratedClassifierCV(
            XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=0),
            method="isotonic", cv=5,
        ).fit(X_train, y_train),
    }

    print(f"\nTrain rows: {len(train_df)}  Test rows: {len(test_df)}  Baseline (majority class): {baseline:.3f}\n")
    edges = {}
    for name, model in models.items():
        X_eval = X_test_scaled if name == "logistic_regression" else X_test
        edges[name] = evaluate(name, model, X_eval, y_test, baseline)

    best_name = max(edges, key=edges.get)
    best_model = models[best_name]
    print(f"\nBest model: {best_name} (edge_over_baseline={edges[best_name]:+.4f})")
    if edges[best_name] <= 0.005:
        print(
            "NOTE: no model beats the majority-class baseline by a meaningful margin. "
            "This means price-technical features alone do not carry a reliable directional "
            "edge for this stock set/horizon — a known, expected result, not a bug. "
            "See README 'Known limitations'."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler if best_name == "logistic_regression" else None,
        "feature_columns": FEATURE_COLUMNS,
        "horizon": HORIZON,
        "deadband": DEADBAND,
        "baseline_accuracy": baseline,
    }, os.path.join(MODEL_DIR, "model.joblib"))
    print(f"Saved model -> {os.path.join(MODEL_DIR, 'model.joblib')}")


if __name__ == "__main__":
    main()
