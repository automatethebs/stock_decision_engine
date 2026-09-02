"""Backtests the saved model on held-out (time-based) data per stock, and compares
against a simple buy-and-hold strategy."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from train import load_pooled_dataset, time_based_split
from features import FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1
    return drawdown.min()


def backtest_stock(stock_df: pd.DataFrame, model, model_name, scaler, horizon: int) -> dict:
    X = stock_df[FEATURE_COLUMNS]
    X_eval = scaler.transform(X) if model_name == "logistic_regression" else X
    preds = model.predict(X_eval)
    actual = stock_df["target"].astype(int).values
    fwd_return = stock_df["fwd_return"].values

    # Strategy: go long when model predicts 1, else stay in cash (return 0).
    strategy_returns = np.where(preds == 1, fwd_return, 0.0)
    buy_hold_returns = fwd_return

    # Drawdown must use NON-OVERLAPPING periods (every `horizon`-th row) — the daily rows
    # above are overlapping 30-day forward windows, and compounding those day-by-day would
    # double-count the same price move many times over, producing a meaningless drawdown.
    non_overlap = stock_df.iloc[::horizon]
    non_overlap_preds = model.predict(scaler.transform(non_overlap[FEATURE_COLUMNS]) if model_name == "logistic_regression" else non_overlap[FEATURE_COLUMNS])
    non_overlap_strategy_returns = np.where(non_overlap_preds == 1, non_overlap["fwd_return"].values, 0.0)
    non_overlap_buy_hold_returns = non_overlap["fwd_return"].values

    baseline = max(actual.mean(), 1 - actual.mean())
    accuracy = accuracy_score(actual, preds)

    return {
        "accuracy": accuracy,
        "baseline": baseline,
        "edge_over_baseline": accuracy - baseline,
        "precision": precision_score(actual, preds, zero_division=0),
        "recall": recall_score(actual, preds, zero_division=0),
        "win_rate": (strategy_returns[preds == 1] > 0).mean() if (preds == 1).any() else float("nan"),
        "avg_strategy_return": strategy_returns.mean(),
        "avg_buy_hold_return": buy_hold_returns.mean(),
        "strategy_max_drawdown": max_drawdown((1 + pd.Series(non_overlap_strategy_returns)).cumprod()),
        "buy_hold_max_drawdown": max_drawdown((1 + pd.Series(non_overlap_buy_hold_returns)).cumprod()),
        "n_trades": int((preds == 1).sum()),
        "n_periods": len(actual),
        "n_nonoverlapping_periods": len(non_overlap),
    }


def run_backtest():
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    model, model_name, scaler = bundle["model"], bundle["model_name"], bundle["scaler"]
    horizon = bundle["horizon"]

    df = load_pooled_dataset()
    _, test_df = time_based_split(df)

    results = []
    for stock, group in test_df.groupby("stock"):
        metrics = backtest_stock(group.sort_index(), model, model_name, scaler, horizon)
        metrics["stock"] = stock
        results.append(metrics)
    return pd.DataFrame(results).set_index("stock")


def main():
    results = run_backtest()
    for stock, row in results.iterrows():
        print(f"--- {stock} ---")
        for k, v in row.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()
    return results


if __name__ == "__main__":
    main()
