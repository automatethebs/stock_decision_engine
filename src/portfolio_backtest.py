"""Tests whether ranking stocks by model probability and holding a top-K portfolio
(rebalanced every `horizon` trading days, non-overlapping — no double-counting) is
profitable net of realistic transaction costs, versus an equal-weight buy-and-hold of
the same universe. Only uses the held-out test period the model never trained on.

This does NOT create edge from nothing — if the underlying per-stock signal has no real
edge (see README), diversification reduces variance around that edge, it can't turn a
~0% edge into a positive one. This script measures the honest answer either way."""
import os
import joblib
import numpy as np
import pandas as pd

from train import load_pooled_dataset, time_based_split
from features import FEATURE_COLUMNS

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TOP_K = 10
ROUND_TRIP_COST = 0.004  # ~0.4%: brokerage + STT + stamp duty + slippage, conservative estimate


def get_predictions(test_df, model, model_name, scaler):
    X = test_df[FEATURE_COLUMNS]
    X_eval = scaler.transform(X) if model_name == "logistic_regression" else X
    proba = model.predict_proba(X_eval)[:, 1]
    return test_df.assign(pred_proba=proba)


def build_rebalance_snapshots(test_df, horizon, stale_tolerance_days=10):
    """For each rebalance date, the most recent row per stock (within tolerance)."""
    all_dates = test_df.index.unique().sort_values()
    rebalance_dates = all_dates[::horizon]

    snapshots = {}
    for rdate in rebalance_dates:
        rows = []
        for stock, group in test_df.groupby("stock"):
            recent = group[group.index <= rdate]
            if recent.empty:
                continue
            last_row = recent.iloc[-1]
            if (rdate - recent.index[-1]).days > stale_tolerance_days:
                continue
            rows.append(last_row)
        if rows:
            snapshots[rdate] = pd.DataFrame(rows)
    return snapshots


def run_portfolio_backtest(top_k=TOP_K, cost=ROUND_TRIP_COST):
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    model, model_name, scaler, horizon = bundle["model"], bundle["model_name"], bundle["scaler"], bundle["horizon"]

    df = load_pooled_dataset()
    _, test_df = time_based_split(df)
    test_df = get_predictions(test_df, model, model_name, scaler)

    snapshots = build_rebalance_snapshots(test_df, horizon)

    records = []
    for rdate, snap in snapshots.items():
        snap = snap.dropna(subset=["fwd_return"])
        if len(snap) < top_k:
            continue
        ranked = snap.sort_values("pred_proba", ascending=False)
        top_picks = ranked.head(top_k)

        portfolio_return = top_picks["fwd_return"].mean() - cost
        benchmark_return = snap["fwd_return"].mean()  # equal-weight ALL available stocks, no cost (buy-and-hold, no rebalance trades)

        records.append({
            "date": rdate,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "n_stocks_available": len(snap),
            "top_picks": ", ".join(top_picks["stock"].tolist()),
        })

    return pd.DataFrame(records).set_index("date")


def summarize(returns: pd.Series, periods_per_year: float) -> dict:
    cum = (1 + returns).cumprod()
    n_years = len(returns) / periods_per_year
    cagr = cum.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan
    running_max = cum.cummax()
    max_dd = (cum / running_max - 1).min()
    win_rate = (returns > 0).mean()
    return {
        "CAGR": cagr, "Sharpe": sharpe, "Max Drawdown": max_dd,
        "Win rate": win_rate, "Total return": cum.iloc[-1] - 1, "Periods": len(returns),
    }


def main():
    results = run_portfolio_backtest()
    bundle = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    horizon = bundle["horizon"]
    periods_per_year = 252 / horizon

    print(f"Rebalance periods: {len(results)}  Top-K: {TOP_K}  Round-trip cost: {ROUND_TRIP_COST*100:.1f}%\n")

    port_stats = summarize(results["portfolio_return"], periods_per_year)
    bench_stats = summarize(results["benchmark_return"], periods_per_year)

    print(f"{'Metric':20s} {'Top-K Portfolio':>18s} {'Equal-Weight Benchmark':>24s}")
    for k in port_stats:
        pv, bv = port_stats[k], bench_stats[k]
        if k in ("CAGR", "Max Drawdown", "Win rate", "Total return"):
            print(f"{k:20s} {pv*100:>17.2f}% {bv*100:>23.2f}%")
        else:
            print(f"{k:20s} {pv:>18.3f} {bv:>24.3f}")

    return results


if __name__ == "__main__":
    main()
