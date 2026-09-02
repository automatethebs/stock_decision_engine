"""CLI decision engine: python predict.py RELIANCE"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

from features import build_dataset, add_features, FEATURE_COLUMNS
from events import add_events
from news import get_news_sentiment
from pooled import load_pooled_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")


def load_model():
    return joblib.load(os.path.join(MODEL_DIR, "model.joblib"))


def risk_level(volatility_20: float) -> str:
    if volatility_20 < 0.015:
        return "Low"
    if volatility_20 < 0.03:
        return "Medium"
    return "High"


def feature_importance(bundle) -> pd.Series:
    model = bundle["model"]
    # CalibratedClassifierCV wraps the fitted base estimator(s); average importances across folds.
    if hasattr(model, "calibrated_classifiers_"):
        base_estimators = [cc.estimator for cc in model.calibrated_classifiers_]
        if hasattr(base_estimators[0], "feature_importances_"):
            values = np.mean([e.feature_importances_ for e in base_estimators], axis=0)
        else:
            values = np.mean([abs(e.coef_[0]) for e in base_estimators], axis=0)
        return pd.Series(values, index=bundle["feature_columns"]).sort_values(ascending=False)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=bundle["feature_columns"]).sort_values(ascending=False)
    if hasattr(model, "coef_"):
        return pd.Series(abs(model.coef_[0]), index=bundle["feature_columns"]).sort_values(ascending=False)
    return pd.Series(dtype=float)


def predict(stock: str, bundle=None) -> dict:
    bundle = bundle or load_model()
    model, model_name, scaler = bundle["model"], bundle["model_name"], bundle["scaler"]

    csv_path = os.path.join(DATA_DIR, f"{stock}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data for {stock}. Run data_loader.py first.")

    raw = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df = build_dataset(raw, horizon=bundle["horizon"], deadband=bundle["deadband"])
    if df.empty:
        raise ValueError(f"Not enough history for {stock} to compute features.")

    latest = df.iloc[[-1]]
    X = latest[FEATURE_COLUMNS]
    X_eval = scaler.transform(X) if model_name == "logistic_regression" else X

    proba = model.predict_proba(X_eval)[0]  # [P(class=0), P(class=1)]
    positive_prob = proba[1]
    negative_prob = proba[0]

    return {
        "stock": stock,
        "horizon_days": bundle["horizon"],
        "positive_probability": round(positive_prob * 100, 1),
        "negative_probability": round(negative_prob * 100, 1),
        "risk_level": risk_level(latest["volatility_20"].iloc[0]),
        "as_of": latest.index[0].date().isoformat(),
        "feature_importance": feature_importance(bundle),
    }


def get_fundamentals_snapshot(stock: str) -> dict:
    """Most recently reported fundamentals for this stock, if available. Informational —
    not fed into the trained model (see README: only ~4yrs of free history, tested
    separately, not integrated into the default model)."""
    path = os.path.join(FUNDAMENTALS_DIR, f"{stock}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col="known_from", parse_dates=True)
    if df.empty:
        return None
    latest = df.sort_index().iloc[-1]
    return {
        "fiscal_year_end": pd.to_datetime(latest["fiscal_year_end"]).date().isoformat(),
        "roe_pct": round(latest["roe"] * 100, 1) if pd.notna(latest["roe"]) else None,
        "debt_to_equity": round(latest["debt_to_equity"], 2) if pd.notna(latest["debt_to_equity"]) else None,
        "profit_margin_pct": round(latest["profit_margin"] * 100, 1) if pd.notna(latest["profit_margin"]) else None,
        "revenue_growth_yoy_pct": round(latest["revenue_growth_yoy"] * 100, 1) if pd.notna(latest["revenue_growth_yoy"]) else None,
    }


def get_events_snapshot(stock: str) -> dict:
    """Days since last dividend/split, as of the latest available price date. Informational
    — tested for predictive edge and found none (see README), shown as factual context only."""
    csv_path = os.path.join(DATA_DIR, f"{stock}.csv")
    if not os.path.exists(csv_path):
        return None
    raw = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    featured = add_features(raw)
    with_events = add_events(featured, stock)
    latest = with_events.iloc[-1]
    return {
        "days_since_dividend": int(latest["days_since_dividend"]) if latest["days_since_dividend"] >= 0 else None,
        "days_since_split": int(latest["days_since_split"]) if latest["days_since_split"] >= 0 else None,
    }


def get_news_context(stock: str) -> dict:
    """Live news sentiment — informational only, never backtested (see README), not used
    by the trained model."""
    try:
        return get_news_sentiment(stock)
    except Exception:
        return {"headlines": [], "mean_compound": 0.0, "label": "Unavailable"}


def recommend(positive_probability: float) -> str:
    if positive_probability >= 60:
        return "BUY"
    if positive_probability <= 40:
        return "SELL"
    return "HOLD"


def predict_all(stocks, bundle=None) -> pd.DataFrame:
    """Screener: one pooled feature build + a single predict_proba call over every stock's
    latest row, instead of the old per-stock loop that rebuilt each stock's features from
    scratch (500x redundant work — the source of the screener tab's slow first load).

    Output matches the per-stock predict() exactly: one row per stock (stocks with too
    little history are skipped, same as before), sorted by positive probability."""
    bundle = bundle or load_model()
    model, model_name, scaler = bundle["model"], bundle["model_name"], bundle["scaler"]

    df = load_pooled_dataset(horizon=bundle["horizon"], deadband=bundle["deadband"])
    if df.empty:  # no CSVs at all -> frame without a 'stock' column; don't KeyError below
        return pd.DataFrame(columns=["stock", "recommendation", "positive_probability",
                                     "negative_probability", "risk_level", "as_of"])
    df = df[df["stock"].isin(stocks)]
    if df.empty:
        return pd.DataFrame(columns=["stock", "recommendation", "positive_probability",
                                     "negative_probability", "risk_level", "as_of"])
    # Chronologically last surviving row per stock — identical selection to predict()'s
    # df.iloc[[-1]] per stock (the pooled build drops the same NaN-target/feature rows).
    latest = df.sort_index().groupby("stock").tail(1)

    # Caller's stock order, so equal probabilities tie-break exactly like the old
    # per-stock loop (which emitted rows in the order stocks were listed).
    caller_order = latest["stock"].map({s: i for i, s in enumerate(stocks)})
    latest = latest.assign(_order=caller_order).sort_values("_order", kind="stable").drop(columns="_order")

    X = latest[FEATURE_COLUMNS]
    X_eval = scaler.transform(X) if model_name == "logistic_regression" else X
    proba = model.predict_proba(X_eval)  # [P(class=0), P(class=1)] per stock

    rows = []
    for i, (as_of, row) in enumerate(latest.iterrows()):
        positive_prob = round(proba[i, 1] * 100, 1)
        rows.append({
            "stock": row["stock"],
            "recommendation": recommend(positive_prob),
            "positive_probability": positive_prob,
            "negative_probability": round(proba[i, 0] * 100, 1),
            "risk_level": risk_level(row["volatility_20"]),
            "as_of": as_of.date().isoformat(),
        })
    return pd.DataFrame(rows).sort_values("positive_probability", ascending=False, kind="stable").reset_index(drop=True)


def print_report(result: dict):
    print(f"Stock: {result['stock']}")
    print(f"Prediction period: {result['horizon_days']} days")
    print(f"As of: {result['as_of']}\n")
    print(f"Positive probability: {result['positive_probability']}%")
    print(f"Negative probability: {result['negative_probability']}%")
    print(f"Risk level: {result['risk_level']}\n")
    print("Top factors:")
    for i, (feat, score) in enumerate(result["feature_importance"].head(5).items(), 1):
        print(f"{i}. {feat} ({score:.4f})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <STOCK_SYMBOL>")
        sys.exit(1)
    result = predict(sys.argv[1].upper())
    print_report(result)
