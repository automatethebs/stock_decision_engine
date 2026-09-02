"""Downloads corporate action history (dividends, splits) per stock and derives simple,
fully backtestable event-proximity features — this data has decades of real timestamps,
unlike quarterly/annual fundamentals, so it doesn't restrict the training window."""
import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EVENTS_DIR = os.path.join(DATA_DIR, "events")

EVENT_COLUMNS = ["days_since_dividend", "days_since_split"]


def download_events(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    dividends = t.dividends
    splits = t.splits
    dividends.index = dividends.index.tz_localize(None)
    splits.index = splits.index.tz_localize(None)
    return pd.DataFrame({
        "dividend_date": pd.Series(dividends.index),
    }), pd.DataFrame({"split_date": pd.Series(splits.index)})


def download_all(stocks):
    os.makedirs(EVENTS_DIR, exist_ok=True)
    for stock in stocks:
        ticker = f"{stock}.NS"
        try:
            div_df, split_df = download_events(ticker)
        except Exception as e:
            print(f"Skipped {stock}: {e}")
            continue
        div_df.to_csv(os.path.join(EVENTS_DIR, f"{stock}_dividends.csv"), index=False)
        split_df.to_csv(os.path.join(EVENTS_DIR, f"{stock}_splits.csv"), index=False)
        print(f"Saved {len(div_df)} dividend dates, {len(split_df)} split dates for {stock}")


def add_events(df: pd.DataFrame, stock: str) -> pd.DataFrame:
    """Adds days_since_dividend / days_since_split (capped at 365, -1 if none yet).
    Uses only PAST event dates relative to each row — no leakage."""
    df = df.copy()
    div_path = os.path.join(EVENTS_DIR, f"{stock}_dividends.csv")
    split_path = os.path.join(EVENTS_DIR, f"{stock}_splits.csv")

    def days_since(event_dates: pd.Series) -> pd.Series:
        if event_dates.empty:
            return pd.Series(-1, index=df.index)
        event_dates = pd.to_datetime(event_dates).sort_values()
        idx = np.searchsorted(event_dates.values, df.index.values, side="right") - 1
        result = np.full(len(df), -1.0)
        valid = idx >= 0
        result[valid] = (df.index[valid] - event_dates.values[idx[valid]]).days
        return pd.Series(np.clip(result, -1, 365), index=df.index)

    div_dates = pd.read_csv(div_path)["dividend_date"] if os.path.exists(div_path) else pd.Series(dtype="datetime64[ns]")
    split_dates = pd.read_csv(split_path)["split_date"] if os.path.exists(split_path) else pd.Series(dtype="datetime64[ns]")

    df["days_since_dividend"] = days_since(div_dates)
    df["days_since_split"] = days_since(split_dates)
    return df


if __name__ == "__main__":
    from data_loader import fetch_nifty500
    download_all(fetch_nifty500())
