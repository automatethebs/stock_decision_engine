"""Downloads historical OHLCV data (one-off, offline afterwards) and saves one CSV per stock."""
import os
import requests
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
NIFTY500_CACHE = os.path.join(DATA_DIR, "nifty500_symbols.txt")

# Nifty 50 constituents, yfinance ".NS" (NSE) suffix. Kept as a small, fixed fallback/subset —
# NIFTY500 below is the full universe, fetched live from NSE's official list.
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK", "INFY",
    "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK", "LT",
    "M&M", "MARUTI", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]


def fetch_nifty500(use_cache: bool = True) -> list:
    """Fetches the official Nifty 500 constituent list from NSE. Caches to disk so re-runs
    don't need network access (NSE's list changes only a few times a year on rebalancing)."""
    if use_cache and os.path.exists(NIFTY500_CACHE):
        with open(NIFTY500_CACHE) as f:
            symbols = [line.strip() for line in f if line.strip()]
        if symbols:
            return symbols

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Referer": "https://www.nseindia.com/"}
    session = requests.Session()
    session.headers.update(headers)
    session.get("https://www.nseindia.com", timeout=15)  # sets cookies NSE expects before the CSV request
    resp = session.get("https://archives.nseindia.com/content/indices/ind_nifty500list.csv", timeout=15)
    resp.raise_for_status()
    lines = resp.text.strip().split("\n")
    symbols = [line.split(",")[2].strip() for line in lines[1:] if line.strip()]

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(NIFTY500_CACHE, "w") as f:
        f.write("\n".join(symbols))
    return symbols


def download_stock(symbol: str, period: str = "max") -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")
    df.columns = df.columns.get_level_values(0)  # drop ticker level if multi-indexed
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]]


def download_all(stocks=None, period: str = "max"):
    stocks = stocks or [f"{s}.NS" for s in fetch_nifty500()]
    os.makedirs(DATA_DIR, exist_ok=True)
    saved, skipped = 0, 0
    for symbol in stocks:
        name = symbol.split(".")[0]
        try:
            df = download_stock(symbol, period=period)
        except Exception as e:
            print(f"Skipped {name}: {e}")
            skipped += 1
            continue
        out_path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(out_path)
        print(f"Saved {len(df)} rows for {name} -> {out_path}")
        saved += 1
    print(f"\nDone: {saved} saved, {skipped} skipped out of {len(stocks)}")


if __name__ == "__main__":
    download_all()
