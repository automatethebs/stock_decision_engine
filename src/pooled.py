"""Builds (and disk-caches) the pooled multi-stock feature dataset shared by train.py,
backtest.py, portfolio_backtest.py and the screener's vectorized predict_all().

Rebuilding the full Nifty-500 feature dataset takes ~60s (500 CSVs, ~1.5M rows). The cache
makes every later run/tab reuse it in a few seconds. Invalidation is automatic: the cache
stores a manifest of every stock CSV's (name, mtime, size) plus the (horizon, deadband)
config, and is rebuilt whenever any of those change — so it can never serve stale features
or a stale train/test split."""
import glob
import os
import joblib
import pandas as pd

from features import build_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(DATA_DIR, "pooled_dataset.joblib")

DEFAULT_HORIZON = 30
DEFAULT_DEADBAND = 0.03


def _manifest():
    """Sorted (name, mtime, size) of every stock CSV — the data half of the cache key."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    return [(os.path.basename(p), os.path.getmtime(p), os.path.getsize(p)) for p in files]


def _build(horizon, deadband):
    """Fresh build: features + target for every stock CSV, tagged with a 'stock' column."""
    frames = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.csv")):
        stock = os.path.splitext(os.path.basename(path))[0]
        df = build_dataset(
            pd.read_csv(path, index_col="Date", parse_dates=True),
            horizon=horizon, deadband=deadband,
        )
        df["stock"] = stock
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def load_pooled_dataset(horizon=DEFAULT_HORIZON, deadband=DEFAULT_DEADBAND, use_cache=True):
    """Pooled feature dataset for all data/*.csv stocks.

    Returns the same rows/columns as the old per-run rebuild (features, fwd_return,
    target, plus a 'stock' column), but reuses a disk cache when no CSV and no config
    has changed since it was written. Corrupt caches fall back to a fresh build."""
    manifest = _manifest()
    if use_cache and os.path.exists(CACHE_PATH):
        try:
            cached = joblib.load(CACHE_PATH)
            if (cached.get("manifest") == manifest
                    and cached.get("horizon") == horizon
                    and cached.get("deadband") == deadband):
                return cached["df"]
        except Exception:
            pass  # corrupt or unreadable cache -> rebuild below

    df = _build(horizon, deadband)
    os.makedirs(DATA_DIR, exist_ok=True)
    joblib.dump({"manifest": manifest, "horizon": horizon, "deadband": deadband, "df": df}, CACHE_PATH)
    return df
