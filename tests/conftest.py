"""Shared test fixtures. Puts the src/ directory on sys.path so the flat imports used
inside the project (e.g. `from features import ...`) resolve, and provides a synthetic
OHLCV frame for unit tests that must not touch the network or the real data/ directory."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """~2 years of deterministic synthetic OHLCV data, indexed by trading day."""
    rng = np.random.default_rng(42)
    n = 520
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * 0.999
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture
def fundamentals() -> pd.DataFrame:
    """5 annual rows of fundamentals, indexed by the 'known_from' date."""
    rows = []
    for year in range(2019, 2024):
        year_end = pd.Timestamp(f"{year}-03-31")
        rows.append({
            "known_from": year_end + pd.Timedelta(days=60),
            "roe": 0.15 + 0.01 * year,
            "debt_to_equity": 0.5,
            "profit_margin": 0.10,
            "revenue_growth_yoy": 0.12,
            "eps": 12.0,
            "book_value_per_share": 80.0,
        })
    return pd.DataFrame(rows).set_index("known_from")
