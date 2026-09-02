"""Tests for pooled.py — the disk cache must be correct AND actually invalidate when
CSVs or the (horizon, deadband) config change."""
import numpy as np
import pandas as pd
import pytest

import pooled
from features import FEATURE_COLUMNS


def _write_stock_csv(data_dir, name, n=150, start="2023-01-02"):
    idx = pd.bdate_range(start, periods=n)
    # ndarray (not Series) on purpose: a Series would carry a RangeIndex that doesn't
    # align with idx, silently turning every price column into NaN in the constructor.
    close = np.arange(100, 100 + n, dtype=float)
    df = pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1_000_000,
    }, index=idx)
    df.to_csv(data_dir / f"{name}.csv", index_label="Date")


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_stock_csv(data_dir, "AAA")
    _write_stock_csv(data_dir, "BBB", n=120)
    monkeypatch.setattr(pooled, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(pooled, "CACHE_PATH", str(tmp_path / "pooled.joblib"))
    return data_dir


def _count_builds(monkeypatch):
    """Wraps pooled._build with a call counter; returns the counter dict."""
    calls = {"n": 0}
    original = pooled._build

    def counting_build(horizon, deadband):
        calls["n"] += 1
        return original(horizon, deadband)

    monkeypatch.setattr(pooled, "_build", counting_build)
    return calls


class TestBuildAndCache:
    def test_returns_stock_tagged_frame(self, env):
        df = pooled.load_pooled_dataset()
        assert set(df["stock"].unique()) == {"AAA", "BBB"}
        for c in FEATURE_COLUMNS + ["target", "fwd_return", "stock"]:
            assert c in df.columns
        assert df.index.is_monotonic_increasing

    def test_second_call_uses_cache(self, env, monkeypatch):
        calls = _count_builds(monkeypatch)
        pooled.load_pooled_dataset()
        pooled.load_pooled_dataset()
        assert calls["n"] == 1  # second call must NOT rebuild

    def test_cache_equals_fresh_build(self, env):
        df_cached = pooled.load_pooled_dataset(horizon=10, deadband=0.02)
        df_fresh = pooled.load_pooled_dataset(horizon=10, deadband=0.02, use_cache=False)
        pd.testing.assert_frame_equal(df_cached, df_fresh)


class TestInvalidation:
    def test_rebuilds_when_csv_changes(self, env, monkeypatch):
        calls = _count_builds(monkeypatch)
        pooled.load_pooled_dataset()
        assert calls["n"] == 1

        # grow one stock's history -> its size (and mtime) change
        path = env / "AAA.csv"
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        extra_idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=10)
        extra = pd.DataFrame({c: df[c].iloc[-1] for c in df.columns}, index=extra_idx)
        pd.concat([df, extra]).to_csv(path, index_label="Date")

        pooled.load_pooled_dataset()
        assert calls["n"] == 2  # manifest mismatch -> rebuild

    def test_rebuilds_when_config_changes(self, env, monkeypatch):
        calls = _count_builds(monkeypatch)
        pooled.load_pooled_dataset(horizon=5, deadband=0.01)
        pooled.load_pooled_dataset(horizon=5, deadband=0.01)
        assert calls["n"] == 1  # same config -> cache hit

        pooled.load_pooled_dataset(horizon=10, deadband=0.01)
        assert calls["n"] == 2  # different horizon -> rebuild

        pooled.load_pooled_dataset(horizon=5, deadband=0.02)
        assert calls["n"] == 3  # different deadband -> rebuild

    def test_empty_data_dir_returns_empty(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        monkeypatch.setattr(pooled, "DATA_DIR", str(data_dir))
        monkeypatch.setattr(pooled, "CACHE_PATH", str(tmp_path / "pooled.joblib"))
        df = pooled.load_pooled_dataset()
        assert df.empty
