"""Tests for train.py — the time-based split must never leak future data into training."""
import pandas as pd
import pytest

from train import time_based_split, TEST_FRACTION
from features import build_dataset


def _pooled():
    """Two stocks with different history lengths and overlapping calendars."""
    idx_a = pd.bdate_range("2020-01-02", periods=120)
    idx_b = pd.bdate_range("2021-06-01", periods=60)
    df = pd.DataFrame({"stock": ["A"] * 120 + ["B"] * 60,
                       "value": list(range(120)) + list(range(60))},
                      index=pd.DatetimeIndex(list(idx_a) + list(idx_b)))
    return df.sort_index()


class TestTimeBasedSplit:
    def test_partition_covers_all_rows(self):
        df = _pooled()
        train, test = time_based_split(df)
        assert len(train) + len(test) == len(df)

    def test_no_row_in_both(self):
        df = _pooled()
        train, test = time_based_split(df)
        train_keys = set(zip(train["stock"], train.index))
        test_keys = set(zip(test["stock"], test.index))
        assert train_keys.isdisjoint(test_keys)

    def test_test_is_last_fraction_per_stock(self):
        df = _pooled()
        train, test = time_based_split(df)
        for stock, group in df.groupby("stock"):
            group = group.sort_index()
            split_idx = int(len(group) * (1 - TEST_FRACTION))
            assert set(train[train["stock"] == stock].index) == set(group.index[:split_idx])
            assert set(test[test["stock"] == stock].index) == set(group.index[split_idx:])

    def test_chronological_order_preserved(self):
        df = _pooled()
        train, test = time_based_split(df)
        assert train.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing

    def test_with_built_features(self, ohlcv):
        """Sanity: the split works on the actual pooled feature pipeline output."""
        df = build_dataset(ohlcv, horizon=5, deadband=0.01)
        df["stock"] = "S"
        train, test = time_based_split(df)
        assert set(train.index).isdisjoint(set(test.index))
        assert test.index.min() > train.index.max()
