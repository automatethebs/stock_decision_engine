"""Tests for features.py — indicator construction and target labelling."""
import numpy as np
import pandas as pd
import pytest

from features import (
    add_features, add_target, build_dataset,
    FEATURE_COLUMNS, EXTENDED_FEATURE_COLUMNS,
    add_fundamentals,
)


class TestAddFeatures:
    def test_all_core_columns_present(self, ohlcv):
        out = add_features(ohlcv)
        missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
        assert missing == []

    def test_extended_columns_superset(self):
        assert set(FEATURE_COLUMNS) <= set(EXTENDED_FEATURE_COLUMNS)
        assert len(EXTENDED_FEATURE_COLUMNS) == len(FEATURE_COLUMNS) + 7

    def test_no_inf_values_after_warmup(self, ohlcv):
        out = add_features(ohlcv)
        warm = out.iloc[200:]  # skip indicator warm-up window
        assert np.isfinite(warm[FEATURE_COLUMNS].to_numpy()).all()

    def test_rsi_bounded(self, ohlcv):
        out = add_features(ohlcv)
        rsi = out["rsi_14"].dropna()
        assert ((rsi >= 0) & (rsi <= 100)).all()

    def test_returns_input_unchanged(self, ohlcv):
        before = ohlcv.copy()
        add_features(ohlcv)
        pd.testing.assert_frame_equal(ohlcv, before)  # must not mutate caller's frame


class TestAddTarget:
    @pytest.fixture
    def df(self):
        # Known closes: 100, 102, 98, 101, 103 with horizon=2
        return pd.DataFrame({"Close": [100.0, 102.0, 98.0, 101.0, 103.0]},
                            index=pd.bdate_range("2023-01-02", periods=5))

    def test_labels_above_and_below_deadband(self, df):
        out = add_target(df, horizon=2, deadband=0.01)
        # fwd return of row0 = 98/100-1 = -2% -> 0; row1 = 101/102-1 ≈ -0.98% -> neutral (NaN)
        assert out.loc[out.index[0], "target"] == 0
        assert pd.isna(out.loc[out.index[1], "target"])
        # row2 = 103/98-1 ≈ +5.1% -> 1
        assert out.loc[out.index[2], "target"] == 1
        # last `horizon` rows have no forward return
        assert pd.isna(out.loc[out.index[-1], "target"])
        assert pd.isna(out.loc[out.index[-2], "target"])


class TestBuildDataset:
    def test_drops_incomplete_rows(self, ohlcv):
        out = build_dataset(ohlcv, horizon=5, deadband=0.01)
        assert out["target"].notna().all()
        assert out[FEATURE_COLUMNS].notna().all().all()
        assert len(out) < len(ohlcv)

    def test_keeps_chronological_order(self, ohlcv):
        out = build_dataset(ohlcv, horizon=5, deadband=0.01)
        assert out.index.is_monotonic_increasing

    def test_with_fundamentals_adds_columns(self, ohlcv, fundamentals):
        out = build_dataset(ohlcv, horizon=5, deadband=0.01, fundamentals=fundamentals)
        for c in ["roe", "debt_to_equity", "profit_margin", "pe_ratio"]:
            assert c in out.columns


class TestAddFundamentals:
    def test_no_lookahead(self, fundamentals):
        """A daily row dated BEFORE the first known_from must get NaN fundamentals,
        never a future value — the as-of merge must be backward-only."""
        idx = pd.bdate_range("2018-01-02", periods=700)  # spans before AND after all known_from dates
        prices = pd.DataFrame({"Close": np.linspace(100, 200, 700),
                               "Open": np.linspace(100, 200, 700),
                               "High": np.linspace(101, 201, 700),
                               "Low": np.linspace(99, 199, 700),
                               "Volume": 1_000_000}, index=idx)
        merged = add_fundamentals(prices, fundamentals)
        first_known = fundamentals.index.min()
        early = merged.loc[merged.index < first_known]
        assert not early.empty
        assert early["roe"].isna().all()
        # ...and rows after a known_from do carry that year's values (backward as-of)
        later = merged.loc[merged.index > first_known]
        assert later["roe"].notna().any()

    def test_asof_merge_picks_most_recent_known(self, ohlcv, fundamentals):
        merged = add_fundamentals(ohlcv, fundamentals)
        # Row just after the 2023 known_from must carry the 2023 ROE (0.15 + 0.01*2023 = 20.38)
        cutoff = fundamentals.index.max()
        later = merged.loc[merged.index > cutoff]
        assert later["roe"].iloc[0] == pytest.approx(0.15 + 0.01 * 2023)

    def test_pe_ratio_derived_from_close_over_eps(self, ohlcv, fundamentals):
        merged = add_fundamentals(ohlcv, fundamentals)
        later = merged.loc[merged.index > fundamentals.index.max()]
        row = later.iloc[0]
        # eps (12.0 in the fixture) is used to derive pe_ratio and then dropped from the frame
        assert row["pe_ratio"] == pytest.approx(row["Close"] / 12.0)
