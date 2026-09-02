"""Tests for backtest.py — drawdown math and the per-stock backtest loop."""
import numpy as np
import pandas as pd
import pytest

from backtest import max_drawdown, backtest_stock
from features import build_dataset


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        cum = pd.Series([1.0, 1.2, 0.9, 1.1])
        assert max_drawdown(cum) == pytest.approx(-0.25)

    def test_no_drawdown(self):
        cum = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(cum) == pytest.approx(0.0)

    def test_trough_then_recovery(self):
        cum = pd.Series([1.0, 2.0, 1.0, 3.0])
        # peak 2.0 -> trough 1.0 = -50%
        assert max_drawdown(cum) == pytest.approx(-0.5)


class OracleModel:
    """Predicts exactly the label that was stashed into return_1d, so accuracy = 100%."""
    def predict(self, X):
        return X["return_1d"].astype(int).values


class TestBacktestStock:
    @pytest.fixture
    def stock_df(self, ohlcv):
        df = build_dataset(ohlcv, horizon=5, deadband=0.01)
        df["return_1d"] = df["target"]  # oracle reads the label from a feature column
        return df

    def test_oracle_model_scores_perfectly(self, stock_df):
        metrics = backtest_stock(stock_df, OracleModel(), "oracle", None, horizon=5)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["edge_over_baseline"] == pytest.approx(1.0 - metrics["baseline"])
        assert metrics["n_periods"] == len(stock_df)
        assert metrics["n_nonoverlapping_periods"] == pytest.approx(np.ceil(len(stock_df) / 5))

    def test_oracle_win_rate_and_returns(self, stock_df):
        metrics = backtest_stock(stock_df, OracleModel(), "oracle", None, horizon=5)
        assert metrics["win_rate"] == pytest.approx(1.0)  # every flagged move was up
        assert metrics["avg_strategy_return"] > 0
        assert metrics["strategy_max_drawdown"] <= 0.0
