"""Tests for predict.py — thresholds, probability extraction, snapshots and the screener."""
import numpy as np
import pandas as pd
import pytest

import events
import pooled
import predict
from features import FEATURE_COLUMNS


class DummyModel:
    """Fixed-probability model: P(up) = 0.6 for every row."""
    def predict_proba(self, X):
        return np.array([[0.4, 0.6]] * len(X))


class DummyLinear:
    """Bare linear model with coef_ only, to exercise feature_importance()."""
    def __init__(self):
        self.coef_ = np.array([np.arange(1, len(FEATURE_COLUMNS) + 1) * 0.1])


def make_bundle(model=None, model_name="dummy"):
    return {
        "model": model or DummyModel(),
        "model_name": model_name,
        "scaler": None,
        "feature_columns": FEATURE_COLUMNS,
        "horizon": 30,
        "deadband": 0.03,
    }


class TestRecommend:
    @pytest.mark.parametrize("prob,expected", [
        (60.0, "BUY"), (99.9, "BUY"), (59.9, "HOLD"), (50.0, "HOLD"),
        (40.0, "SELL"), (0.0, "SELL"), (40.1, "HOLD"),
    ])
    def test_thresholds(self, prob, expected):
        assert predict.recommend(prob) == expected


class TestRiskLevel:
    @pytest.mark.parametrize("vol,expected", [
        (0.010, "Low"), (0.0149, "Low"), (0.015, "Medium"),
        (0.0299, "Medium"), (0.03, "High"), (0.05, "High"),
    ])
    def test_thresholds(self, vol, expected):
        assert predict.risk_level(vol) == expected


class TestFeatureImportance:
    def test_uses_coef_abs_values(self):
        bundle = make_bundle(model=DummyLinear(), model_name="logistic_regression")
        imp = predict.feature_importance(bundle)
        assert isinstance(imp, pd.Series)
        assert len(imp) == len(FEATURE_COLUMNS)
        # sorted descending by |coef|
        assert imp.is_monotonic_decreasing
        assert imp.index[0] == FEATURE_COLUMNS[-1]  # largest coef column first


class TestPredict:
    def test_predict_end_to_end(self, tmp_path, ohlcv, monkeypatch):
        ohlcv.to_csv(tmp_path / "RELIANCE.csv", index_label="Date")
        monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
        result = predict.predict("RELIANCE", bundle=make_bundle())
        assert result["stock"] == "RELIANCE"
        assert result["positive_probability"] == 60.0
        assert result["negative_probability"] == 40.0
        assert result["horizon_days"] == 30
        assert result["risk_level"] in {"Low", "Medium", "High"}
        assert result["as_of"] <= ohlcv.index[-1].date().isoformat()

    def test_predict_missing_stock_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError):
            predict.predict("NOPE", bundle=make_bundle())

    def test_predict_too_short_history_raises(self, tmp_path, monkeypatch):
        short = pd.DataFrame({"Open": [1.0] * 15, "High": [1.0] * 15, "Low": [1.0] * 15,
                              "Close": [1.0] * 15, "Volume": [1000] * 15},
                             index=pd.bdate_range("2023-01-02", periods=15))
        short.to_csv(tmp_path / "SHORT.csv", index_label="Date")
        monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
        with pytest.raises(ValueError):
            predict.predict("SHORT", bundle=make_bundle())


def _point_predict_all_at(tmp_path, monkeypatch, *csvs):
    """Points both predict.DATA_DIR (per-stock predict) and pooled.DATA_DIR/CACHE_PATH
    (vectorized predict_all) at one temp dir containing the given CSVs."""
    monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pooled, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pooled, "CACHE_PATH", str(tmp_path / "pooled.joblib"))
    for name, df in csvs:
        df.to_csv(tmp_path / f"{name}.csv", index_label="Date")


def _short_frame(n):
    return pd.DataFrame({"Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n,
                         "Close": [1.0] * n, "Volume": [1000] * n},
                        index=pd.bdate_range("2023-01-02", periods=n))


class TestPredictAll:
    def test_skips_failing_stocks(self, tmp_path, ohlcv, monkeypatch):
        _point_predict_all_at(tmp_path, monkeypatch, ("GOOD", ohlcv), ("SHORT", _short_frame(10)))
        df = predict.predict_all(["GOOD", "SHORT", "MISSING"], bundle=make_bundle())
        assert list(df["stock"]) == ["GOOD"]
        assert df.loc[0, "recommendation"] == "BUY"
        assert df.loc[0, "positive_probability"] == 60.0

    def test_empty_input_returns_empty_frame(self, tmp_path, monkeypatch):
        _point_predict_all_at(tmp_path, monkeypatch)
        df = predict.predict_all([], bundle=make_bundle())
        assert df.empty

    def test_vectorized_matches_per_stock_reference(self, tmp_path, ohlcv, monkeypatch):
        """The new single-pass predict_all must produce byte-identical output to the
        old per-stock predict() loop — same rows, same order, same values."""
        _point_predict_all_at(
            tmp_path, monkeypatch,
            ("GOOD", ohlcv), ("MID", ohlcv.iloc[:120]), ("SHORT", _short_frame(10)),
        )
        bundle = make_bundle()

        vectorized = predict.predict_all(["GOOD", "MID", "SHORT"], bundle=bundle)

        # Reference: the legacy implementation, kept verbatim for comparison.
        rows = []
        for stock in ["GOOD", "MID", "SHORT"]:
            try:
                r = predict.predict(stock, bundle=bundle)
            except (FileNotFoundError, ValueError):
                continue
            rows.append({
                "stock": r["stock"],
                "recommendation": predict.recommend(r["positive_probability"]),
                "positive_probability": r["positive_probability"],
                "negative_probability": r["negative_probability"],
                "risk_level": r["risk_level"],
                "as_of": r["as_of"],
            })
        reference = (pd.DataFrame(rows)
                     .sort_values("positive_probability", ascending=False)
                     .reset_index(drop=True))

        pd.testing.assert_frame_equal(vectorized, reference)
        assert list(vectorized["stock"]) == ["GOOD", "MID"]  # SHORT skipped


class TestFundamentalsSnapshot:
    def test_reads_latest_row(self, tmp_path, monkeypatch):
        fund_dir = tmp_path / "fundamentals"
        fund_dir.mkdir()
        rows = [
            {"known_from": "2023-05-30", "fiscal_year_end": "2023-03-31",
             "roe": 0.15, "debt_to_equity": 0.5, "profit_margin": 0.10,
             "revenue_growth_yoy": 0.12},
            {"known_from": "2024-05-30", "fiscal_year_end": "2024-03-31",
             "roe": 0.18, "debt_to_equity": 0.45, "profit_margin": 0.12,
             "revenue_growth_yoy": 0.15},
        ]
        pd.DataFrame(rows).set_index("known_from").to_csv(fund_dir / "XYZ.csv")
        monkeypatch.setattr(predict, "FUNDAMENTALS_DIR", str(fund_dir))

        snap = predict.get_fundamentals_snapshot("XYZ")
        assert snap["fiscal_year_end"] == "2024-03-31"
        assert snap["roe_pct"] == 18.0
        assert snap["debt_to_equity"] == 0.45
        assert snap["profit_margin_pct"] == 12.0
        assert snap["revenue_growth_yoy_pct"] == 15.0

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict, "FUNDAMENTALS_DIR", str(tmp_path / "nope"))
        assert predict.get_fundamentals_snapshot("ANY") is None


class TestEventsSnapshot:
    def test_days_since_from_event_files(self, tmp_path, ohlcv, monkeypatch):
        ohlcv.to_csv(tmp_path / "ABC.csv", index_label="Date")
        events_dir = tmp_path / "events"
        events_dir.mkdir()
        last_day = ohlcv.index[-1].date().isoformat()
        pd.DataFrame({"dividend_date": [last_day]}).to_csv(events_dir / "ABC_dividends.csv", index=False)
        pd.DataFrame({"split_date": []}).to_csv(events_dir / "ABC_splits.csv", index=False)
        monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(events, "EVENTS_DIR", str(events_dir))

        snap = predict.get_events_snapshot("ABC")
        assert snap["days_since_dividend"] == 0
        assert snap["days_since_split"] is None

    def test_no_event_files_returns_none(self, tmp_path, ohlcv, monkeypatch):
        ohlcv.to_csv(tmp_path / "DEF.csv", index_label="Date")
        monkeypatch.setattr(predict, "DATA_DIR", str(tmp_path))
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path / "empty_events"))
        snap = predict.get_events_snapshot("DEF")
        assert snap == {"days_since_dividend": None, "days_since_split": None}
