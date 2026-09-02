"""Tests for value_screener.py — the non-predictive value/quality/durability ranking."""
import numpy as np
import pandas as pd
import pytest

import value_screener


class TestComputeTrend:
    def test_turnaround(self):
        fund = pd.DataFrame({
            "revenue_growth_yoy": [0.02, 0.03, -0.02, 0.05, 0.15, 0.20],
            "profit_margin": [0.05, 0.06, 0.06, 0.07, 0.08, 0.09],
        })
        out = value_screener.compute_trend(fund)
        assert out["business_trend"] == "Turnaround"

    def test_declining(self):
        fund = pd.DataFrame({
            "revenue_growth_yoy": [0.20, 0.18, 0.15, 0.10, 0.01, -0.05],
            "profit_margin": [0.15, 0.14, 0.13, 0.12, 0.11, 0.10],
        })
        out = value_screener.compute_trend(fund)
        assert out["business_trend"] == "Declining"

    def test_compounder(self):
        fund = pd.DataFrame({
            "revenue_growth_yoy": [0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
            "profit_margin": [0.10, 0.11, 0.11, 0.12, 0.12, 0.13],
        })
        out = value_screener.compute_trend(fund)
        assert out["business_trend"] == "Compounder"
        assert out["revenue_cagr"] > 0.08

    def test_stable_when_margin_deteriorating(self):
        # growth looks compounder-ish but margins are falling -> not a Compounder
        fund = pd.DataFrame({
            "revenue_growth_yoy": [0.10, 0.09, 0.11, 0.10, 0.11, 0.10],
            "profit_margin": [0.15, 0.14, 0.12, 0.10, 0.08, 0.07],
        })
        out = value_screener.compute_trend(fund)
        assert out["business_trend"] == "Stable"

    def test_not_enough_history(self):
        fund = pd.DataFrame({
            "revenue_growth_yoy": [0.10, 0.10, 0.10],
            "profit_margin": [0.10, 0.10, 0.10],
        })
        out = value_screener.compute_trend(fund)
        assert out["business_trend"] == "Not enough history"
        assert out["durability_raw"] is None


def _write_stock(tmp_path, stock, price, eps, roe, de, growth, margin, years=5):
    """Writes price + fundamentals CSVs for one stock into tmp_path, returns None."""
    fund_dir = tmp_path / "fundamentals"
    fund_dir.mkdir(exist_ok=True)
    price_path = tmp_path / f"{stock}.csv"
    pd.DataFrame({"Close": [price] * 3},
                 index=pd.bdate_range("2024-01-02", periods=3)).to_csv(price_path, index_label="Date")

    rows = []
    for i in range(years):
        year = 2019 + i
        rows.append({
            "known_from": f"{year}-05-30",
            "fiscal_year_end": f"{year}-03-31",
            "roe": roe, "debt_to_equity": de, "profit_margin": margin,
            "revenue_growth_yoy": growth, "eps": eps,
        })
    pd.DataFrame(rows).set_index("known_from").to_csv(fund_dir / f"{stock}.csv")


class TestComputeValueTable:
    def test_ranks_better_fundamentals_higher(self, tmp_path, monkeypatch):
        monkeypatch.setattr(value_screener, "FUNDAMENTALS_DIR", str(tmp_path / "fundamentals"))
        monkeypatch.setattr(value_screener, "DATA_DIR", str(tmp_path))
        _write_stock(tmp_path, "GOODCO", price=100, eps=10, roe=0.25, de=0.1, growth=0.20, margin=0.20)
        _write_stock(tmp_path, "BADCO", price=200, eps=10, roe=0.05, de=1.5, growth=0.02, margin=0.03)

        df = value_screener.compute_value_table(["GOODCO", "BADCO"])
        assert list(df["stock"]) == ["GOODCO", "BADCO"]  # sorted by value_score desc
        assert df.loc[0, "value_score"] > df.loc[1, "value_score"]

    def test_pe_ratio_from_price_over_eps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(value_screener, "FUNDAMENTALS_DIR", str(tmp_path / "fundamentals"))
        monkeypatch.setattr(value_screener, "DATA_DIR", str(tmp_path))
        _write_stock(tmp_path, "PEY", price=150, eps=10, roe=0.1, de=0.5, growth=0.1, margin=0.1)
        df = value_screener.compute_value_table(["PEY"])
        assert df.loc[0, "pe_ratio"] == pytest.approx(150 / 10)
        assert df.loc[0, "earnings_yield"] == pytest.approx(10 / 150)
        assert df.loc[0, "years_of_history"] == 5

    def test_missing_stock_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(value_screener, "FUNDAMENTALS_DIR", str(tmp_path / "fundamentals"))
        monkeypatch.setattr(value_screener, "DATA_DIR", str(tmp_path))
        df = value_screener.compute_value_table(["NOTHERE"])
        assert df.empty
