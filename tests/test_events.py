"""Tests for events.py — the as-of dividend/split proximity features."""
import os

import numpy as np
import pandas as pd
import pytest

import events


class TestAddEvents:
    def test_no_event_files_means_neg_one(self, ohlcv, tmp_path, monkeypatch):
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path / "missing"))
        out = events.add_events(ohlcv, "X")
        assert (out["days_since_dividend"] == -1).all()
        assert (out["days_since_split"] == -1).all()

    def test_dividend_on_last_day_gives_zero(self, ohlcv, tmp_path, monkeypatch):
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path))
        last_day = ohlcv.index[-1].date().isoformat()
        pd.DataFrame({"dividend_date": [last_day]}).to_csv(
            tmp_path / "X_dividends.csv", index=False
        )
        out = events.add_events(ohlcv, "X")
        assert out["days_since_dividend"].iloc[-1] == 0

    def test_days_since_counts_calendar_days(self, ohlcv, tmp_path, monkeypatch):
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path))
        target = ohlcv.index[-10]
        pd.DataFrame({"dividend_date": [target.date().isoformat()]}).to_csv(
            tmp_path / "X_dividends.csv", index=False
        )
        out = events.add_events(ohlcv, "X")
        assert out["days_since_dividend"].iloc[-1] == (ohlcv.index[-1] - target).days

    def test_capped_at_365(self, ohlcv, tmp_path, monkeypatch):
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path))
        old_date = (ohlcv.index[-1] - pd.Timedelta(days=800)).date().isoformat()
        pd.DataFrame({"dividend_date": [old_date]}).to_csv(
            tmp_path / "X_dividends.csv", index=False
        )
        out = events.add_events(ohlcv, "X")
        assert out["days_since_dividend"].iloc[-1] == 365

    def test_no_lookahead(self, ohlcv, tmp_path, monkeypatch):
        """An event dated AFTER a price row must not be visible to that row."""
        monkeypatch.setattr(events, "EVENTS_DIR", str(tmp_path))
        future_date = (ohlcv.index[100] + pd.Timedelta(days=30)).date().isoformat()
        pd.DataFrame({"dividend_date": [future_date]}).to_csv(
            tmp_path / "X_dividends.csv", index=False
        )
        out = events.add_events(ohlcv, "X")
        before_event = out.iloc[:100]
        assert (before_event["days_since_dividend"] == -1).all()
