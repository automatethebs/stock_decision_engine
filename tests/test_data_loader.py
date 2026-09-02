"""Tests for data_loader.py — universe list fetching and OHLCV download column handling."""
import numpy as np
import pandas as pd
import pytest
import requests

import data_loader


class TestNIFTY50:
    def test_universe_size_and_members(self):
        assert len(data_loader.NIFTY50) == 50
        assert "RELIANCE" in data_loader.NIFTY50
        assert "TCS" in data_loader.NIFTY50

    def test_no_duplicates(self):
        assert len(set(data_loader.NIFTY50)) == len(data_loader.NIFTY50)


class TestFetchNifty500:
    def test_uses_cache_without_network(self, tmp_path, monkeypatch):
        cache = tmp_path / "symbols.txt"
        cache.write_text("RELIANCE\nTCS\n", encoding="utf-8")
        monkeypatch.setattr(data_loader, "NIFTY500_CACHE", str(cache))

        hit_network = {"called": False}
        original_session = requests.Session

        def fake_session(*args, **kwargs):
            hit_network["called"] = True
            return original_session(*args, **kwargs)

        monkeypatch.setattr(requests, "Session", fake_session)
        symbols = data_loader.fetch_nifty500(use_cache=True)
        assert symbols == ["RELIANCE", "TCS"]
        assert hit_network["called"] is False

    def test_parses_live_csv_format(self, tmp_path, monkeypatch):
        # Mirrors the real NSE archive file: header is
        # "Company Name,Industry,Symbol,Series,ISIN Code" — symbol at index 2.
        csv_text = (
            "Company Name,Industry,Symbol,Series,ISIN Code\n"
            "360 ONE WAM Ltd.,Financial Services,360ONE,EQ,INE466L01038\n"
            "3M India Ltd.,Diversified,3MINDIA,EQ,INE470A01017\n"
        )
        monkeypatch.setattr(data_loader, "NIFTY500_CACHE", str(tmp_path / "symbols.txt"))

        class FakeResponse:
            def __init__(self, text=""):
                self.text = text

            def raise_for_status(self):
                return None

        class FakeSession:
            def __init__(self):
                self.headers = {}

            def get(self, url, timeout=None):
                if "ind_nifty500list" in url:
                    return FakeResponse(csv_text)
                return FakeResponse("")  # home page just sets cookies

        monkeypatch.setattr(requests, "Session", FakeSession)
        symbols = data_loader.fetch_nifty500(use_cache=False)
        assert symbols == ["360ONE", "3MINDIA"]
        # cache written for next run
        assert (tmp_path / "symbols.txt").read_text(encoding="utf-8") == "360ONE\n3MINDIA"


class TestDownloadStock:
    def test_flattens_multiindex_columns(self, monkeypatch):
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["RELIANCE.NS"]]
        )
        data = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 5)), columns=cols)
        data.index = pd.bdate_range("2024-01-02", periods=60)

        monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: data)
        out = data_loader.download_stock("RELIANCE.NS")
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert out.index.name == "Date"
        assert len(out) == 60

    def test_raises_on_empty(self, monkeypatch):
        monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: pd.DataFrame())
        with pytest.raises(ValueError):
            data_loader.download_stock("NOTHING.NS")
