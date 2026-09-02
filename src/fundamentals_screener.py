"""Fundamentals sourced from screener.in (free, no API key) instead of yfinance — gives
~10-11 years of annual history per stock vs yfinance's ~4-5 year cap. Scrapes public HTML
tables; row labels vary a bit by sector (banks use Revenue/Deposits, others use Sales/
Borrowings), so columns are matched by flexible label search rather than fixed position.
Output format matches fundamentals.py exactly, so features.py's add_fundamentals() and
predict.py's get_fundamentals_snapshot() work unchanged."""
import io
import os
import re
import time
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
REPORTING_LAG_DAYS = 60
FUNDAMENTAL_COLUMNS = [
    "roe", "debt_to_equity", "profit_margin", "revenue_growth_yoy", "eps",
    "dividend_payout_pct", "book_value_per_share",
]

# Indian companies' annual results always end in March — quarterly tables mix Jun/Sep/Dec/Mar
# columns, so requiring "Mar YYYY" specifically is what actually distinguishes annual from
# quarterly (a looser regex matched quarterly Mar columns too, silently truncating history).
YEAR_COL_RE = re.compile(r"^Mar\s+\d{4}$")


def _clean_label(label) -> str:
    return re.sub(r"[^a-zA-Z ]", "", str(label)).strip().lower()


def _to_float(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _find_annual_table(tables, required_label_substrings):
    """Finds the table whose columns look like a full annual history (many Mar YYYY columns)
    and whose first column contains all the given label substrings somewhere. Requires >=6
    year columns specifically to reject quarterly tables, which can contain a few incidental
    Mar-quarter columns (quarter-end points) without being real annual history."""
    best = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        year_cols = [c for c in cols if YEAR_COL_RE.match(c)]
        if len(year_cols) < 6:
            continue
        labels = [_clean_label(x) for x in t.iloc[:, 0].tolist()]
        if all(any(sub in lab for lab in labels) for sub in required_label_substrings):
            if best is None or len(year_cols) > len(best[1]):
                best = (t, year_cols, labels)
    return best if best else (None, None, None)


def _row_values(t, labels, year_cols, substring):
    idx = next((i for i, lab in enumerate(labels) if substring in lab), None)
    if idx is None:
        return None
    row = t.iloc[idx]
    return {col: _to_float(row[col]) for col in year_cols}


def compute_fundamentals(symbol: str) -> pd.DataFrame:
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code == 404:
        url = f"https://www.screener.in/company/{symbol}/"
        resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))

    pl_table, pl_years, pl_labels = _find_annual_table(tables, ["net profit"])
    bs_table, bs_years, bs_labels = _find_annual_table(tables, ["equity capital", "reserves"])
    if pl_table is None or bs_table is None:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

    sales = _row_values(pl_table, pl_labels, pl_years, "sales") or _row_values(pl_table, pl_labels, pl_years, "revenue")
    net_profit = _row_values(pl_table, pl_labels, pl_years, "net profit")
    eps = _row_values(pl_table, pl_labels, pl_years, "eps")
    dividend_payout = _row_values(pl_table, pl_labels, pl_years, "dividend payout")
    equity_capital = _row_values(bs_table, bs_labels, bs_years, "equity capital")
    reserves = _row_values(bs_table, bs_labels, bs_years, "reserves")
    borrowings = _row_values(bs_table, bs_labels, bs_years, "borrowing") \
        or _row_values(bs_table, bs_labels, bs_years, "deposit")

    common_years = sorted(set(pl_years) & set(bs_years), key=lambda y: int(y.split()[1]))
    rows = []
    prev_sales = None
    for year in common_years:
        eq = (equity_capital.get(year, 0) or 0) + (reserves.get(year, 0) or 0)
        np_ = net_profit.get(year) if net_profit else np.nan
        sales_v = sales.get(year) if sales else np.nan
        if pd.isna(np_) or pd.isna(eq) or eq == 0:
            prev_sales = sales_v
            continue
        year_end = pd.Timestamp(f"{year.split()[1]}-03-31")
        eps_v = eps.get(year) if eps else np.nan
        shares = (np_ / eps_v) if eps_v else np.nan  # derived: net profit / EPS = diluted share count
        rows.append({
            "fiscal_year_end": year_end,
            "known_from": year_end + pd.Timedelta(days=REPORTING_LAG_DAYS),
            "roe": np_ / eq,
            "debt_to_equity": (borrowings.get(year, 0) / eq) if borrowings and pd.notna(borrowings.get(year)) else np.nan,
            "profit_margin": (np_ / sales_v) if sales_v else np.nan,
            "revenue_growth_yoy": ((sales_v - prev_sales) / prev_sales) if prev_sales and sales_v and prev_sales != 0 else np.nan,
            "eps": eps_v,
            "dividend_payout_pct": dividend_payout.get(year) if dividend_payout else np.nan,
            "book_value_per_share": (eq / shares) if shares else np.nan,
        })
        prev_sales = sales_v

    df = pd.DataFrame(rows).sort_values("fiscal_year_end").reset_index(drop=True)
    return df.set_index("known_from")[FUNDAMENTAL_COLUMNS + ["fiscal_year_end"]]


def download_all(stocks):
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    for stock in stocks:
        try:
            df = compute_fundamentals(stock)
        except Exception as e:
            print(f"Skipped {stock}: {e}")
            continue
        if df.empty:
            print(f"Skipped {stock}: no usable fundamentals")
            continue
        out_path = os.path.join(FUNDAMENTALS_DIR, f"{stock}.csv")
        df.to_csv(out_path)
        print(f"Saved {len(df)} fiscal years for {stock} -> {out_path}")
        time.sleep(1.0)  # polite delay — screener.in is a free community site, not a paid API


if __name__ == "__main__":
    from data_loader import fetch_nifty500
    download_all(fetch_nifty500())
