"""Downloads annual fundamentals per stock and saves one CSV per stock, indexed by the
date the fundamentals become effectively "known" (fiscal year-end + reporting lag), so
merging them onto daily price data later never leaks future information."""
import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")

# Indian companies must report annual results within 60 days of fiscal year-end (SEBI);
# using that as the lag is conservative (keeps us safely on the "already public" side).
REPORTING_LAG_DAYS = 60

FUNDAMENTAL_COLUMNS = ["roe", "debt_to_equity", "profit_margin", "revenue_growth_yoy", "eps"]


def compute_fundamentals(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    financials = t.financials
    balance_sheet = t.balance_sheet
    if financials.empty or balance_sheet.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

    years = sorted(financials.columns)
    rows = []
    for year_end in years:
        try:
            net_income = financials.loc["Net Income", year_end]
            revenue = financials.loc["Total Revenue", year_end]
            diluted_shares = financials.loc["Diluted Average Shares", year_end]
            equity = balance_sheet.loc["Stockholders Equity", year_end]
            debt = balance_sheet.loc["Total Debt", year_end]
        except KeyError:
            continue
        if pd.isna(net_income) or pd.isna(revenue) or pd.isna(equity) or equity == 0:
            continue
        rows.append({
            "fiscal_year_end": year_end,
            "known_from": year_end + pd.Timedelta(days=REPORTING_LAG_DAYS),
            "revenue": revenue,
            "roe": net_income / equity,
            "debt_to_equity": (debt / equity) if pd.notna(debt) else np.nan,
            "profit_margin": net_income / revenue if revenue else np.nan,
            "eps": net_income / diluted_shares if pd.notna(diluted_shares) and diluted_shares else np.nan,
        })

    df = pd.DataFrame(rows).sort_values("fiscal_year_end").reset_index(drop=True)
    df["revenue_growth_yoy"] = df["revenue"].pct_change()
    return df.set_index("known_from")[FUNDAMENTAL_COLUMNS + ["fiscal_year_end"]]


def download_all(stocks):
    os.makedirs(FUNDAMENTALS_DIR, exist_ok=True)
    for stock in stocks:
        ticker = f"{stock}.NS"
        try:
            df = compute_fundamentals(ticker)
        except Exception as e:
            print(f"Skipped {stock}: {e}")
            continue
        if df.empty:
            print(f"Skipped {stock}: no usable fundamentals")
            continue
        out_path = os.path.join(FUNDAMENTALS_DIR, f"{stock}.csv")
        df.to_csv(out_path)
        print(f"Saved {len(df)} fiscal years for {stock} -> {out_path}")
        time.sleep(0.3)  # be polite to the endpoint across 49 sequential requests


if __name__ == "__main__":
    from data_loader import NIFTY50
    download_all(NIFTY50)
