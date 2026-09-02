"""Value & quality screener — deliberately NOT a prediction. Ranks stocks by objective,
already-known fundamentals (cheapness, profitability, balance-sheet health, growth, and
multi-year durability) instead of forecasting future price direction. This is the classic
Graham/Buffett/Greenblatt-style value+quality approach: no forecasting involved, just
comparing companies on facts that are true right now. Complements (does not replace) the
ML probability signal elsewhere in the app.

Durability/trend analysis exists specifically to catch "value traps" — a stock that looks
cheap on a single year's snapshot but whose business is in structural decline — and its
mirror case, a "turnaround": a business that WAS declining but has recently re-accelerated.
Both are read off the multi-year trend already sitting in the 10-12yr fundamentals history,
not from news (news is too short-term/noisy for a structural, multi-year question — see
README)."""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FUNDAMENTALS_DIR = os.path.join(DATA_DIR, "fundamentals")

# Each metric: whether higher is better, and its weight in the composite score.
# Equal-weighted across five factors: value, quality, safety, growth, durability.
METRICS = {
    "earnings_yield": {"higher_is_better": True, "weight": 1.0},    # value: cheapness (1/PE)
    "roe": {"higher_is_better": True, "weight": 1.0},                # quality: profitability
    "debt_to_equity": {"higher_is_better": False, "weight": 1.0},    # safety: low leverage
    "revenue_growth_yoy": {"higher_is_better": True, "weight": 1.0},  # growth: latest year
    "durability_raw": {"higher_is_better": True, "weight": 1.0},     # multi-year trend health
}

MIN_YEARS_FOR_TREND = 4


def compute_trend(fund: pd.DataFrame) -> dict:
    """Splits available history into an 'early' period and a 'recent' (last ~2yr) period,
    compares them. Catches value traps (was growing, now stalling/declining) and turnarounds
    (was weak, now re-accelerating) using nothing but the multi-year fundamentals already on
    disk — no news, no forecasting, just arithmetic on real historical numbers."""
    growth = fund["revenue_growth_yoy"].dropna()
    margin = fund["profit_margin"].dropna()

    if len(growth) < MIN_YEARS_FOR_TREND:
        return {"revenue_cagr": None, "margin_trend": None, "durability_raw": None, "business_trend": "Not enough history"}

    recent_n = max(1, min(2, len(growth) // 3))
    recent_growth = growth.iloc[-recent_n:].mean()
    early_growth = growth.iloc[:-recent_n].mean()
    revenue_cagr = (1 + growth).prod() ** (1 / len(growth)) - 1

    margin_trend = None
    if len(margin) >= MIN_YEARS_FOR_TREND:
        m_recent_n = max(1, min(2, len(margin) // 3))
        margin_trend = margin.iloc[-m_recent_n:].mean() - margin.iloc[:-m_recent_n].mean()

    growth_improvement = recent_growth - early_growth
    if early_growth < 0.05 and recent_growth > 0.08 and growth_improvement > 0.08:
        label = "Turnaround"
    elif early_growth > 0.05 and (recent_growth < 0.02 or growth_improvement < -0.08):
        label = "Declining"
    elif revenue_cagr > 0.08 and (margin_trend is None or margin_trend >= -0.01):
        label = "Compounder"
    else:
        label = "Stable"

    durability_raw = 0.6 * revenue_cagr + 0.4 * (margin_trend if margin_trend is not None else 0)

    return {
        "revenue_cagr": revenue_cagr, "margin_trend": margin_trend,
        "durability_raw": durability_raw, "business_trend": label,
    }


def compute_value_table(stocks) -> pd.DataFrame:
    rows = []
    for stock in stocks:
        fund_path = os.path.join(FUNDAMENTALS_DIR, f"{stock}.csv")
        price_path = os.path.join(DATA_DIR, f"{stock}.csv")
        if not os.path.exists(fund_path) or not os.path.exists(price_path):
            continue
        fund = pd.read_csv(fund_path, index_col="known_from", parse_dates=True).sort_index()
        if fund.empty:
            continue
        latest = fund.iloc[-1]
        price = pd.read_csv(price_path, index_col="Date", parse_dates=True)["Close"].iloc[-1]

        eps = latest.get("eps")
        pe_ratio = (price / eps) if pd.notna(eps) and eps else None
        earnings_yield = (1 / pe_ratio) if pe_ratio and pe_ratio > 0 else None
        trend = compute_trend(fund)

        rows.append({
            "stock": stock,
            "price": price,
            "pe_ratio": pe_ratio,
            "earnings_yield": earnings_yield,
            "roe": latest.get("roe"),
            "debt_to_equity": latest.get("debt_to_equity"),
            "profit_margin": latest.get("profit_margin"),
            "revenue_growth_yoy": latest.get("revenue_growth_yoy"),
            "dividend_payout_pct": latest.get("dividend_payout_pct"),
            "fiscal_year_end": pd.to_datetime(latest.get("fiscal_year_end")).date().isoformat()
                if pd.notna(latest.get("fiscal_year_end")) else None,
            "years_of_history": len(fund),
            **trend,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Composite score: average percentile rank across five factors (0-100, higher = better
    # value+quality+durability). Percentile rank rather than raw z-score so one extreme
    # outlier stock can't dominate the score. Stocks with no durability reading (too little
    # history) get a neutral 50th-percentile durability contribution rather than being
    # penalized for a data gap that isn't their fault.
    score_components = []
    for metric, cfg in METRICS.items():
        pct_rank = df[metric].rank(pct=True, ascending=cfg["higher_is_better"])
        if metric == "durability_raw":
            pct_rank = pct_rank.fillna(0.5)
        score_components.append(pct_rank * cfg["weight"])
    total_weight = sum(cfg["weight"] for cfg in METRICS.values())
    df["value_score"] = (sum(score_components) / total_weight * 100).round(1)

    return df.sort_values("value_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_loader import fetch_nifty500
    table = compute_value_table(fetch_nifty500())
    pd.set_option("display.width", 180)
    print(table[["stock", "value_score", "pe_ratio", "roe", "revenue_growth_yoy", "revenue_cagr", "business_trend"]].head(15))
    print("\nTurnaround candidates:")
    print(table[table["business_trend"] == "Turnaround"][["stock", "value_score", "revenue_growth_yoy", "revenue_cagr"]])
    print("\nDeclining flags:")
    print(table[table["business_trend"] == "Declining"][["stock", "value_score", "revenue_growth_yoy", "revenue_cagr"]])
