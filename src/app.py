"""Local Streamlit UI for the decision engine. Run: streamlit run src/app.py"""
import os
import glob
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from predict import (
    predict, predict_all, load_model,
    get_fundamentals_snapshot, get_events_snapshot, get_news_context,
)
from features import add_features
from backtest import run_backtest
from value_screener import compute_value_table

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")

# --- Design tokens -----------------------------------------------------------------
INK_950, INK_850, INK_700 = "#10141B", "#171D26", "#2A323F"
TEXT, MUTED = "#ECEEF1", "#8B93A0"
ACCENT = "#C08A2E"
POSITIVE, NEGATIVE = "#1FA971", "#E15252"
RISK_COLOR = {"Low": POSITIVE, "Medium": ACCENT, "High": NEGATIVE}
SIGNAL_COLOR = {"BUY": POSITIVE, "HOLD": ACCENT, "SELL": NEGATIVE}
CONFIDENCE_COLOR = {"Low": MUTED, "Medium": ACCENT, "High": POSITIVE}

FEATURE_LABELS = {
    "return_1d": "1-day price change",
    "dist_sma_5": "Distance from 5-day average price",
    "dist_sma_10": "Distance from 10-day average price",
    "dist_sma_20": "Distance from 20-day average price",
    "dist_sma_50": "Distance from 50-day average price",
    "macd": "MACD (trend strength)",
    "macd_signal": "MACD signal line",
    "rsi_14": "RSI, 14-day (overbought/oversold)",
    "bb_pctb": "Position within Bollinger Bands",
    "volatility_20": "20-day volatility",
    "volume_change": "Daily volume change",
    "momentum_5": "5-day momentum",
    "momentum_10": "10-day momentum",
    "momentum_20": "20-day momentum",
}


def recommendation(pos: float):
    if pos >= 60:
        return "BUY"
    if pos <= 40:
        return "SELL"
    return "HOLD"


def confidence_from_edge(edge):
    """Honest confidence label — derived from THIS stock's actual backtested edge over
    baseline, not from how far the probability sits from 50%. A stock can show a strong
    probability and still have Low confidence if its track record shows no real edge."""
    if edge is None or pd.isna(edge):
        return "Unrated", "Not enough held-out history to backtest this stock yet."
    if edge > 0.03:
        return "High", f"This stock's signal beat its own historical baseline by {edge*100:+.1f}pp in backtesting — the strongest tier available, though still modest in absolute terms."
    if edge > 0:
        return "Medium", f"This stock's signal beat its own historical baseline by {edge*100:+.1f}pp in backtesting — a small, real edge."
    return "Low", f"This stock's signal did NOT beat its own historical baseline ({edge*100:+.1f}pp) — treat this signal as roughly coin-flip-level for this stock."


st.set_page_config(page_title="Stock Decision Engine", layout="wide", page_icon="\U0001F4C8")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background: {INK_950}; }}

    h1, h2, h3, .display-num {{ font-family: 'Fraunces', serif; text-wrap: balance; }}
    .mono, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {{
        font-family: 'IBM Plex Mono', monospace !important; font-variant-numeric: tabular-nums;
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em;
        text-transform: uppercase; color: {MUTED}; margin-bottom: 4px;
    }}

    .card {{
        background: {INK_850}; border: 1px solid {INK_700}; border-radius: 12px;
        padding: 20px 22px; height: 100%;
    }}

    .pill {{
        display: inline-flex; align-items: center; gap: 6px; padding: 6px 16px;
        border-radius: 999px; font-weight: 600; font-size: 14px; color: {INK_950};
    }}
    .pill-outline {{
        display: inline-flex; align-items: center; gap: 6px; padding: 5px 14px;
        border-radius: 999px; font-weight: 600; font-size: 13px; border: 1.5px solid;
        background: transparent;
    }}

    .signal-hero {{
        font-family: 'Fraunces', serif; font-weight: 700; font-size: 40px; letter-spacing: 0.02em;
    }}

    .headline-num {{
        font-family: 'Fraunces', serif; font-weight: 600; font-size: 56px; line-height: 1;
    }}

    .news-item {{
        padding: 8px 0; border-bottom: 1px solid {INK_700}; font-size: 14px;
    }}
    .news-item:last-child {{ border-bottom: none; }}

    .disclaimer {{
        background: {INK_850}; border: 1px solid {INK_700}; border-left: 3px solid {ACCENT};
        border-radius: 8px; padding: 14px 18px; font-size: 13px; color: {MUTED}; line-height: 1.6;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

available_stocks = sorted(
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))
)

with st.sidebar:
    st.markdown("<div class='eyebrow'>Local prototype · traditional ML · no LLM</div>", unsafe_allow_html=True)
    st.title("Stock Decision Engine")
    if not available_stocks:
        st.error("No data found. Run `python src/data_loader.py` first.")
        st.stop()
    stock = st.selectbox("Stock", available_stocks)
    chart_days = st.slider("Chart window (days)", 60, 500, 250, step=10)
    st.divider()
    st.caption(
        "**How to read this:** the probability is how often, historically, similar "
        "technical setups preceded a >1% move over 30 days. Confidence is separate — "
        "it reflects whether *this specific stock's* signal has actually beaten guessing "
        "in backtesting, not just how extreme today's probability looks."
    )

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

bundle = load_model()

tab_detail, tab_screener, tab_value = st.tabs(["Stock Detail", "Screener — all stocks", "Value Screener"])

# ============================================================== VALUE SCREENER =====
with tab_value:
    st.markdown("<div class='eyebrow'>Not a prediction — fundamentals as they stand today</div>", unsafe_allow_html=True)
    st.subheader("Value & quality ranking")
    st.caption(
        "Ranks stocks by objective, already-known fundamentals — cheapness (earnings yield), "
        "profitability (ROE), low leverage (debt/equity), growth (revenue YoY), and a "
        "multi-year **durability** trend — the classic value + quality approach. This does NOT "
        "forecast future price direction like the ML signal in the other tabs; it's a "
        "comparison of facts that are true right now."
    )
    st.caption(
        "Durability compares each stock's recent (~2yr) revenue/margin trend against its "
        "earlier history — catching **value traps** (cheap but structurally declining) and "
        "**turnarounds** (was weak, now re-accelerating). This uses the 10-12yr fundamentals "
        "history already on disk, not news — news is too short-term and noisy for a "
        "multi-year structural question, and can't be backtested at all (see README)."
    )

    @st.cache_data(show_spinner="Ranking stocks by fundamentals...")
    def get_value_table(stocks):
        return compute_value_table(stocks)

    value_df = get_value_table(available_stocks)
    if value_df.empty:
        st.info("No fundamentals data available. Run `python src/fundamentals_screener.py` first.")
    else:
        TREND_COLOR = {"Compounder": POSITIVE, "Turnaround": ACCENT, "Stable": MUTED, "Declining": NEGATIVE}

        top3 = value_df.head(3)
        cols = st.columns(3)
        for col, (_, row) in zip(cols, top3.iterrows()):
            trend = row.get("business_trend", "Stable")
            trend_color = TREND_COLOR.get(trend, MUTED)
            with col:
                st.markdown(
                    f"<div class='card'>"
                    f"<div class='eyebrow'>Value score {row['value_score']:.0f}/100</div>"
                    f"<div style='font-family:Fraunces,serif; font-size:26px; font-weight:600;'>{row['stock']}</div>"
                    f"<span class='pill-outline' style='color:{trend_color}; border-color:{trend_color}; margin-top:8px;'>{trend}</span>"
                    f"<div class='mono' style='margin-top:10px; color:{MUTED}; font-size:13px;'>"
                    f"P/E {row['pe_ratio']:.1f} &middot; ROE {row['roe']*100:.1f}% &middot; D/E {row['debt_to_equity']:.2f}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        turnarounds = value_df[value_df["business_trend"] == "Turnaround"]
        if not turnarounds.empty:
            names = ", ".join(turnarounds["stock"].tolist())
            st.markdown(
                f"<div class='disclaimer' style='border-left-color:{ACCENT};'>"
                f"🔄 <strong>Turnaround candidates</strong> — recent growth meaningfully better than "
                f"their own earlier trend: {names}</div>",
                unsafe_allow_html=True,
            )

        st.write("")
        display_df = value_df.copy()
        display_df["roe"] = display_df["roe"] * 100
        display_df["profit_margin"] = display_df["profit_margin"] * 100
        display_df["revenue_growth_yoy"] = display_df["revenue_growth_yoy"] * 100
        display_df["revenue_cagr"] = display_df["revenue_cagr"] * 100
        st.dataframe(
            display_df.rename(columns={
                "stock": "Stock", "value_score": "Value Score", "pe_ratio": "P/E",
                "roe": "ROE", "debt_to_equity": "Debt/Equity", "profit_margin": "Margin",
                "revenue_growth_yoy": "Rev Growth", "dividend_payout_pct": "Payout %",
                "price": "Price", "fiscal_year_end": "FY End", "revenue_cagr": "Rev CAGR",
                "business_trend": "Trend",
            })[["Stock", "Value Score", "Trend", "Price", "P/E", "ROE", "Debt/Equity",
                "Margin", "Rev Growth", "Rev CAGR", "Payout %", "FY End"]],
            width="stretch",
            hide_index=True,
            column_config={
                "Value Score": st.column_config.ProgressColumn("Value Score", min_value=0, max_value=100, format="%.0f"),
                "ROE": st.column_config.NumberColumn("ROE", format="%.1f%%"),
                "Debt/Equity": st.column_config.NumberColumn("Debt/Equity", format="%.2f"),
                "Margin": st.column_config.NumberColumn("Margin", format="%.1f%%"),
                "Rev Growth": st.column_config.NumberColumn("Rev Growth", format="%.1f%%"),
                "Rev CAGR": st.column_config.NumberColumn("Rev CAGR", format="%.1f%%", help="Multi-year compound annual growth rate"),
                "Payout %": st.column_config.NumberColumn("Payout %", format="%.0f%%"),
                "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
                "Price": st.column_config.NumberColumn("Price", format="₹%.2f"),
            },
        )
        st.caption(
            "Value Score = average percentile rank across earnings yield, ROE, debt/equity "
            "(inverted — lower is better), revenue growth, and durability (multi-year trend), "
            "scaled 0-100. Trend: Compounder = steady growth; Turnaround = recent "
            "re-acceleration after a weak stretch; Declining = was growing, now stalling; "
            "Stable = neither strongly improving nor declining."
        )

# ============================================================== SCREENER =====
with tab_screener:
    st.markdown("<div class='eyebrow'>Universe scan</div>", unsafe_allow_html=True)
    st.subheader(f"{len(available_stocks)} Nifty 500 stocks ranked by signal")
    st.caption("Same BUY (≥60%) / HOLD / SELL (≤40%) thresholds as the detail view. Click a column to sort.")

    @st.cache_data(show_spinner="Scanning all stocks...")
    def get_screener_table(_bundle, stocks):
        return predict_all(stocks, bundle=_bundle)

    screener_df = get_screener_table(bundle, available_stocks)
    st.dataframe(
        screener_df.rename(columns={
            "stock": "Stock", "recommendation": "Signal",
            "positive_probability": "Chance UP %", "negative_probability": "Chance DOWN %",
            "risk_level": "Risk", "as_of": "As of",
        }),
        width="stretch",
        hide_index=True,
        column_config={
            "Chance UP %": st.column_config.ProgressColumn(
                "Chance UP %", min_value=0, max_value=100, format="%.1f%%",
            ),
        },
    )
    buy_count = (screener_df["recommendation"] == "BUY").sum()
    sell_count = (screener_df["recommendation"] == "SELL").sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("BUY signals", buy_count)
    m2.metric("SELL signals", sell_count)
    m3.metric("HOLD", len(screener_df) - buy_count - sell_count)

# ============================================================== DETAIL =====
with tab_detail:
    result = predict(stock, bundle=bundle)

    raw = pd.read_csv(os.path.join(DATA_DIR, f"{stock}.csv"), index_col="Date", parse_dates=True)
    featured = add_features(raw)
    last_close = featured["Close"].iloc[-1]
    prev_close = featured["Close"].iloc[-2]
    day_change_pct = (last_close / prev_close - 1) * 100

    pos, neg = result["positive_probability"], result["negative_probability"]
    risk = result["risk_level"]
    signal = recommendation(pos)

    @st.cache_data(show_spinner="Running backtest...")
    def get_backtest_results(_bundle):
        return run_backtest()

    bt = get_backtest_results(bundle)
    edge = bt.loc[stock, "edge_over_baseline"] if stock in bt.index else None
    conf_label, conf_explainer = confidence_from_edge(edge)

    # --- Hero row --------------------------------------------------------------
    hero_left, hero_right = st.columns([2.2, 1])
    with hero_left:
        st.markdown(f"<div class='eyebrow'>{result['horizon_days']}-day outlook · as of {result['as_of']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='signal-hero'>{stock}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="display:flex; gap:10px; align-items:center; margin-top:10px;">
                <span class="pill" style="background:{SIGNAL_COLOR[signal]};">{signal}</span>
                <span class="pill-outline" style="color:{RISK_COLOR[risk]}; border-color:{RISK_COLOR[risk]};">Risk: {risk}</span>
                <span class="pill-outline" style="color:{CONFIDENCE_COLOR[conf_label]}; border-color:{CONFIDENCE_COLOR[conf_label]};">Confidence: {conf_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div style='margin-top:14px; color:{MUTED}; font-size:14px; max-width:520px;'>{conf_explainer}</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Last close", f"₹{last_close:,.2f}", f"{day_change_pct:+.2f}%")
        m2.metric("Chance UP", f"{pos}%")
        m3.metric("Chance DOWN", f"{neg}%")

    with hero_right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pos,
            number={"suffix": "%", "font": {"family": "Fraunces, serif", "size": 48, "color": TEXT}},
            gauge={
                "axis": {
                    "range": [0, 100], "tickcolor": "rgba(0,0,0,0)", "tickfont": {"color": MUTED, "size": 11},
                    "tickmode": "array", "tickvals": [20, 40, 60, 80],
                },
                "bar": {"color": ACCENT, "thickness": 0.3},
                "bgcolor": INK_850,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(225,82,82,0.18)"},
                    {"range": [40, 60], "color": "rgba(192,138,46,0.18)"},
                    {"range": [60, 100], "color": "rgba(31,169,113,0.18)"},
                ],
                "threshold": {"line": {"color": TEXT, "width": 2}, "thickness": 0.85, "value": pos},
            },
        ))
        gauge.update_layout(
            height=260, margin=dict(l=45, r=45, t=40, b=30),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": MUTED, "family": "IBM Plex Sans"},
        )
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        st.caption("Chance of a >1% upward move over the prediction period.")

    st.divider()

    # --- Chart + factors ---------------------------------------------------------
    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='eyebrow'>Price history</div>", unsafe_allow_html=True)
        plot_df = featured.tail(chart_days)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25], vertical_spacing=0.03,
        )
        fig.add_trace(go.Candlestick(
            x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"],
            increasing_line_color=POSITIVE, decreasing_line_color=NEGATIVE, name="Price",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_20"], line=dict(color=ACCENT, width=1.3), name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_50"], line=dict(color="#6B7FD7", width=1.3), name="SMA 50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], line=dict(color=INK_700, width=1), name="Bollinger", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], line=dict(color=INK_700, width=1), fill="tonexty", fillcolor="rgba(107,127,215,0.06)", showlegend=False), row=1, col=1)

        vol_colors = [POSITIVE if c >= o else NEGATIVE for c, o in zip(plot_df["Close"], plot_df["Open"])]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, opacity=0.5, name="Volume"), row=2, col=1)

        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi_14"], line=dict(color="#4FBFA8", width=1.3), name="RSI 14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color=NEGATIVE, opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=POSITIVE, opacity=0.5, row=3, col=1)

        fig.update_layout(
            height=560, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=INK_850,
            font={"color": MUTED, "family": "IBM Plex Sans", "size": 11},
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
            xaxis_rangeslider_visible=False,
        )
        for r in (1, 2, 3):
            fig.update_xaxes(gridcolor=INK_700, row=r, col=1)
            fig.update_yaxes(gridcolor=INK_700, row=r, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with right:
        st.markdown("<div class='eyebrow'>What drove this signal</div>", unsafe_allow_html=True)
        top_features = result["feature_importance"].head(8).sort_values()
        labels = [FEATURE_LABELS.get(f, f) for f in top_features.index]
        bar = go.Figure(go.Bar(
            x=top_features.values, y=labels, orientation="h",
            marker_color=ACCENT,
        ))
        bar.update_layout(
            height=560, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=INK_850,
            font={"color": MUTED, "family": "IBM Plex Sans", "size": 11},
            xaxis=dict(gridcolor=INK_700), yaxis=dict(gridcolor=INK_700),
        )
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

    st.divider()

    # --- Track record --------------------------------------------------------------
    st.markdown("<div class='eyebrow'>Track record — held-out test period</div>", unsafe_allow_html=True)
    st.caption("How this exact model actually performed on data it never trained on, for this specific stock.")
    if stock in bt.index:
        row = bt.loc[stock]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Historical accuracy", f"{row['accuracy']*100:.1f}%")
        c2.metric("This stock's baseline", f"{row['baseline']*100:.1f}%")
        c3.metric("Edge over baseline", f"{row['edge_over_baseline']*100:+.1f}pp")
        c4.metric("Win rate on BUY signals", f"{row['win_rate']*100:.1f}%")
        c5.metric("Avg return per signal", f"{row['avg_strategy_return']*100:+.2f}%")
    else:
        st.info("Not enough held-out history for this stock to backtest yet.")

    st.divider()

    # --- Fundamentals / Events / News --------------------------------------------------------------
    st.markdown("<div class='eyebrow'>Context — not part of the model above</div>", unsafe_allow_html=True)
    st.subheader("Fundamentals, events & news")
    st.caption(
        "Tested for real predictive edge and didn't show enough to justify including in the trained "
        "model (see README). News can't be backtested at all — no free historical archive. Shown as "
        "honest, separate context for your own judgment."
    )

    # Fetch data first, then render each card as ONE composite HTML block — Streamlit
    # renders every st.markdown() call as a separate DOM node, so opening a <div> in one
    # call and closing it in another leaves an empty floating box instead of a real wrapper.
    fund = get_fundamentals_snapshot(stock)
    events = get_events_snapshot(stock)
    with st.spinner("Fetching news..."):
        news = get_news_context(stock)

    fcol, ecol, ncol = st.columns([1, 1, 1.4])

    with fcol:
        if fund:
            body = (
                f"<span class='mono'>ROE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{fund['roe_pct']}%</span><br>"
                f"<span class='mono'>Debt/Equity&nbsp;{fund['debt_to_equity']}</span><br>"
                f"<span class='mono'>Margin&nbsp;&nbsp;&nbsp;{fund['profit_margin_pct']}%</span><br>"
                f"<span class='mono'>Rev growth&nbsp;{fund['revenue_growth_yoy_pct']}%</span>"
                f"<div style='color:{MUTED}; font-size:12px; margin-top:10px;'>FY end: {fund['fiscal_year_end']}</div>"
            )
        else:
            body = f"<span style='color:{MUTED};'>No fundamentals data available.</span>"
        st.markdown(
            f"<div class='card'><strong>Fundamentals</strong>"
            f"<div style='color:{MUTED}; font-size:12px; margin:4px 0 12px;'>Most recent reported fiscal year</div>"
            f"{body}</div>",
            unsafe_allow_html=True,
        )

    with ecol:
        if events:
            div_txt = f"{events['days_since_dividend']}d ago" if events['days_since_dividend'] is not None else "no record"
            split_txt = f"{events['days_since_split']}d ago" if events['days_since_split'] is not None else "no record"
            body = (
                f"<span class='mono'>Last dividend&nbsp;&nbsp;{div_txt}</span><br>"
                f"<span class='mono'>Last split&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{split_txt}</span>"
            )
        else:
            body = f"<span style='color:{MUTED};'>No event data available.</span>"
        st.markdown(
            f"<div class='card'><strong>Corporate events</strong>"
            f"<div style='color:{MUTED}; font-size:12px; margin:4px 0 12px;'>Dividend / split proximity</div>"
            f"{body}</div>",
            unsafe_allow_html=True,
        )

    with ncol:
        sentiment_color = {"Positive": POSITIVE, "Negative": NEGATIVE, "Neutral": ACCENT}.get(news["label"], MUTED)
        headlines_html = ""
        if news["headlines"]:
            for h in news["headlines"][:5]:
                dot = POSITIVE if h["sentiment"] > 0.15 else (NEGATIVE if h["sentiment"] < -0.15 else MUTED)
                headlines_html += f"<div class='news-item'><span style='color:{dot};'>&#9679;</span> {h['title']}</div>"
        else:
            headlines_html = f"<span style='color:{MUTED};'>No recent news found.</span>"
        st.markdown(
            f"<div class='card'><strong>Recent news sentiment</strong>"
            f"<div style='color:{MUTED}; font-size:12px; margin:4px 0 12px;'>Live, informational only, not backtested</div>"
            f"<span class='pill-outline' style='color:{sentiment_color}; border-color:{sentiment_color};'>"
            f"{news['label']} ({news['mean_compound']:+.2f})</span>"
            f"<div style='margin-top:12px;'>{headlines_html}</div></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Recent price data"):
        st.dataframe(raw.tail(20).sort_index(ascending=False), width="stretch")

    st.divider()
    st.markdown(
        f"""
        <div class="disclaimer">
        ⚠️ Research prototype for learning purposes only, not investment advice. The BUY/HOLD/SELL
        label is a simple threshold on the model's probability output, not a recommendation from a
        financial professional. Confidence reflects this stock's own backtested track record, not a
        guarantee — most stocks in this universe show Low confidence because price-technical features
        alone don't carry a reliable edge (see README.md). Do not make real trading decisions based on
        this tool.
        </div>
        """,
        unsafe_allow_html=True,
    )
