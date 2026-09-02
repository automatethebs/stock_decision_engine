# Stock Decision Engine (local prototype)

Traditional ML (no LLM/AI-chat) probability signals for stock movement direction,
based on historical technical-indicator patterns. Runs 100% locally.

## Setup

```
py -m pip install -r requirements.txt            # runtime dependencies
py -m pip install -r requirements-dev.txt        # test-only dependencies (pytest)
```

## Tests

The core logic — feature construction, target labelling, the time-based train/test split,
backtest drawdown math, value-screener scoring, and data-loading/event parsing — has a
unit-test suite that runs **entirely offline against synthetic data** (no network calls,
no dependency on the downloaded CSVs or the trained model):

```
py -m pytest            # run the whole suite
py -m pytest tests/test_features.py   # one file
py -m pytest --cov=src --cov-report=term-missing   # with coverage
```

`pyproject.toml` configures `pythonpath = ["src"]` and `testpaths = ["tests"]`, so the
tests import the `src` modules directly, exactly the way the CLI scripts do.

## Usage

```
py src\data_loader.py      # download full available OHLCV history for the full Nifty 500 universe
py src\train.py            # build features, train + compare models, save the best one
py src\backtest.py         # evaluate the saved model on held-out time-ordered data
py src\predict.py RELIANCE # CLI report for one stock
streamlit run src\app.py   # local web UI (opens http://localhost:8501)
```

Default universe is the full **Nifty 500** — `data_loader.py`'s `fetch_nifty500()` pulls the
official constituent list live from NSE's archive CSV and caches it to
`data/nifty500_symbols.txt` (NSE rebalances this list only a few times a year, so the cache is
reused rather than re-fetched every run; delete that file to force a refresh). All 500 symbols
downloaded successfully (0 skipped) in the current dataset. The smaller, fixed `NIFTY50` list
in `data_loader.py` is kept as a fallback if you want a faster, smaller run — pass it explicitly
to `download_all(stocks=[...])` if so.

To also populate fundamentals/events context (optional, shown in the UI but not part of the
trained model — see "Three data pillars" below):
```
py src\fundamentals_screener.py   # annual fundamentals per stock (10-12yr, screener.in) -> data/fundamentals/
py src\events.py                  # dividend/split history per stock -> data/events/
```
`fundamentals_screener.py` scrapes the free public site [screener.in](https://www.screener.in)
and is the recommended source — 10-12 years of history per stock vs. `fundamentals.py`'s
older yfinance-based version, which is capped at ~4-5 years and kept only as a fallback if
screener.in is ever unreachable. Both write the same CSV format, so nothing downstream needs
to change if you switch between them.

## Three data pillars — what's validated vs informational

The app combines **technical, fundamental, and news/event data**, but they are not all
treated the same, because they don't all hold up to the same evidence:

| Pillar | Used in the trained model? | Why |
|---|---|---|
| **Technical** (price/volume indicators) | ✅ Yes — the only pillar in `model.joblib` | Full multi-year history per stock, rigorously tested (see below), tied to baseline but at least not worse than it. |
| **Fundamentals** (ROE, debt/equity, margins, revenue growth, P/E) | ❌ No — shown as context only | Tested twice on the original 49-stock universe. First with yfinance's ~4-5yr history: technical-only did *worse* than baseline on that short window (-3 to -4%), fundamentals recovered it to roughly flat. Retested with `fundamentals_screener.py`'s 10-12yr history (3x more data, 48 stocks): single-split edge stayed ~flat (±0.3%), and **4-fold walk-forward confirmed it wasn't a lucky split** — mean edge -0.08% (Random Forest, essentially zero/noise) to -4.1% (XGBoost, consistently worse). `pe_ratio` ranks as the single most "important" feature by the model's own accounting in both tests, yet accuracy doesn't move — more history didn't change the honest conclusion. |
| **Events** (dividend/split proximity) | ❌ No — shown as context only | Tested on the *full* history (dividends/splits go back decades, no window restriction needed): technical+events was slightly *worse* than technical-only, despite `days_since_dividend` ranking as the single most "important" feature by the model's own accounting. High importance without real accuracy gain — a useful lesson, not a green light. |
| **News sentiment** (live headlines, VADER lexicon scoring) | ❌ No — cannot be | No free historical news archive exists to backtest against. Shown live, informational only. |

**Why show fundamentals/events/news at all if they're not in the model?** Because they're
genuinely useful *context* for a human decision — knowing a stock is cheap (low P/E), about
to go ex-dividend, or in the news for a lawsuit is real information — it's just not something
this project could prove adds predictive edge to the specific 30-day direction forecast. The
UI shows them clearly labeled and separated from the validated probability, instead of
silently blending them into one number that would then overstate its own rigor.

## How it works

1. **data_loader.py** — downloads full available price history (`period="max"`, often 20+
   years where listed) via yfinance for the full Nifty 500 universe (list fetched live from
   NSE, cached locally), saves one CSV per stock.
2. **features.py** — builds technical indicators (returns, SMA/EMA distance, MACD, RSI,
   Bollinger %B, volatility, volume change, momentum) and the target label: whether the
   price moves >±1% over the next N days (30 by default). Moves inside that deadband are
   dropped as "neutral" — this avoids training on sign-of-near-zero noise.
3. **train.py** — pools all stocks into one dataset, splits **by time** (last 20% of each
   stock's history held out, never shuffled) to avoid future leakage, trains Logistic
   Regression / Random Forest / XGBoost (each wrapped in `CalibratedClassifierCV` so the
   output probabilities are calibrated, not raw overconfident model scores), and saves
   whichever has the highest **edge over the majority-class baseline** — not raw accuracy,
   which is a misleading selection criterion here (see below) — to `models/model.joblib`.
4. **backtest.py** — re-scores the held-out period per stock: accuracy/precision/recall,
   win rate, average return, and compares a "go long only when the model predicts positive"
   strategy against plain buy-and-hold. Drawdown is computed on **non-overlapping** periods
   (every `horizon`-th row) so it's a valid compounding calculation, not the overlapping-window
   artifact from earlier versions.
5. **fundamentals_screener.py** — scrapes annual P&L/balance sheet tables from screener.in
   (free, ~10-12yr history per stock; row labels are matched flexibly since they vary by
   sector — e.g. banks use "Revenue"/"Deposits" instead of "Sales"/"Borrowings"), computes
   ROE, debt/equity, profit margin, revenue growth, EPS/P/E, indexed by the date they become
   "known" (fiscal year-end + a 60-day reporting lag, so no leakage). `fundamentals.py` is
   the older yfinance-based version (~4-5yr cap), kept as a fallback. Context only, not in
   the trained model (see table above) — tested at both history depths, neither showed a
   robust edge.
6. **events.py** — downloads dividend/split history per stock (decades of real dates) and
   derives `days_since_dividend` / `days_since_split`. Context only, not in the trained model.
7. **news.py** — fetches live news headlines via yfinance and scores them with **VADER**
   (a rule-based lexicon + grammar heuristics sentiment tool, not a trained model or LLM —
   consistent with this project's no-AI/no-LLM constraint). Live/informational only, never
   backtested.
8. **predict.py** — loads the saved (technical-only) model, computes features for the latest
   available date for one stock, and returns positive/negative probability, a
   volatility-derived risk level, a BUY/HOLD/SELL recommendation, and feature importances.
   `predict_all()` runs this across every downloaded stock for the screener.
   `get_fundamentals_snapshot()` / `get_events_snapshot()` / `get_news_context()` fetch the
   other three pillars for display, separate from the trained model's output.
9. **value_screener.py** — a deliberately **non-predictive** companion: ranks stocks by
   objective, already-known fundamentals (earnings yield, ROE, debt/equity, revenue growth,
   and a **durability** trend factor) into a 0-100 value+quality score, using percentile
   ranks across the universe so one extreme stock can't dominate the score. This is the
   classic Graham/Buffett/Greenblatt value-investing approach — it doesn't forecast
   anything, so it doesn't carry the accuracy caveats below. It's a genuinely different
   tool sitting next to the ML signal, not another attempt at the same forecasting problem.

   **Durability** specifically exists to catch two failure modes of a single-year snapshot:
   a **value trap** (looks cheap, but the business is structurally declining) and its
   mirror, a **turnaround** (was declining, but has recently re-accelerated). It compares
   each stock's recent (~2yr) revenue growth and profit margin against its own earlier
   history, using the 10-12yr fundamentals already on disk — arithmetic on real historical
   numbers, not news. News was deliberately *not* used for this: a structural, multi-year
   question needs multi-year data, and live headlines are too short-term/noisy to answer it
   (and can't be backtested at all — see below). Stocks get labeled Compounder / Turnaround /
   Declining / Stable based on this comparison.
10. **app.py** — the Streamlit UI, three tabs:
    - **Stock Detail** — one stock at a time: BUY/HOLD/SELL badge, risk badge, probability bar,
      price chart with SMA/Bollinger overlay + RSI subplot, plain-English top factors, an
      expandable **track record** panel (this stock's actual backtested accuracy/win-rate/
      average return on held-out data), a **Fundamentals, events & news** section (clearly
      labeled as context, not part of the model), and a recent-data table.
    - **Screener** — scans the full universe (500 stocks) at once into one sortable table
      (signal, probabilities, risk, as-of date), so you don't have to check stocks one by one.
    - **Value Screener** — ranks all stocks by the value+quality+durability score from
      `value_screener.py`, top 3 highlighted, turnaround candidates called out separately.
      Not a prediction — a comparison of current and historical fundamentals.

## Accuracy ceiling — read this before trusting the output

**Directional accuracy on price-technical features alone is at the majority-class baseline,
not meaningfully above it — and this is a real, well-evidenced finding, not a bug to fix.**

Two rounds of rigorous testing back this up:

1. **Horizon × deadband sweep** (12 configs: horizons 5/10/20/30 days × deadbands 1%/3%/5%,
   Random Forest and XGBoost, ~270k pooled rows). Result: **edge over baseline was ≤0 in
   every single configuration.** "Accuracy" numbers rise with a wider deadband (up to 65%
   at horizon=30/deadband=5%), but that's the majority-class baseline rising too — Nifty 50
   has a long-run positive drift, so a wider deadband skews the label distribution, not the
   model's skill. Raw accuracy alone is a misleading metric for this reason; edge-over-baseline
   is the honest one, and that stayed flat-to-negative throughout.
2. **Walk-forward validation** (4 rolling folds, horizon=30/deadband=3%, Logistic Regression /
   Random Forest / XGBoost). Result: mean edge over baseline was **+0.02% for Logistic
   Regression, -0.07% for Random Forest, -0.67% for XGBoost** — all within noise of zero,
   confirming the sweep result wasn't a single-split fluke. XGBoost's consistently negative
   edge indicates mild overfitting to noise rather than any real signal.
3. **Cross-sectional ranking** (Information Coefficient — Spearman rank correlation between
   predicted probability and actual forward return, computed weekly across all 49 stocks).
   Result: **IC = +0.03** — a real but very weak signal, well below the ~0.05-0.10 threshold
   quant research treats as minimally usable.
4. **Fundamentals and events**, tested the same way (see "Three data pillars" above): fundamentals
   showed a modest effect on yfinance's short (~4yr) window, but re-testing with 3x more history
   from screener.in (10-12yr) — including a proper 4-fold walk-forward, not just one split — found
   no robust edge either. Events (dividend/split proximity) showed no improvement (net negative)
   even on the full multi-decade history. More data didn't change the honest conclusion for either.
5. **Extended technical indicators** (added Stochastic %K/%D, Williams %R, CCI, ADX, ATR%,
   OBV slope — 21 features total, `EXTENDED_FEATURE_COLUMNS` in `features.py`). Result:
   `adx_14` (trend strength) ranked as the single most "important" feature by the model's own
   accounting — higher than MACD, higher than anything in the original 14 — yet walk-forward
   edge stayed flat (RF: -0.04% mean) to negative (XGBoost: -1.41% mean). Not integrated into
   the default model.
6. **Extended fundamentals** (added `dividend_payout_pct` and `price_to_book` — 7 ratios total,
   `EXTENDED_FUNDAMENTAL_COLUMNS`). Same pattern: `price_to_book` ranked #2 by importance
   (above revenue growth, profit margin, ROE), walk-forward edge stayed flat (RF: -0.31% mean)
   to more negative than the original 5-ratio set (XGBoost: -5.60% mean). Not integrated.

7. **Scaling from 49 to the full Nifty 500 universe** (500 stocks, 1,245,525 training rows +
   311,632 test rows — 5.7x more data than every test above). Result: **edge over baseline
   was -0.01pp** (essentially exactly zero) — the same honest ceiling holds at 10x the stock
   count and 5.7x the data. This is the strongest single piece of evidence in this project
   that the ceiling is a genuine market-efficiency finding, not a small-sample artifact:
   scaling the universe up an order of magnitude changed nothing.

**The recurring lesson across all seven tests above**: a feature ranking high in the model's
own "importance" accounting is not evidence it improves real accuracy — it can just mean the
model is leaning on a feature that's correlated with the outcome by coincidence in-sample,
without that correlation holding out-of-sample. Every one of these tests would have looked
like a success if evaluated only by feature importance or a single train/test split; only the
walk-forward check catches it.

**Why 90% (or any large edge) isn't a realistic target:** daily-bar stock direction from
price/technical indicators alone is close to a random walk at this level of tooling — this
is consistent with decades of market-efficiency research, not a limitation specific to this
codebase. A model that scored 90% on this task would almost certainly indicate a data leakage
or evaluation bug, not genuine skill. Real, durable edges (where they exist at all) tend to
come from sources this prototype deliberately excludes: alternative data, fundamentals,
order-flow/microstructure, or much shorter holding periods traded at scale — not retail-grade
daily OHLCV technicals.

**What was actually improved, honestly:**
- Model selection now optimizes for edge-over-baseline instead of raw accuracy (the old
  criterion would have silently rewarded a wider deadband for a fake accuracy bump).
- Probabilities are now calibrated (`CalibratedClassifierCV`) so a "62%" reflects observed
  historical frequency more faithfully, instead of an overconfident raw model score.
- Confirmed via independent sweep + walk-forward validation that this is a genuine ceiling
  for this feature set, not an artifact of one lucky/unlucky split.

**Other known limitations:**
- yfinance is used for historical downloads only (not real-time) — no scheduled/live data
  pipeline exists; re-run `data_loader.py` manually to refresh.
- The screener, value screener, and per-stock track record all call the model/fundamentals
  across the full 500-stock universe and are cached per session (`st.cache_data`) — the first
  load of each tab is noticeably slower (tens of seconds) at this scale than the 49-stock
  version was; subsequent loads are instant from cache.

**Where a real edge would have to come from next** (not attempted here — bigger scope):
fundamental/valuation features, sector/index-relative strength, order-flow or options data,
or ensembling across many small, uncorrelated weak signals rather than one model on one
feature set.

## Portfolio backtest — does stock-picking with this signal actually beat just holding everything?

*(Numbers below are from the original 49-stock Nifty 50 universe — not yet re-run at the full
500-stock scale. Given the technical-only edge stayed at ~0% even after 10x-ing the stock
count, there's no reason to expect this conclusion would flip; re-run `portfolio_backtest.py`
yourself against the current 500-stock model if you want the up-to-date numbers.)*

`portfolio_backtest.py` answers this directly: rank all 49 stocks by predicted probability,
hold the top 10, rebalance every 30 days (non-overlapping, so no double-counting), net of a
realistic ~0.4% round-trip trading cost — on the held-out test period only — versus simply
holding all 49 stocks equal-weight and doing nothing.

| Metric | Top-10 signal-based portfolio | Equal-weight buy & hold (no signal) |
|---|---|---|
| CAGR | 27.17% | 27.04% |
| **Sharpe ratio** | **1.39** | **1.55** |
| **Max drawdown** | **-18.89%** | **-9.90%** |
| Win rate | 66.0% | 71.7% |

**The signal-based portfolio is worse on every risk-adjusted metric** — nearly identical
return, but roughly double the drawdown and a lower Sharpe ratio. This is the expected
consequence of concentrating into 10 stocks based on a signal with no real edge (see above):
you lose diversification without gaining expected return, so you take on more risk for the
same payoff. Both portfolios' ~27% CAGR reflects the Nifty 50's broad-market strength over
this period, not stock-picking skill.

**The honest, evidence-backed conclusion of this entire project: diversified buy-and-hold of
the stock universe outperformed every stock-picking approach actually built and tested here**,
on a risk-adjusted basis. Run `py src\portfolio_backtest.py` to reproduce.
