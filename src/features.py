"""Builds price-based technical features and the classification target from OHLCV data."""
import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change()

    for window in (5, 10, 20, 50):
        df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        df[f"dist_sma_{window}"] = df["Close"] / df[f"sma_{window}"] - 1

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].replace([np.inf, -np.inf], 100).fillna(50)

    # Bollinger Bands (20-day, 2 std)
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_pctb"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Volatility (20-day std of daily returns)
    df["volatility_20"] = df["return_1d"].rolling(20).std()

    # Volume change
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_sma_20"] = df["Volume"].rolling(20).mean()

    # Momentum (N-day price change)
    for window in (5, 10, 20):
        df[f"momentum_{window}"] = df["Close"].pct_change(window)

    # --- Additional indicators: trend strength, oscillators, volume-flow, volatility-normalized range ---
    high_14, low_14 = df["High"].rolling(14).max(), df["Low"].rolling(14).min()
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["williams_r"] = -100 * (high_14 - df["Close"]) / (high_14 - low_14)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (typical_price - sma_tp) / (0.015 * mean_dev)

    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"], (df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    df["atr_pct"] = atr_14 / df["Close"] * 100

    up_move, down_move = df["High"].diff(), -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    smoothed_tr = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / smoothed_tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / smoothed_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    obv = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    df["obv_slope_10"] = (obv - obv.shift(10)) / (df["Volume"].rolling(10).mean() * 10)

    return df


def add_target(df: pd.DataFrame, horizon: int = 30, deadband: float = 0.01) -> pd.DataFrame:
    """1 = positive move > deadband, 0 = negative move < -deadband, NaN = neutral (dropped)."""
    df = df.copy()
    fwd_return = df["Close"].shift(-horizon) / df["Close"] - 1
    df["fwd_return"] = fwd_return
    df["target"] = np.select(
        [fwd_return > deadband, fwd_return < -deadband],
        [1, 0],
        default=np.nan,
    )
    return df


FEATURE_COLUMNS = [
    "return_1d",
    "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_sma_50",
    "macd", "macd_signal",
    "rsi_14",
    "bb_pctb",
    "volatility_20",
    "volume_change",
    "momentum_5", "momentum_10", "momentum_20",
]

# New indicators not yet proven to add edge — kept separate from FEATURE_COLUMNS (the
# default trained model) until tested with the same sweep/walk-forward rigor as everything
# else in this project. See README for the result.
EXTENDED_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "stoch_k", "stoch_d", "williams_r", "cci_20", "atr_pct", "adx_14", "obv_slope_10",
]

FUNDAMENTAL_COLUMNS = ["roe", "debt_to_equity", "profit_margin", "revenue_growth_yoy", "pe_ratio"]
EXTENDED_FUNDAMENTAL_COLUMNS = FUNDAMENTAL_COLUMNS + ["dividend_payout_pct", "price_to_book"]


def add_fundamentals(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: each daily row gets the most recent fundamentals known by that date
    (fundamentals.index is already the "known_from" date, set with a reporting lag in
    fundamentals.py — this never looks ahead)."""
    df = df.copy()
    fundamentals = fundamentals.sort_index()
    fund_cols = [c for c in ["roe", "debt_to_equity", "profit_margin", "revenue_growth_yoy", "eps",
                              "dividend_payout_pct", "book_value_per_share"] if c in fundamentals.columns]
    merged = pd.merge_asof(
        df.sort_index(), fundamentals[fund_cols],
        left_index=True, right_index=True, direction="backward",
    )
    merged["pe_ratio"] = (merged["Close"] / merged["eps"]).replace([np.inf, -np.inf], np.nan)
    if "book_value_per_share" in merged.columns:
        merged["price_to_book"] = (merged["Close"] / merged["book_value_per_share"]).replace([np.inf, -np.inf], np.nan)
        merged = merged.drop(columns="book_value_per_share")
    return merged.drop(columns="eps")


def build_dataset(df: pd.DataFrame, horizon: int = 30, deadband: float = 0.01, fundamentals: pd.DataFrame = None,
                   technical_columns=None, fundamental_columns=None) -> pd.DataFrame:
    technical_columns = technical_columns or FEATURE_COLUMNS
    fundamental_columns = fundamental_columns or FUNDAMENTAL_COLUMNS
    df = add_features(df)
    df = add_target(df, horizon=horizon, deadband=deadband)
    feature_cols = list(technical_columns)
    if fundamentals is not None:
        df = add_fundamentals(df, fundamentals)
        feature_cols = feature_cols + list(fundamental_columns)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["target"])
    return df
