# signals.py
# ─────────────────────────────────────────────────────────────────────────────
# All signals are STATE-BASED — they output +1, -1, or 0 every single day
# based on current market conditions, not just at crossover moments.
# This means the tool always has a signal for today.
#
# State-based logic:
#   +1 = currently bullish condition
#   -1 = currently bearish condition
#    0 = neutral / not enough conviction
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
from indicators import (moving_average, rsi, macd, bollinger_bands,
                        stochastic_oscillator, psar, crossovers,
                        find_extremums, derivative_window, atr, cmf)
from profiles import PROFILES

_DEFAULT_PROFILE = PROFILES["swing"]


# ─── Trend filter ──────────────────────────────────────────────────────────────
def trend_filter(close, window=200):
    ma = moving_average(close, window=window)
    trend = pd.Series(0, index=close.index)
    trend[close > ma] =  1
    trend[close < ma] = -1
    return trend


# ─── State-based signals ───────────────────────────────────────────────────────

def rsi_signal(close, period=14, oversold=30, overbought=70):
    """
    State: +1 while RSI < oversold, -1 while RSI > overbought.
    Fires every day the condition holds.
    """
    rsi_values = rsi(close, period=period)
    signal = pd.Series(0, index=close.index)
    signal[rsi_values < oversold]   =  1
    signal[rsi_values > overbought] = -1
    return signal


def macd_signal(close):
    """
    State: +1 while MACD line is above signal line (bullish momentum).
           -1 while MACD line is below signal line (bearish momentum).
    Fires every day, not just at crossover.
    """
    macd_df = macd(close)
    signal  = pd.Series(0, index=close.index)
    signal[macd_df["macd"] > macd_df["macd_signal_line"]] =  1
    signal[macd_df["macd"] < macd_df["macd_signal_line"]] = -1
    return signal


def bbands_signal(close):
    """
    State: +1 while price is near/below lower band (oversold).
           -1 while price is near/above upper band (overbought).
    """
    bb = bollinger_bands(close)
    signal = pd.Series(0, index=close.index)
    signal[bb["bb_price_proximity"] == -1] =  1
    signal[bb["bb_price_proximity"] ==  1] = -1
    return signal


def stochastic_signal(close, high, low):
    """
    State: +1 while %K < 20 (oversold) AND %K > %D (momentum turning up).
           -1 while %K > 80 (overbought) AND %K < %D (momentum turning down).
    """
    stoch  = stochastic_oscillator(close, high, low)
    signal = pd.Series(0, index=close.index)
    signal[(stoch["%K"] < 20) & (stoch["%K"] > stoch["%D"])] =  1
    signal[(stoch["%K"] > 80) & (stoch["%K"] < stoch["%D"])] = -1
    return signal


def psar_signal(close, high, low):
    """
    State: +1 while PSAR is below price (uptrend).
           -1 while PSAR is above price (downtrend).
    Fires every day based on current trend direction.
    """
    psar_df = psar(close, high, low)
    signal  = pd.Series(0, index=close.index)
    signal[psar_df["psar"] < close] =  1
    signal[psar_df["psar"] > close] = -1
    return signal


def ma_crossover_signal(close, short_window=20, long_window=50, exponential=True):
    """
    State: +1 while short MA is above long MA (uptrend).
           -1 while short MA is below long MA (downtrend).
    Fires every day based on current MA alignment.
    """
    ma_short = moving_average(close, window=short_window, exponential=exponential)
    ma_long  = moving_average(close, window=long_window,  exponential=exponential)
    signal   = pd.Series(0, index=close.index)
    signal[ma_short > ma_long] =  1
    signal[ma_short < ma_long] = -1
    return signal


def support_resistance_signal(close, window=5, lookback=20):
    """
    State: +1 while price is above the most recent resistance (breakout confirmed).
           -1 while price is below the most recent support (breakdown confirmed).
    Persists as long as price stays on the breakout side.
    """
    signal = pd.Series(0, index=close.index)
    ext    = find_extremums(close, window=window)

    for i in range(lookback, len(close)):
        recent      = ext.iloc[i - lookback: i]
        resistances = recent[recent["extremum_type"] ==  1]["extremum_value"].dropna()
        supports    = recent[recent["extremum_type"] == -1]["extremum_value"].dropna()
        if resistances.empty or supports.empty:
            continue
        resistance = resistances.iloc[-1]
        support    = supports.iloc[-1]
        curr_close = close.iloc[i]
        if curr_close > resistance:
            signal.iloc[i] =  1
        elif curr_close < support:
            signal.iloc[i] = -1

    return signal


def trend_strength_signal(close, window=5):
    """
    State: +1 while price structure shows HH + HL (uptrend).
           -1 while price structure shows LH + LL (downtrend).
    Forward-filled so it persists between extremum detections.
    """
    ext    = find_extremums(close, window=window)
    signal = pd.Series(0, index=close.index)
    highs  = ext[ext["extremum_type"] ==  1]["extremum_value"].dropna()
    lows   = ext[ext["extremum_type"] == -1]["extremum_value"].dropna()

    for i in range(1, len(highs)):
        idx = highs.index[i]
        signal[idx] = 1 if highs.iloc[i] > highs.iloc[i - 1] else -1

    for i in range(1, len(lows)):
        idx = lows.index[i]
        if lows.iloc[i] > lows.iloc[i - 1]:
            if signal[idx] >= 0:
                signal[idx] =  1
        else:
            if signal[idx] <= 0:
                signal[idx] = -1

    signal = signal.replace(0, np.nan).ffill().fillna(0).astype(int)
    return signal


def squeeze_breakout_signal(close, high, low):
    """
    State: +1 while price is above upper band after a squeeze (bullish breakout).
           -1 while price is below lower band after a squeeze (bearish breakout).
    Persists as long as price stays outside the bands post-squeeze.
    """
    bb         = bollinger_bands(close)
    atr_values = atr(close, high, low)
    signal     = pd.Series(0, index=close.index)

    squeeze    = bb["bb_squeeze"] == -1
    atr_expand = derivative_window(atr_values) > 0

    # Persist breakout state while price stays outside bands
    above_upper = close > bb["bb_upper_band"]
    below_lower = close < bb["bb_lower_band"]

    signal[squeeze.shift(1).fillna(False) & above_upper & atr_expand] =  1
    signal[squeeze.shift(1).fillna(False) & below_lower & atr_expand] = -1
    return signal


def volume_confirmed_reversal_signal(close, high, low, volume, window=5, cmf_threshold=0.05):
    """
    State: +1 while CMF is positive at a local low (buying pressure at support).
           -1 while CMF is negative at a local high (selling pressure at resistance).
    Forward-filled to persist for window days after the extremum.
    """
    ext        = find_extremums(close, window=window)
    cmf_values = cmf(close, high, low, volume)
    signal     = pd.Series(0, index=close.index)

    signal[(ext["extremum_type"] == -1) & (cmf_values >  cmf_threshold)] =  1
    signal[(ext["extremum_type"] ==  1) & (cmf_values < -cmf_threshold)] = -1

    # Forward-fill for window days so signal persists after the extremum
    signal = signal.replace(0, np.nan).ffill(limit=window).fillna(0).astype(int)
    return signal


def divergence_signal(close, rsi_period=14, window=5):
    """
    State: +1 after bullish divergence detected (price lower low + RSI higher low).
           -1 after bearish divergence detected (price higher high + RSI lower high).
    Forward-filled for window days after detection.
    """
    rsi_values = rsi(close, period=rsi_period)
    price_ext  = find_extremums(close,      window=window)
    rsi_ext    = find_extremums(rsi_values, window=window)
    signal     = pd.Series(0, index=close.index)

    price_highs = price_ext[price_ext["extremum_type"] ==  1]["extremum_value"].dropna()
    rsi_highs   = rsi_ext  [rsi_ext  ["extremum_type"] ==  1]["extremum_value"].dropna()
    price_lows  = price_ext[price_ext["extremum_type"] == -1]["extremum_value"].dropna()
    rsi_lows    = rsi_ext  [rsi_ext  ["extremum_type"] == -1]["extremum_value"].dropna()

    for i in range(1, len(price_highs)):
        idx   = price_highs.index[i]
        rsi_h = rsi_highs[rsi_highs.index <= idx]
        if len(rsi_h) >= 2:
            if price_highs.iloc[i] > price_highs.iloc[i-1] and rsi_h.iloc[-1] < rsi_h.iloc[-2]:
                signal[idx] = -1

    for i in range(1, len(price_lows)):
        idx   = price_lows.index[i]
        rsi_l = rsi_lows[rsi_lows.index <= idx]
        if len(rsi_l) >= 2:
            if price_lows.iloc[i] < price_lows.iloc[i-1] and rsi_l.iloc[-1] > rsi_l.iloc[-2]:
                signal[idx] =  1

    # Forward-fill for window days so signal persists after detection
    signal = signal.replace(0, np.nan).ffill(limit=window).fillna(0).astype(int)
    return signal


# ─── Safe wrapper ──────────────────────────────────────────────────────────────
def _safe(fn, index, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, ZeroDivisionError, KeyError) as e:
        print(f"  ⚠️  Skipping {fn.__name__}: {e}")
        return pd.Series(0, index=index)


# ─── Confirmation filter ───────────────────────────────────────────────────────
def confirm_signal(decision, window=1):
    """
    Confirm a signal if it appeared at least once in the prior `window` days.
    window=1 → no confirmation needed, fires immediately.
    """
    if window <= 1:
        return decision
    confirmed = pd.Series("NEUTRAL", index=decision.index)
    for i in range(window, len(decision)):
        today    = decision.iloc[i]
        if today == "NEUTRAL":
            continue
        lookback = decision.iloc[i - window: i]
        if (lookback == today).any():
            confirmed.iloc[i] = today
    return confirmed


# ─── Combined signal ───────────────────────────────────────────────────────────
def combined_signal(close, high, low, volume=None, profile=None):
    """
    Pipeline:
      1. Weighted score from all active state-based indicators
      2. Re-normalise if indicators were skipped
      3. Trend filter: attenuate counter-trend scores by trend_penalty
      4. Threshold → decision
      5. Confirmation window
    """
    if profile is None:
        profile = _DEFAULT_PROFILE

    n   = len(close)
    idx = close.index

    indicator_fns = {
        "rsi":                (lambda: _safe(rsi_signal, idx, close,
                                   period=profile["rsi_period"],
                                   oversold=profile["rsi_oversold"],
                                   overbought=profile["rsi_overbought"]),
                               profile["rsi_period"] + 1),
        "macd":               (lambda: _safe(macd_signal, idx, close),             26),
        "bbands":             (lambda: _safe(bbands_signal, idx, close),            20),
        "stochastic":         (lambda: _safe(stochastic_signal, idx, close, high, low), 14),
        "psar":               (lambda: _safe(psar_signal, idx, close, high, low),   2),
        "ma_crossover":       (lambda: _safe(ma_crossover_signal, idx, close,
                                   short_window=profile["ma_short_window"],
                                   long_window=profile["ma_long_window"],
                                   exponential=profile["ma_exponential"]),
                               profile["ma_long_window"]),
        "support_resistance": (lambda: _safe(support_resistance_signal, idx, close,
                                   window=profile.get("extremum_window", 5),
                                   lookback=profile.get("sr_lookback", 20)),
                               profile.get("extremum_window", 5) * 2),
        "trend_strength":     (lambda: _safe(trend_strength_signal, idx, close,
                                   window=profile.get("extremum_window", 5)),
                               profile.get("extremum_window", 5) * 4),
        "squeeze_breakout":   (lambda: _safe(squeeze_breakout_signal, idx, close, high, low), 20),
        "volume_reversal":    (lambda: (
                                   _safe(volume_confirmed_reversal_signal, idx,
                                         close, high, low, volume,
                                         window=profile.get("extremum_window", 5),
                                         cmf_threshold=profile.get("cmf_threshold", 0.05))
                                   if volume is not None else pd.Series(0, index=idx)
                               ), 21),
        "divergence":         (lambda: _safe(divergence_signal, idx, close,
                                   rsi_period=profile["rsi_period"],
                                   window=profile.get("extremum_window", 5)),
                               profile["rsi_period"] + profile.get("extremum_window", 5)),
    }

    weights       = profile["weights"]
    score         = pd.Series(0.0, index=idx)
    active_weight = 0.0

    for name, (fn, min_bars) in indicator_fns.items():
        if name not in weights or weights[name] == 0:
            continue
        if n < min_bars:
            print(f"  ⚠️  Skipping {name}: need {min_bars} bars, have {n}")
            continue
        signal        = fn()
        score        += signal.reindex(idx, fill_value=0) * weights[name]
        active_weight += weights[name]

    if 0 < active_weight < 1.0:
        score = score / active_weight
        print(f"  ℹ️  Score re-normalised (active weight: {active_weight:.2f})")

    trend_window = profile["trend_filter_window"]
    if n >= trend_window:
        trend        = trend_filter(close, window=trend_window)
        penalty      = profile.get("trend_penalty", 0.5)
        counter_buy  = (trend == -1) & (score > 0)
        counter_sell = (trend ==  1) & (score < 0)
        score        = score.copy()
        score[counter_buy]  *= penalty
        score[counter_sell] *= penalty
    else:
        print(f"  ⚠️  Trend filter skipped: need {trend_window} bars, have {n}")

    threshold = profile["score_threshold"]
    decision  = pd.Series("NEUTRAL", index=idx)
    decision[score >  threshold] = "BUY"
    decision[score < -threshold] = "SELL"
    decision = confirm_signal(decision, window=profile["confirmation_window"])

    return pd.DataFrame({"score": score, "decision": decision})