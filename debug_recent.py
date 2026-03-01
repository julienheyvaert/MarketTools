"""
Debug recent days — run from MarketAnalysis directory:
    python3 debug_recent.py
"""
import pandas as pd
from fetch_data import fetch_data
from profiles import load_profiles
from signals import (rsi_signal, macd_signal, psar_signal, ma_crossover_signal,
                     support_resistance_signal, squeeze_breakout_signal,
                     divergence_signal, trend_filter, combined_signal)
from indicators import moving_average, rsi

TICKER  = "BTC-USD"
PERIOD  = "1y"
PROFILE_KEY = "swing"

profiles = load_profiles()
PROFILE  = profiles[PROFILE_KEY]

data   = fetch_data(TICKER, period=PERIOD)
close  = data["Close"].squeeze()
high   = data["High"].squeeze()
low    = data["Low"].squeeze()
volume = data["Volume"].squeeze()
idx    = close.index
w      = PROFILE["weights"]

# ── Raw RSI values ─────────────────────────────────────────
rsi_values = rsi(close, period=PROFILE["rsi_period"])

# ── All signals ────────────────────────────────────────────
sigs = {
    "rsi":                rsi_signal(close, period=PROFILE["rsi_period"],
                              oversold=PROFILE["rsi_oversold"],
                              overbought=PROFILE["rsi_overbought"]),
    "macd":               macd_signal(close),
    "psar":               psar_signal(close, high, low),
    "ma_crossover":       ma_crossover_signal(close,
                              short_window=PROFILE["ma_short_window"],
                              long_window=PROFILE["ma_long_window"],
                              exponential=PROFILE["ma_exponential"]),
    "support_resistance": support_resistance_signal(close,
                              window=PROFILE.get("extremum_window", 5),
                              lookback=PROFILE.get("sr_lookback", 20)),
    "squeeze_breakout":   squeeze_breakout_signal(close, high, low),
    "divergence":         divergence_signal(close,
                              rsi_period=PROFILE["rsi_period"],
                              window=PROFILE.get("extremum_window", 5)),
}

# ── Trend ──────────────────────────────────────────────────
ma  = moving_average(close, window=PROFILE["trend_filter_window"])
trend = pd.Series(0, index=idx)
trend[close > ma] =  1
trend[close < ma] = -1

# ── Score BEFORE penalty ───────────────────────────────────
score_raw = pd.Series(0.0, index=idx)
for name, sig in sigs.items():
    if w.get(name, 0) > 0:
        score_raw += sig.reindex(idx, fill_value=0) * w[name]

# ── Score AFTER penalty ────────────────────────────────────
penalty = PROFILE["trend_penalty"]
score_penalised = score_raw.copy()
score_penalised[(trend == -1) & (score_raw > 0)] *= penalty
score_penalised[(trend ==  1) & (score_raw < 0)] *= penalty

# ── Actual combined signal ─────────────────────────────────
result = combined_signal(close, high, low, volume=volume, profile=PROFILE)

threshold = PROFILE["score_threshold"]

print(f"\n{'═'*80}")
print(f"  LAST 10 DAYS SIGNAL DEBUG — {TICKER} | {PROFILE_KEY}")
print(f"  RSI oversold<{PROFILE['rsi_oversold']}  overbought>{PROFILE['rsi_overbought']}  threshold=±{threshold}  penalty={penalty}")
print(f"{'═'*80}")
print(f"  {'Date':<12} {'Close':>8}  {'RSI':>5}  {'Raw':>7}  {'Penalised':>10}  {'Decision':>8}  Active signals")
print(f"  {'─'*12} {'─'*8}  {'─'*5}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*30}")

for date in idx[-10:]:
    t     = trend[date]
    t_str = "↑" if t == 1 else "↓" if t == -1 else "─"
    r     = rsi_values[date]
    raw   = score_raw[date]
    pen   = score_penalised[date]
    dec   = result.loc[date, "decision"]
    dec_str = "BUY 🟢" if dec == "BUY" else "SELL 🔴" if dec == "SELL" else "·"

    # Which signals are active on this day
    active = []
    for name, sig in sigs.items():
        if w.get(name, 0) == 0:
            continue
        val = sig.reindex(idx).get(date, 0)
        if val == 1:
            active.append(f"{name}(+)")
        elif val == -1:
            active.append(f"{name}(-)")

    active_str = ", ".join(active) if active else "none"
    print(f"  {str(date.date()):<12} {close[date]:>8,.0f}  {r:>5.1f}  {raw:>+7.3f}  {pen:>+10.3f}  {dec_str:>8}  {active_str}")

print(f"{'═'*80}\n")