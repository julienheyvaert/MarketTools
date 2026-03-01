"""
Deep dive: trace exactly what happens to the 14 BUY candidate days.
Run from your MarketAnalysis directory: python3 diagnose2.py
"""
import pandas as pd
import numpy as np
from fetch_data import fetch_data
from signals import (rsi_signal, macd_signal, psar_signal, ma_crossover_signal,
                     support_resistance_signal, trend_strength_signal,
                     squeeze_breakout_signal, volume_confirmed_reversal_signal,
                     divergence_signal, trend_filter)
from indicators import moving_average
from profiles import PROFILES

TICKER = "BTC-USD"
PERIOD = "1y"
profile = PROFILES["swing"]

data   = fetch_data(TICKER, period=PERIOD)
close  = data["Close"].squeeze()
high   = data["High"].squeeze()
low    = data["Low"].squeeze()
volume = data["Volume"].squeeze()
n      = len(close)
idx    = close.index
w      = profile["weights"]

# Build all individual signals
sigs = {
    "rsi":                rsi_signal(close, period=profile["rsi_period"],
                              oversold=profile["rsi_oversold"], overbought=profile["rsi_overbought"]),
    "macd":               macd_signal(close),
    "psar":               psar_signal(close, high, low),
    "ma_crossover":       ma_crossover_signal(close, short_window=profile["ma_short_window"],
                              long_window=profile["ma_long_window"], exponential=profile["ma_exponential"]),
    "support_resistance": support_resistance_signal(close, window=profile.get("extremum_window",5),
                              lookback=profile.get("sr_lookback",20)),
    "trend_strength":     trend_strength_signal(close, window=profile.get("extremum_window",5)),
    "squeeze_breakout":   squeeze_breakout_signal(close, high, low),
    "volume_reversal":    volume_confirmed_reversal_signal(close, high, low, volume,
                              window=profile.get("extremum_window",5),
                              cmf_threshold=profile.get("cmf_threshold",0.05)),
    "divergence":         divergence_signal(close, rsi_period=profile["rsi_period"],
                              window=profile.get("extremum_window",5)),
}

# Raw score
score = pd.Series(0.0, index=idx)
for name, sig in sigs.items():
    if w.get(name, 0) > 0:
        score += sig.reindex(idx, fill_value=0) * w[name]

# Trend
ma100 = moving_average(close, window=profile["trend_filter_window"])
trend = pd.Series(0, index=idx)
trend[close > ma100] =  1
trend[close < ma100] = -1

threshold = profile["score_threshold"]
penalty   = profile.get("trend_penalty", 0.5)

# Find the candidate days (above threshold BEFORE filter)
buy_candidates  = score[score >  threshold].index
sell_candidates = score[score < -threshold].index

print(f"\n{'═'*70}")
print(f"  CANDIDATE BUY DAYS (score > +{threshold} before trend filter)")
print(f"{'═'*70}")
print(f"  {'Date':<12} {'Score':>7}  {'Trend':>6}  {'Penalised':>10}  {'Fires?':>7}  Contributing signals")
print(f"  {'─'*12} {'─'*7}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*30}")
for date in buy_candidates:
    s     = score[date]
    t     = trend[date]
    t_str = "UP" if t==1 else "DOWN" if t==-1 else "FLAT"
    pen_s = s * penalty if t == -1 else s
    fires = "✅ YES" if pen_s > threshold else "❌ NO"
    # Which signals voted BUY on this day?
    voters = [name for name, sig in sigs.items() if w.get(name,0)>0 and sig.reindex(idx).get(date,0)==1]
    print(f"  {str(date.date()):<12} {s:>+7.4f}  {t_str:>6}  {pen_s:>+10.4f}  {fires:>7}  {', '.join(voters)}")

print(f"\n{'═'*70}")
print(f"  CANDIDATE SELL DAYS (score < -{threshold} before trend filter)")
print(f"{'═'*70}")
print(f"  {'Date':<12} {'Score':>7}  {'Trend':>6}  {'Penalised':>10}  {'Fires?':>7}  Contributing signals")
print(f"  {'─'*12} {'─'*7}  {'─'*6}  {'─'*10}  {'─'*7}  {'─'*30}")
for date in sell_candidates:
    s     = score[date]
    t     = trend[date]
    t_str = "UP" if t==1 else "DOWN" if t==-1 else "FLAT"
    pen_s = s * penalty if t == 1 else s
    fires = "✅ YES" if pen_s < -threshold else "❌ NO"
    voters = [name for name, sig in sigs.items() if w.get(name,0)>0 and sig.reindex(idx).get(date,0)==-1]
    print(f"  {str(date.date()):<12} {s:>+7.4f}  {t_str:>6}  {pen_s:>+10.4f}  {fires:>7}  {', '.join(voters)}")

print(f"\n  Confirmation window: {profile['confirmation_window']} days")
print(f"  (A signal must appear on {profile['confirmation_window']} consecutive days to be confirmed)\n")