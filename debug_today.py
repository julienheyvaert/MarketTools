"""
Shows exactly what signal the tool would give TODAY.
Run from MarketAnalysis directory: python3 debug_today.py
"""
import pandas as pd
from fetch_data import fetch_data
from profiles import load_profiles
from indicators import moving_average, rsi, macd, psar, bollinger_bands, atr, derivative_window
from signals import combined_signal

TICKER      = "BTC-USD"
PROFILE_KEY = "swing"

profiles = load_profiles()
profile  = profiles[PROFILE_KEY]

data   = fetch_data(TICKER, period=profile["default_period"])
close  = data["Close"].squeeze()
high   = data["High"].squeeze()
low    = data["Low"].squeeze()
volume = data["Volume"].squeeze()

# ── Raw indicator values on last day ───────────────────────
rsi_val  = rsi(close, period=profile["rsi_period"])
macd_df  = macd(close)
psar_df  = psar(close, high, low)
ma_short = moving_average(close, window=profile["ma_short_window"], exponential=profile["ma_exponential"])
ma_long  = moving_average(close, window=profile["ma_long_window"],  exponential=profile["ma_exponential"])
ma_trend = moving_average(close, window=profile["trend_filter_window"])

# ── Final combined signal ──────────────────────────────────
result   = combined_signal(close, high, low, volume=volume, profile=profile)
last     = result.iloc[-1]
decision = last["decision"]
score    = last["score"]

print(f"\n{'═'*55}")
print(f"  📊 TODAY'S SIGNAL — {TICKER}")
print(f"{'═'*55}")
print(f"  Date:         {close.index[-1].date()}")
print(f"  Price:        ${close.iloc[-1]:,.2f}")
print(f"{'─'*55}")
print(f"  INDICATOR VALUES:")
print(f"  RSI({profile['rsi_period']}):      {rsi_val.iloc[-1]:.1f}  "
      f"(oversold<{profile['rsi_oversold']}  overbought>{profile['rsi_overbought']})")
print(f"  MACD:         {macd_df['macd'].iloc[-1]:+.1f}  "
      f"Signal: {macd_df['macd_signal_line'].iloc[-1]:+.1f}  "
      f"→ {'BULLISH' if macd_df['macd'].iloc[-1] > macd_df['macd_signal_line'].iloc[-1] else 'BEARISH'}")
print(f"  PSAR:         {psar_df['psar'].iloc[-1]:,.0f}  "
      f"→ {'BELOW price (uptrend)' if psar_df['psar'].iloc[-1] < close.iloc[-1] else 'ABOVE price (downtrend)'}")
print(f"  MA{profile['ma_short_window']}:        {ma_short.iloc[-1]:,.0f}")
print(f"  MA{profile['ma_long_window']}:        {ma_long.iloc[-1]:,.0f}  "
      f"→ {'MA short ABOVE long (bullish)' if ma_short.iloc[-1] > ma_long.iloc[-1] else 'MA short BELOW long (bearish)'}")
print(f"  Trend MA{profile['trend_filter_window']}: {ma_trend.iloc[-1]:,.0f}  "
      f"→ {'Price ABOVE (uptrend)' if close.iloc[-1] > ma_trend.iloc[-1] else 'Price BELOW (downtrend)'}")
print(f"{'─'*55}")
print(f"  Score:        {score:+.4f}  (threshold ±{profile['score_threshold']})")

if decision == "BUY":
    print(f"  Decision:     BUY 🟢")
elif decision == "SELL":
    print(f"  Decision:     SELL 🔴")
else:
    print(f"  Decision:     HOLD ⚪ — no strong signal today")
    # Explain why
    if abs(score) < profile["score_threshold"]:
        gap = profile["score_threshold"] - abs(score)
        print(f"  (score needs {gap:.3f} more to trigger a signal)")

print(f"{'═'*55}\n")