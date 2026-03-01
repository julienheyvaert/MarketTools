"""
Run this from your MarketAnalysis directory:
    python3 diagnose.py
"""
from fetch_data import fetch_data
from signals import *
from profiles import PROFILES

TICKER = "BTC-USD"
PERIOD = "2y"
PROFILE_KEY = "long"

profile = PROFILES[PROFILE_KEY]
data    = fetch_data(TICKER, period=PERIOD)
close   = data["Close"].squeeze()
high    = data["High"].squeeze()
low     = data["Low"].squeeze()
volume  = data["Volume"].squeeze()
n       = len(close)
idx     = close.index

print(f"\n{'═'*60}")
print(f"  SIGNAL DIAGNOSTICS — {TICKER} | {PERIOD} | {PROFILE_KEY}")
print(f"  {n} bars of data")
print(f"{'═'*60}")

checks = {
    "rsi":                (lambda: rsi_signal(close, period=profile["rsi_period"],
                               oversold=profile["rsi_oversold"],
                               overbought=profile["rsi_overbought"]), 0),
    "macd":               (lambda: macd_signal(close), 0),
    "bbands":             (lambda: bbands_signal(close), 0),
    "stochastic":         (lambda: stochastic_signal(close, high, low), 0),
    "psar":               (lambda: psar_signal(close, high, low), 0),
    "ma_crossover":       (lambda: ma_crossover_signal(close,
                               short_window=profile["ma_short_window"],
                               long_window=profile["ma_long_window"],
                               exponential=profile["ma_exponential"]), 0),
    "support_resistance": (lambda: support_resistance_signal(close,
                               window=profile.get("extremum_window", 5),
                               lookback=profile.get("sr_lookback", 20)), 0),
    "trend_strength":     (lambda: trend_strength_signal(close,
                               window=profile.get("extremum_window", 5)), 0),
    "squeeze_breakout":   (lambda: squeeze_breakout_signal(close, high, low), 0),
    "volume_reversal":    (lambda: volume_confirmed_reversal_signal(close, high, low, volume,
                               window=profile.get("extremum_window", 5),
                               cmf_threshold=profile.get("cmf_threshold", 0.05)), 0),
    "divergence":         (lambda: divergence_signal(close,
                               rsi_period=profile["rsi_period"],
                               window=profile.get("extremum_window", 5)), 0),
}

weights = profile["weights"]

print(f"\n  {'Signal':<22} {'Weight':>7}  {'BUY':>5}  {'SELL':>5}  {'Fire%':>6}")
print(f"  {'─'*22} {'─'*7}  {'─'*5}  {'─'*5}  {'─'*6}")

for name, (fn, _) in checks.items():
    w = weights.get(name, 0)
    if w == 0:
        print(f"  {name:<22} {'0.00':>7}  {'—':>5}  {'—':>5}  {'—':>6}")
        continue
    try:
        sig = fn()
        buys  = (sig ==  1).sum()
        sells = (sig == -1).sum()
        fire_pct = (buys + sells) / n * 100
        print(f"  {name:<22} {w:>7.2f}  {buys:>5}  {sells:>5}  {fire_pct:>5.1f}%")
    except Exception as e:
        print(f"  {name:<22} {w:>7.2f}  ERROR: {e}")

# Trend filter analysis
print(f"\n  --- Trend filter analysis (window={profile['trend_filter_window']}) ---")
try:
    ma  = moving_average(close, window=profile["trend_filter_window"])
    up  = (close > ma).sum()
    dn  = (close < ma).sum()
    print(f"  Days in uptrend:   {up} ({up/n*100:.1f}%) — SELL signals suppressed")
    print(f"  Days in downtrend: {dn} ({dn/n*100:.1f}%) — BUY  signals suppressed")
except:
    print(f"  Not enough data for trend filter")

# Score distribution without trend filter
print(f"\n  --- Score distribution (NO trend filter) ---")
raw_scores = pd.Series(0.0, index=idx)
active_w   = 0.0
for name, (fn, _) in checks.items():
    w = weights.get(name, 0)
    if w == 0: continue
    try:
        sig = fn()
        raw_scores += sig.reindex(idx, fill_value=0) * w
        active_w   += w
    except:
        pass
if active_w < 1.0:
    raw_scores = raw_scores / active_w
print(f"  min={raw_scores.min():.4f}  max={raw_scores.max():.4f}  mean={raw_scores.mean():.4f}")
above = (raw_scores >  profile["score_threshold"]).sum()
below = (raw_scores < -profile["score_threshold"]).sum()
print(f"  Days above +{profile['score_threshold']}: {above}  |  Days below -{profile['score_threshold']}: {below}")

# Actual result
print(f"\n  --- Actual result (with trend filter + confirmation) ---")
result = combined_signal(close, high, low, volume=volume, profile=profile)
print(f"  min={result['score'].min():.4f}  max={result['score'].max():.4f}")
print(f"  BUY: {(result['decision']=='BUY').sum()}  SELL: {(result['decision']=='SELL').sum()}  NEUTRAL: {(result['decision']=='NEUTRAL').sum()}")
print(f"{'═'*60}\n")