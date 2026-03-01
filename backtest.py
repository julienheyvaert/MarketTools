# backtest.py
# ─────────────────────────────────────────────────────────────────────────────
# Optimization priorities (in order):
#   1. Win rate       — % of signals that were correct
#   2. Avg return     — average price move after signal
#   3. Signal count   — fewer signals preferred (quality over quantity)
#   4. No streaks     — penalise long losing streaks
# ─────────────────────────────────────────────────────────────────────────────
import itertools
import numpy as np
import pandas as pd
from fetch_data import fetch_data
from signals import combined_signal
from profiles import load_profiles, update_profile, ask_profile


def backtest(close, high, low, volume, profile, forward_window):
    """
    Evaluate signal quality on historical data.
    Returns detailed metrics for each BUY/SELL signal.
    """
    result     = combined_signal(close, high, low, volume=volume, profile=profile)
    buy_dates  = result[result["decision"] == "BUY"].index
    sell_dates = result[result["decision"] == "SELL"].index

    if len(buy_dates) + len(sell_dates) == 0:
        return None

    records = []

    for date in buy_dates:
        i = close.index.get_loc(date)
        if i + forward_window >= len(close):
            continue
        entry  = close.iloc[i]
        exit   = close.iloc[i + forward_window]
        ret    = (exit - entry) / entry     # % return
        correct = ret > 0
        records.append({
            "type":    "BUY",
            "date":    date,
            "return":  ret,
            "correct": correct
        })

    for date in sell_dates:
        i = close.index.get_loc(date)
        if i + forward_window >= len(close):
            continue
        entry  = close.iloc[i]
        exit   = close.iloc[i + forward_window]
        ret    = (entry - exit) / entry     # % return (profit if price drops)
        correct = ret > 0
        records.append({
            "type":    "SELL",
            "date":    date,
            "return":  ret,
            "correct": correct
        })

    if not records:
        return None

    df = pd.DataFrame(records).sort_values("date")
    return df


def score_results(df, forward_window):
    """
    Compute a composite score based on priorities:
      1. Win rate       (weight: 50%)
      2. Avg return     (weight: 30%)
      3. Signal penalty (weight: 15%) — heavily penalise too many signals
      4. Streak penalty (weight:  5%) — penalise long losing streaks
    """
    if df is None or len(df) == 0:
        return -1, {}

    n         = len(df)
    win_rate  = df["correct"].mean()
    avg_ret   = df["return"].mean()

    # ── Signal count penalty ───────────────────────────────
    # Ideal: 5-15 signals per year. Heavy penalty above 30.
    signals_per_year = n / (forward_window / 252 * 252) * 252 / forward_window
    if n <= 5:
        signal_penalty = 0.5      # too few signals — unreliable
    elif n <= 15:
        signal_penalty = 1.0      # ideal range
    elif n <= 30:
        signal_penalty = 0.7      # acceptable
    elif n <= 50:
        signal_penalty = 0.4      # too many
    else:
        signal_penalty = 0.1      # way too many — mostly noise

    # ── Streak penalty ─────────────────────────────────────
    # Find longest losing streak
    streak     = 0
    max_streak = 0
    for correct in df["correct"]:
        if not correct:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    if max_streak <= 2:
        streak_factor = 1.0
    elif max_streak <= 4:
        streak_factor = 0.85
    elif max_streak <= 6:
        streak_factor = 0.65
    else:
        streak_factor = 0.40

    # ── Composite score ────────────────────────────────────
    # Normalize avg_ret: cap at ±20% per trade
    norm_ret = max(min(avg_ret, 0.20), -0.20) / 0.20

    composite = (
        win_rate   * 0.50 +
        norm_ret   * 0.30 +
        0.0        * 0.15 +   # placeholder, multiplied below
        0.0        * 0.05     # placeholder, multiplied below
    ) * signal_penalty * streak_factor

    metrics = {
        "win_rate":      win_rate,
        "avg_return":    avg_ret,
        "signals":       n,
        "max_streak":    max_streak,
        "signal_factor": signal_penalty,
        "streak_factor": streak_factor,
        "composite":     composite
    }

    return composite, metrics


def optimize(close, high, low, volume, profile, forward_window):
    """
    Grid search over threshold and trend_penalty.
    Scored by composite metric prioritising win rate and avg return.
    """
    thresholds   = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
    penalties    = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    total_combos = len(thresholds) * len(penalties)

    best_score  = -1
    best_params = {}
    best_metrics = {}
    results     = []
    done        = 0

    print(f"\n  Testing {total_combos} parameter combinations...")
    print(f"  Forward window: {forward_window} days\n")

    for threshold, penalty in itertools.product(thresholds, penalties):
        test_profile = profile.copy()
        test_profile["weights"] = profile["weights"].copy()
        test_profile["score_threshold"] = threshold
        test_profile["trend_penalty"]   = penalty

        df = backtest(close, high, low, volume, test_profile, forward_window)
        composite, metrics = score_results(df, forward_window)

        done += 1
        print(f"  [{done:>3}/{total_combos}] "
              f"threshold={threshold:.2f}  penalty={penalty:.1f}  → ", end="")

        if composite < 0 or df is None:
            print("not enough signals")
            continue

        results.append({
            "threshold":      threshold,
            "penalty":        penalty,
            "win_rate":       metrics["win_rate"],
            "avg_return":     metrics["avg_return"],
            "signals":        metrics["signals"],
            "max_streak":     metrics["max_streak"],
            "signal_factor":  metrics["signal_factor"],
            "streak_factor":  metrics["streak_factor"],
            "composite":      composite
        })

        print(f"win={metrics['win_rate']:.1%}  "
              f"avg_ret={metrics['avg_return']:+.2%}  "
              f"signals={metrics['signals']}  "
              f"streak={metrics['max_streak']}  "
              f"score={composite:.4f}")

        if composite > best_score:
            best_score   = composite
            best_params  = {"score_threshold": threshold, "trend_penalty": penalty}
            best_metrics = metrics

    print()
    df_results = (
        pd.DataFrame(results).sort_values("composite", ascending=False)
        if results else pd.DataFrame()
    )
    return best_params, best_metrics, df_results


def print_results(ticker, profile_key, profile, best_params, best_metrics, df):
    """Print a detailed summary of the backtest results."""
    print("\n" + "═" * 60)
    print(f"  📊 BACKTEST RESULTS — {ticker} | {profile_key}")
    print("═" * 60)
    print(f"  Forward window:  {profile['forward_window']} days")
    print(f"  Combinations:    {len(df)} tested")

    if best_params and best_metrics:
        print(f"\n  ── BEST PARAMETERS ──────────────────────────────")
        print(f"  score_threshold: {best_params['score_threshold']}")
        print(f"  trend_penalty:   {best_params['trend_penalty']}")
        print(f"\n  ── PERFORMANCE ──────────────────────────────────")
        print(f"  Win rate:        {best_metrics['win_rate']:.1%}")
        print(f"  Avg return:      {best_metrics['avg_return']:+.2%} per signal")
        print(f"  Signals:         {best_metrics['signals']}")
        print(f"  Max losing streak: {best_metrics['max_streak']}")
        print(f"  Composite score: {best_metrics['composite']:.4f}")

    if not df.empty:
        print(f"\n  ── TOP 5 COMBINATIONS ───────────────────────────")
        print(f"  {'Thresh':>6}  {'Penalty':>7}  {'Win%':>6}  "
              f"{'AvgRet':>7}  {'Signals':>7}  {'Streak':>6}  {'Score':>7}")
        print(f"  {'─'*6}  {'─'*7}  {'─'*6}  "
              f"{'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}")
        for _, row in df.head(5).iterrows():
            print(f"  {row['threshold']:>6.2f}  {row['penalty']:>7.1f}  "
                  f"{row['win_rate']:>6.1%}  {row['avg_return']:>+7.2%}  "
                  f"{int(row['signals']):>7}  {int(row['max_streak']):>6}  "
                  f"{row['composite']:>7.4f}")

    print("═" * 60)


if __name__ == "__main__":
    # ─── Select profile ────────────────────────────────────
    profile_key, profile = ask_profile()
    forward_window = profile["forward_window"]

    # ─── Select ticker ─────────────────────────────────────
    ticker = input("  Enter ticker (e.g. BTC-USD): ").strip().upper()
    if not ticker:
        ticker = "BTC-USD"
        print(f"  No ticker entered, using default: {ticker}")

    period = profile["default_period"]
    print(f"\n  Fetching {period} of data for {ticker}...")

    # ─── Fetch data ────────────────────────────────────────
    data = fetch_data(ticker, period=period)
    if data is None:
        print("❌ Could not fetch data.")
        exit()

    close  = data["Close"].squeeze()
    high   = data["High"].squeeze()
    low    = data["Low"].squeeze()
    volume = data["Volume"].squeeze()

    # ─── Run optimization ──────────────────────────────────
    best_params, best_metrics, results_df = optimize(
        close, high, low, volume, profile, forward_window
    )

    if not best_params:
        print("❌ No valid combinations found — try a longer period or different profile.")
        exit()

    # ─── Print results ─────────────────────────────────────
    print_results(ticker, profile_key, profile, best_params, best_metrics, results_df)

    # ─── Auto update profiles.json ─────────────────────────
    confirm = input("\n  Update profiles.json automatically? (y/n, default: y): ").strip().lower()
    if confirm in ("", "y", "yes"):
        update_profile(profile_key, best_params)
    else:
        print(f"\n  Skipped. Best params for '{profile_key}':")
        print(f"  score_threshold: {best_params['score_threshold']}")
        print(f"  trend_penalty:   {best_params['trend_penalty']}\n")