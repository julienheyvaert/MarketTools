import os
from fetch_data import fetch_data
from signals import combined_signal
from visualize import plot
from profiles import ask_profile, load_profiles, update_profile
from backtest import optimize, print_results
from indicators import moving_average, rsi, macd, psar

VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
OUTPUT_DIR    = "outputs"


def print_summary(ticker, result, close, profile, high, low, volume):
    last      = result.iloc[-1]
    score     = last["score"]
    decision  = last["decision"]
    bar       = "█" * int(abs(score) * 30)
    sign      = "+" if score > 0 else ""

    buy_count     = (result["decision"] == "BUY").sum()
    sell_count    = (result["decision"] == "SELL").sum()
    neutral_count = (result["decision"] == "NEUTRAL").sum()

    # ── Indicator states ───────────────────────────────────
    rsi_val  = rsi(close, period=profile["rsi_period"]).iloc[-1]
    macd_df  = macd(close)
    psar_df  = psar(close, high, low)
    ma_short = moving_average(close, window=profile["ma_short_window"],
                              exponential=profile["ma_exponential"]).iloc[-1]
    ma_long  = moving_average(close, window=profile["ma_long_window"],
                              exponential=profile["ma_exponential"]).iloc[-1]
    ma_trend = moving_average(close, window=profile["trend_filter_window"]).iloc[-1]

    bullish = []
    bearish = []

    if rsi_val < profile["rsi_oversold"]:
        bullish.append(f"RSI oversold ({rsi_val:.0f})")
    elif rsi_val > profile["rsi_overbought"]:
        bearish.append(f"RSI overbought ({rsi_val:.0f})")
    else:
        neutral_rsi = f"RSI neutral ({rsi_val:.0f})"

    if macd_df["macd"].iloc[-1] > macd_df["macd_signal_line"].iloc[-1]:
        bullish.append("MACD above signal line")
    else:
        bearish.append("MACD below signal line")

    if psar_df["psar"].iloc[-1] < close.iloc[-1]:
        bullish.append("PSAR below price (uptrend)")
    else:
        bearish.append("PSAR above price (downtrend)")

    if ma_short > ma_long:
        bullish.append(f"MA{profile['ma_short_window']} above MA{profile['ma_long_window']}")
    else:
        bearish.append(f"MA{profile['ma_short_window']} below MA{profile['ma_long_window']}")

    if close.iloc[-1] > ma_trend:
        bullish.append(f"Price above MA{profile['trend_filter_window']} (uptrend)")
    else:
        bearish.append(f"Price below MA{profile['trend_filter_window']} (downtrend)")

    print("\n" + "═" * 55)
    print(f"  📊 MARKET ANALYSIS — {ticker}")
    print("═" * 55)
    print(f"  Style:           {profile['label']}")
    print(f"  Period:          {result.index[0].date()} → {result.index[-1].date()}")
    print(f"  Current price:   ${close.iloc[-1]:,.2f}")
    print("─" * 55)

    if decision == "BUY":
        print(f"  TODAY'S SIGNAL:  BUY 🟢")
    elif decision == "SELL":
        print(f"  TODAY'S SIGNAL:  SELL 🔴")
    else:
        print(f"  TODAY'S SIGNAL:  HOLD ⚪")

    print(f"  Score:           {sign}{score:.3f}  {bar}")
    print(f"  Threshold:       ±{profile['score_threshold']}")
    print("─" * 55)
    print(f"  BULLISH factors ({len(bullish)}):")
    for b in bullish:
        print(f"    ✅ {b}")
    print(f"  BEARISH factors ({len(bearish)}):")
    for b in bearish:
        print(f"    ❌ {b}")
    print("─" * 55)
    print(f"  BUY signals:     {buy_count} days")
    print(f"  SELL signals:    {sell_count} days")
    print(f"  NEUTRAL:         {neutral_count} days")
    print("═" * 55 + "\n")


def save_results(ticker, result):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{ticker}_signals.csv"
    result.to_csv(filename)
    print(f"✅ Results saved to {filename}")


def ask_user(default_period):
    print("\n" + "═" * 50)
    print("  📈 MARKET ANALYSIS TOOL")
    print("═" * 50)

    ticker = input("\n  Enter ticker (e.g. BTC-USD, ETH-USD): ").strip().upper()
    if not ticker:
        ticker = "BTC-USD"
        print(f"  No ticker entered, using default: {ticker}")

    print(f"\n  Available periods: {', '.join(VALID_PERIODS)}")
    print(f"  (Recommended for your style: {default_period})")
    while True:
        period = input(f"  Enter period (default: {default_period}): ").strip().lower()
        if not period:
            period = default_period
            break
        if period in VALID_PERIODS:
            break
        print(f"  ❌ '{period}' is invalid. Please choose from: {', '.join(VALID_PERIODS)}")

    return ticker, period


def run_backtest(ticker, close, high, low, volume, profile_key, profile):
    """Run backtest optimization and update profiles.json automatically."""
    forward_window = profile["forward_window"]

    print("\n" + "═" * 50)
    print("  🔬 RUNNING BACKTEST OPTIMIZATION")
    print("═" * 50)
    print(f"  Optimizing parameters for '{profile_key}' on {ticker}...")

    best_params, best_metrics, results_df = optimize(
        close, high, low, volume, profile, forward_window
    )

    if not best_params:
        print("  ⚠️  No valid combinations found — keeping current parameters.")
        return profile

    print_results(ticker, profile_key, profile, best_params, best_metrics, results_df)

    # Auto update profiles.json
    update_profile(profile_key, best_params)

    # Return updated profile
    updated_profiles = load_profiles()
    return updated_profiles[profile_key]


def main():
    # Step 1 — Choose trading style
    profile_key, profile = ask_profile()

    # Step 2 — Choose ticker and period
    ticker, period = ask_user(default_period=profile["default_period"])

    # Step 3 — Fetch data
    print(f"\n  Fetching data for {ticker}...")
    data = fetch_data(ticker, period=period)
    if data is None:
        print(f"❌ Could not fetch data for {ticker}.")
        return

    close  = data["Close"].squeeze()
    high   = data["High"].squeeze()
    low    = data["Low"].squeeze()
    volume = data["Volume"].squeeze()

    # Step 4 — Run backtest and auto-update profile
    profile = run_backtest(ticker, close, high, low, volume, profile_key, profile)

    # Step 5 — Calculate signals with optimized parameters
    print("\n  Calculating signals with optimized parameters...")
    result = combined_signal(close, high, low, volume=volume, profile=profile)

    # Step 6 — Print summary with today's signal and market context
    print_summary(ticker, result, close, profile, high, low, volume)

    # Step 7 — Save CSV
    save_results(ticker, result)

    # Step 8 — Plot and save chart
    print("  Generating chart...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"{ticker}_chart.png")
    plot(close, high, low, volume, ticker=ticker, save_path=save_path, profile=profile)


if __name__ == "__main__":
    main()