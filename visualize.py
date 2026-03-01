import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from indicators import moving_average, rsi, macd, bollinger_bands
from signals import combined_signal

BG_COLOR   = "#1b263b"
TEXT_COLOR = "white"


def _safe_indicator(fn, *args, **kwargs):
    """Return None instead of crashing if there's not enough data."""
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return None


def plot(close, high, low, volume, ticker="BTC-USD", save_path=None, profile=None):
    """
    Plot price, moving averages, bollinger bands, buy/sell signals, RSI, MACD and volume.
    Indicators are skipped gracefully if there is not enough data.
    """
    n = len(close)

    # ─── Global text color ─────────────────────────────────
    plt.rcParams.update({
        "text.color":       TEXT_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
    })

    # ─── Calculate indicators (safely) ─────────────────────
    ma_short_w = profile["ma_short_window"] if profile else 20
    ma_long_w  = profile["ma_long_window"]  if profile else 50

    ma_short   = _safe_indicator(moving_average, close, window=ma_short_w)
    ma_long    = _safe_indicator(moving_average, close, window=ma_long_w)
    rsi_values = _safe_indicator(rsi, close)
    macd_df    = _safe_indicator(macd, close)
    bb         = _safe_indicator(bollinger_bands, close)
    signals    = combined_signal(close, high, low, profile=profile)

    buy_signals  = signals[signals["decision"] == "BUY"].index
    sell_signals = signals[signals["decision"] == "SELL"].index

    # ─── Layout ────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{ticker} Market Analysis", fontsize=16, fontweight="bold", color=TEXT_COLOR)

    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.4)

    ax_price  = fig.add_subplot(gs[0])
    ax_rsi    = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd   = fig.add_subplot(gs[2], sharex=ax_price)
    ax_volume = fig.add_subplot(gs[3], sharex=ax_price)

    # ─── Price ─────────────────────────────────────────────
    ax_price.plot(close, color="white", linewidth=1.2, label="Price")

    if ma_short is not None:
        ax_price.plot(ma_short, color="orange", linewidth=1.0, label=f"MA{ma_short_w}")
    if ma_long is not None:
        ax_price.plot(ma_long,  color="cyan",   linewidth=1.0, label=f"MA{ma_long_w}")
    if bb is not None:
        ax_price.plot(bb["bb_upper_band"], color="gray", linewidth=0.8, linestyle="--", label="BB Upper")
        ax_price.plot(bb["bb_lower_band"], color="gray", linewidth=0.8, linestyle="--", label="BB Lower")
        ax_price.fill_between(close.index, bb["bb_upper_band"], bb["bb_lower_band"], alpha=0.1, color="gray")

    # ─── Buy/Sell signals ──────────────────────────────────
    ax_price.scatter(buy_signals,  close[buy_signals],  marker="^", color="lime", s=100, zorder=5, label="BUY")
    ax_price.scatter(sell_signals, close[sell_signals], marker="v", color="red",  s=100, zorder=5, label="SELL")

    ax_price.set_ylabel("Price (USD)")
    legend = ax_price.legend(loc="upper left", fontsize=8)
    legend.get_frame().set_facecolor(BG_COLOR)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # ─── RSI ───────────────────────────────────────────────
    if rsi_values is not None:
        ax_rsi.plot(rsi_values, color="purple", linewidth=1.0)
        ax_rsi.axhline(70, color="red",   linewidth=0.8, linestyle="--")
        ax_rsi.axhline(30, color="green", linewidth=0.8, linestyle="--")
        ax_rsi.fill_between(close.index, rsi_values, 70, where=(rsi_values >= 70), alpha=0.3, color="red")
        ax_rsi.fill_between(close.index, rsi_values, 30, where=(rsi_values <= 30), alpha=0.3, color="green")
        ax_rsi.set_ylim(0, 100)
    else:
        ax_rsi.text(0.5, 0.5, "RSI — not enough data", transform=ax_rsi.transAxes,
                    ha="center", va="center", color=TEXT_COLOR, fontsize=9)
    ax_rsi.set_ylabel("RSI")

    # ─── MACD ──────────────────────────────────────────────
    if macd_df is not None:
        ax_macd.plot(macd_df["macd"],             color="cyan",   linewidth=1.0, label="MACD")
        ax_macd.plot(macd_df["macd_signal_line"], color="orange", linewidth=1.0, label="Signal")
        ax_macd.bar(close.index, macd_df["macd_hist"],
                    color=["lime" if v >= 0 else "red" for v in macd_df["macd_hist"]],
                    alpha=0.5, label="Histogram")
        ax_macd.axhline(0, color="white", linewidth=0.5)
        legend = ax_macd.legend(loc="upper left", fontsize=8)
        legend.get_frame().set_facecolor(BG_COLOR)
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
    else:
        ax_macd.text(0.5, 0.5, "MACD — not enough data", transform=ax_macd.transAxes,
                     ha="center", va="center", color=TEXT_COLOR, fontsize=9)
    ax_macd.set_ylabel("MACD")

    # ─── Volume ────────────────────────────────────────────
    colors = ["lime" if close.iloc[i] >= close.iloc[i-1] else "red" for i in range(len(close))]
    ax_volume.bar(close.index, volume, color=colors, alpha=0.7)
    ax_volume.set_ylabel("Volume")

    # ─── Global styling ────────────────────────────────────
    for ax in [ax_price, ax_rsi, ax_macd, ax_volume]:
        ax.set_facecolor(BG_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)

    fig.patch.set_facecolor(BG_COLOR)
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"✅ Chart saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from fetch_data import fetch_data

    data   = fetch_data("BTC-USD", start="2026-01-01")
    close  = data["Close"].squeeze()
    high   = data["High"].squeeze()
    low    = data["Low"].squeeze()
    volume = data["Volume"].squeeze()

    plot(close, high, low, volume)