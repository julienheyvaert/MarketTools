import numpy as np
import pandas as pd

def to_series(data):
    if isinstance(data, list):
        return pd.Series(data)
    if not isinstance(data, pd.Series):
        raise ValueError("Error, input must be a pandas.Series or a python list.")
    return data

def derivative_window(series, window=3):
    series = to_series(series)
    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")
    return series.diff().rolling(window=window).mean()

def since_critical(series, check_values):
    """
    Count steps since last critical value.
    ex: series=[3,6,8,1,9,4,3,5,4,1,5,6], check_values=[1,6]
    output=    [NaN,0,1,0,1,2,3,4,5,0,1,0]
    """
    series = to_series(series)
    if not check_values:
        raise ValueError("Error: check_values is empty.")

    n = len(series)
    count = np.nan
    since_critical_result = pd.Series(np.nan, index=series.index)

    for i in range(n):
        if series.iloc[i] in check_values:
            count = 0
        elif pd.notna(count):
            count += 1
        since_critical_result.iloc[i] = count

    return since_critical_result

def moving_average(series, window=21, exponential=False):
    if isinstance(series, list):
        series = pd.Series(series)

    if not isinstance(series, pd.Series):
        raise ValueError("Error, series must be a pandas.Series or a python list.")
    
    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")
    
    if len(series) <= window:
        raise ValueError("Not enough data for window value.")
    
    if exponential:
        return series.ewm(span=window, adjust=False).mean()
    else:
        return series.rolling(window=window).mean()

def crossovers(series1, series2):
    """
    Output:
        1  → series1 crosses ABOVE series2
       -1  → series1 crosses BELOW series2
        0  → no crossover
    """
    series1 = to_series(series1)
    series2 = to_series(series2)

    if len(series1) != len(series2):
        raise ValueError("The series provided have different lengths.")

    delta = series1 - series2

    above = (delta > 0).astype(int)
    prev  = above.shift(1)

    crossover = (above - prev).fillna(0)

    return crossover.astype(int)

def find_extremums(series, window=5):
    series = to_series(series)

    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")

    extremums = pd.DataFrame({"extremum_value": np.nan, "extremum_type": 0}, index=series.index)

    # center=False — only looks at past bars, works correctly on recent data
    rolling_min = series.rolling(window=window, center=False).min()
    rolling_max = series.rolling(window=window, center=False).max()

    minima_mask = (series == rolling_min) & (series.notna())
    extremums.loc[minima_mask, "extremum_value"] = series[minima_mask]
    extremums.loc[minima_mask, "extremum_type"] = -1

    maxima_mask = (series == rolling_max) & (series.notna())
    extremums.loc[maxima_mask, "extremum_value"] = series[maxima_mask]
    extremums.loc[maxima_mask, "extremum_type"] = 1

    return extremums

def rsi(series, period=14):
    series = to_series(series)

    if not isinstance(period, int):
         raise ValueError('Error, period must be an integer.')
    
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def macd(prices, short_ema_period=12, long_ema_period=26, signal_line_period=9):
    """
    input : 
        - prices : pandas.series or python list.
        - short_ema_period, long_ema_period, signal_line_period : integers.
    
    output:
        - macd_df : pandas.DataFrame,
            -- ['macd'] = macd value
            -- [macd_signal_line] = macd signal line value
            -- [macd_hist] = macd - signal line
    """
    prices = to_series(prices)
    
    macd_short_ema = moving_average(prices, window=short_ema_period, exponential=True)
    macd_long_ema = moving_average(prices, window=long_ema_period, exponential=True)
    macd = macd_short_ema - macd_long_ema
    
    macd_signal_line = moving_average(macd, window=signal_line_period, exponential=True)
    macd_hist = macd - macd_signal_line

    macd_df = pd.DataFrame({
        'macd': macd,
        'macd_signal_line': macd_signal_line,
        'macd_hist': macd_hist
    })

    return macd_df

def bollinger_bands(series, window=20, num_std_dev=2, proximity_threshold=0.02):
    series = to_series(series)
    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")
    if not isinstance(num_std_dev, int):
        raise ValueError("Error, num_std_dev must be an integer.")
    if not isinstance(proximity_threshold, float):
        raise ValueError("Error, proximity_threshold must be a float.")

    bb_sma = moving_average(series, window=window)
    bb_sd = series.rolling(window=window).std(ddof=0)

    bb_upper_band = bb_sma + bb_sd * num_std_dev
    bb_lower_band = bb_sma - bb_sd * num_std_dev

    conditions = [
        series >= bb_upper_band * (1 - proximity_threshold),
        series <= bb_lower_band * (1 + proximity_threshold)
    ]
    bb_price_proximity = np.select(conditions, [1, -1], default=0)

    bb_upper_crossover = crossovers(series, bb_upper_band)
    bb_lower_crossover = crossovers(series, bb_lower_band)

    bb_up_slope = derivative_window(bb_upper_band)
    bb_low_slope = derivative_window(bb_lower_band)

    bb_squeeze = -find_extremums(bb_upper_band)["extremum_type"]

    return pd.DataFrame({
        "bb_upper_band": bb_upper_band,
        "bb_lower_band": bb_lower_band,
        "bb_price_proximity": bb_price_proximity,
        "bb_upper_crossover": bb_upper_crossover,
        "bb_lower_crossover": bb_lower_crossover,
        "bb_up_slope": bb_up_slope,
        "bb_low_slope": bb_low_slope,
        "bb_squeeze": bb_squeeze
    })

def stochastic_oscillator(close, high, low, K_length=14, K_smoothing=1, D_smoothing=3):
    """
    input:
        - close, high, low: pandas.Series
        - K_length, K_smoothing, D_smoothing: int
    output:
        - stoch_df: pandas.DataFrame with %K and %D columns
    """
    close = to_series(close)
    high = to_series(high)
    low = to_series(low)

    for param, name in [(K_length, "K_length"), (K_smoothing, "K_smoothing"), (D_smoothing, "D_smoothing")]:
        if not isinstance(param, int):
            raise ValueError(f"Error, {name} must be an integer.")

    if not (len(close) == len(high) == len(low)):
        raise ValueError("close, high and low must have the same length.")

    low_min = low.rolling(window=K_length).min()
    high_max = high.rolling(window=K_length).max()

    stoch_k = (close - low_min) / (high_max - low_min) * 100
    stoch_k_smooth = stoch_k.rolling(window=K_smoothing).mean()
    stoch_d = stoch_k_smooth.rolling(window=D_smoothing).mean()

    return pd.DataFrame({
        "%K": stoch_k_smooth,
        "%D": stoch_d
    })

def atr(close, high, low, window=14):
    """
    Calculate Average True Range (ATR).
    Measures market volatility.
    """
    close = to_series(close)
    high = to_series(high)
    low = to_series(low)

    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")
    if not (len(close) == len(high) == len(low)):
        raise ValueError("close, high and low must have the same length.")

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.ewm(alpha=1/window, adjust=False).mean()

def cmf(close, high, low, volumes, window=21):
    """
    Chaikin Money Flow (CMF).
    Measures buying and selling pressure using price and volume.
    CMF > 0 = money flowing in, CMF < 0 = money flowing out.
    """
    close = to_series(close)
    high = to_series(high)
    low = to_series(low)
    volumes = to_series(volumes)

    if not isinstance(window, int):
        raise ValueError("Error, window must be an integer.")
    if not (len(close) == len(high) == len(low) == len(volumes)):
        raise ValueError("Error, all series must have the same length.")

    clv = ((close - low) - (high - close)) / (high - low)

    clv_volume = clv * volumes
    result = clv_volume.rolling(window=window).sum() / volumes.rolling(window=window).sum()

    return result

def psar(close, high, low, start_af=0.02, increment=0.02, max_af=0.2):
    """
    Parabolic SAR.
    Dots below price = uptrend, dots above price = downtrend.
    Signal of 1 = trend turned bullish, -1 = trend turned bearish.
    """
    close = to_series(close)
    high = to_series(high)
    low = to_series(low)

    if not (len(close) == len(high) == len(low)):
        raise ValueError("close, high and low must have the same length.")
    if not isinstance(start_af, float) or not isinstance(increment, float) or not isinstance(max_af, float):
        raise ValueError("start_af, increment and max_af must be floats.")

    n = len(close)
    psar_values = np.zeros(n)
    psar_values[0] = low.iloc[0]
    trend = 1
    ep = high.iloc[0]
    af = start_af
    psar_position = np.zeros(n)

    for i in range(1, n):
        psar_values[i] = psar_values[i-1] + af * (ep - psar_values[i-1])

        if trend == 1:
            if psar_values[i] > low.iloc[i]:
                psar_values[i] = ep
                trend = -1
                ep = low.iloc[i]
                af = start_af
                psar_position[i] = -1
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + increment, max_af)
        else:
            if psar_values[i] < high.iloc[i]:
                psar_values[i] = ep
                trend = 1
                ep = high.iloc[i]
                af = start_af
                psar_position[i] = 1
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + increment, max_af)

    return pd.DataFrame({
        "psar": psar_values,
        "psar_position": psar_position
    }, index=close.index)