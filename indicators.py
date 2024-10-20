import numpy as np
import pandas as pd

def derivative_window(series, window=1):
    return series.diff().rolling(window=window).mean()

def since_critical(series, check_values):
    """
    Input : - series, a column of pandas dataframe or a list
            - check_values, python list of numeric values
    
    Output: - since_critical_result, a pandas dataframe (date, result).

    When a value in the series is the same as a value in the check_values list, 
    start counting the number of iterations since the last value that appears in the check_values list.

    ex: series = [3,6,8,1,9,4,3,5,4,1,5,6], check_values = [1,6]
    output = [nan,0,1,0,1,2,3,4,5,0,1,0]
    """
    
    if not check_values:
        raise ValueError('Error : check_values.')
    
    if isinstance(series, list):
            series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise ValueError("Error, series must be a pandas.Series or a python list.")

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

def find_extremums(series, window=5):
    """
    input : 
        - series, pandas.Series or a list
        - window, int
    
    output :
        - extremums, pandas.DataFrame :
            -- ['extremum_value'] : if is extremum, the price value, else : nan
            -- ['extremum_type'] : maximum = 1, minimum = -1, neither = 0
    """
    if isinstance(series, list):
            series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise ValueError("Error, series must be a pandas.Series or a python list.")
    if not isinstance(window, int):
        raise ValueError('Error, window must be an integer.')

    # Initialisation
    extremums = pd.DataFrame(index=series.index, columns=['extremum_value', 'extremum_type'])
    extremums['extremum_value'] = np.nan
    extremums['extremum_type'] = 0

    # Extremums (rolling window)
    rolling_min = series.rolling(window=window, center=True).min()
    rolling_max = series.rolling(window=window, center=True).max()

    # Identify local maxima
    maxima_mask = (series == rolling_max) & (series.notna())
    extremums.loc[maxima_mask, 'extremum_value'] = series[maxima_mask]
    extremums.loc[maxima_mask, 'extremum_type'] = 1

    # Identify local minima
    minima_mask = (series == rolling_min) & (series.notna())
    extremums.loc[minima_mask, 'extremum_value'] = series[minima_mask]
    extremums.loc[minima_mask, 'extremum_type'] = -1

    # last window
    last_window_max = series[-window:].max()
    last_window_min = series[-window:].min()
    last_value = series.iloc[-1]

    if last_window_max:
        extremums.loc[series.index[-1], 'extremum_value'] = last_value
        extremums.loc[series.index[-1], 'extremum_type'] = 1  # Maximum

    elif last_window_min:
        extremums.loc[series.index[-1], 'extremum_value'] = last_value
        extremums.loc[series.index[-1], 'extremum_type'] = -1

    return extremums

def moving_average(series, window = 21, exponential = False):
    """
    input : 
        - data : pandas.Series or a list.
        - window : int,
        - exponential : boolean
    
    output :
        - moving_average : pandas.Series
    """
    if isinstance(series, list):
            series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')

    if not isinstance(window, int):
        raise ValueError('Error, window must be an integer.')
    
    if len(series) <= window:
        raise ValueError('Not enough data for window value.')
    
    if exponential:
        moving_average = series.ewm(span=window, adjust=False).mean()
    else:
        moving_average = series.rolling(window=window).mean()
    return moving_average

def crossovers(series1, series2):
    """
    input :
        - series1, series2 : pandas.Series or list.
    output :
        - crossover_series : if series1 passes above series 2 = 1, ... = -1, 0 if no crossover
    """
    if isinstance(series1, list):
            series1 = pd.Series(series1)
    elif not isinstance(series1, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')
    
    if isinstance(series2, list):
            series2 = pd.Series(series2)
    elif not isinstance(series2, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')
    
    if len(series1) != len(series2):
        raise ValueError("The series provided have different lengths")

    delta = series1 - series2
    s1_above_s2 = np.where(delta > 0, 1, 0)
    crossover = np.diff(s1_above_s2, prepend=0)
    crossover_series = pd.Series(crossover, index=series1.index)

    return crossover_series

def rsi(series, period=14):
    """
    input:
        - series : pandas.Series or python list.
        - period : integer
    
    output:
        - rsi : pandas.Series
    """
    if isinstance(series, list):
            series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')
    
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

def macd(prices_data, short_ema_period=12, long_ema_period=26, signal_line_period=9):
    """
    input : 
        - prices_data : pandas.series or python list.
        - short_ema_period, long_ema_period, signal_line_period : integers.
    
    output:
        - macd_df : pandas.DataFrame,
            -- ['macd'] = macd value
            -- [macd_signal_line] = macd signal line value
            -- [macd_hist] = macd - signal line
    """
    if isinstance(prices_data, list):
            prices_data = pd.Series(prices_data)
    elif not isinstance(prices_data, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')
    
    macd_short_ema = moving_average(prices_data, window=short_ema_period, exponential=True)
    macd_long_ema = moving_average(prices_data, window=long_ema_period, exponential=True)
    macd = macd_short_ema - macd_long_ema
    
    macd_signal_line = moving_average(macd, window=signal_line_period, exponential=True)
    macd_hist = macd - macd_signal_line

    macd_df = pd.DataFrame({
        'macd': macd,
        'macd_signal_line': macd_signal_line,
        'macd_hist': macd_hist
    })

    return macd_df

def bollinger_bands(series, window=20, num_std_dev=2, proximity_threshold= 0.02):
    """
    input:
        - series : pandas.Series or python list.
        - window, num_std_dev : integers.
        - proximity_treshold : float.

    output:
        - bb_df = pandas.DataFrame
    """
    if isinstance(series, list):
            series = pd.Series(series)
    elif not isinstance(series, pd.Series):
        raise ValueError('Error, series must be a pandas.Series or a python List.')
    
    bb_sma = moving_average(series, window=window)
    bb_sd = series.rolling(window=window).std()

    # Bands Calculation
    bb_upper_band = bb_sma + bb_sd * num_std_dev
    bb_lower_band = bb_sma - bb_sd * num_std_dev

    # Bands vs Price
    bb_price_proximity = np.where(series >= bb_upper_band * (1 - proximity_threshold), 1, 0)
    bb_price_proximity = np.where(series <= bb_lower_band * (1 + proximity_threshold), -1, 0)

    # Bands vs Prices Crossover
    bb_price_crossover = crossovers(series, bb_upper_band)

    # Bands slopes
    bb_up_slope = derivative_window(bb_upper_band)
    bb_low_slope = derivative_window(bb_lower_band)

    # Bands squeeze
    bb_squeeze = -find_extremums(bb_upper_band)['extremum_type']

    bb_df = pd.DataFrame({
        'bb_upper_band': bb_upper_band,
        'bb_lower_band' : bb_lower_band,
        'bb_price_proximity' : bb_price_proximity,
        'bb_up_slope' : bb_up_slope,
        'bb_low_slope' : bb_low_slope,
        'bb_price_crossover' : bb_price_crossover,
        'bb_squeeze' : bb_squeeze
    })

    return bb_df

def psar(prices_close, prices_high, prices_low, start=0.02, increment=0.03, maximum=0.2):
    """
    input :
        - prices_close, prices_high, prices_low : pandas.Series or python list.
        - start, increment, maximum = float.
    
    output:
        - psar_df : pandas.DataFrame,
            -- ['psar'] = psar value.
            -- ['psar_position'] = 1 if bullish trend inversion, ... -1, O if no trend inversion.
    """
    if isinstance(prices_close, list):
         prices_close = pd.Series(prices_close)

    if isinstance(prices_high, list):
         prices_high = pd.Series(prices_high)

    if isinstance(prices_low, list):
         prices_low = pd.Series(prices_low)
    
    if not isinstance(prices_close, pd.Series) or not isinstance(prices_high, pd.Series) or not isinstance(prices_low, pd.Series):
        raise ValueError('Error, prices data must be pandas.Series or python lists.')
    
    if len(prices_close) != len(prices_high) != len(prices_low):
        raise ValueError('Error, prices data must have the same length.')
    
    # Initialization
    n = len(prices_close)
    psar = np.zeros(n)
    ep = prices_high.iloc[0]
    af = start
    bullish = True
    psar[0] = prices_low.iloc[0]
    psar_position = np.zeros(n)

    for i in range(1, n):
        if bullish:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])

            # ep, af update
            if prices_high.iloc[i] > ep:
                ep = prices_high.iloc[i]
                af = min(af + increment, maximum)

            # Trend inversion
            if prices_close.iloc[i] < psar[i] and psar_position[i-1] != -1:
                bullish = False
                psar[i] = ep
                ep = prices_low.iloc[i]
                af = start
                psar_position[i] = -1

        else:
            psar[i] = psar[i - 1] + af * (ep - psar[i - 1])

            # ep, af update
            if prices_low.iloc[i] < ep:
                ep = prices_low.iloc[i]
                af = min(af + increment, maximum)

            # Trend inversion
            if prices_close.iloc[i] > psar[i] and psar_position[i-1] != 1:
                bullish = True
                psar[i] = ep
                ep = prices_high.iloc[i]
                af = start
                psar_position[i] = 1
    
    psar_df = pd.DataFrame({
        'psar': psar,
        'psar_position' : psar_position
    },index=prices_close.index)
    return psar_df

def stochastic_oscillator(close, high, low, K_length=14, K_smoothing=1, D_smoothing=3):
    """
    input:
        - close, high, low = pandas.DataFrame.
        - K_length, K_smoothing, D_smoothing = int.

    output:
        - stoch_df = pandas.DataFrame
    """
    if not isinstance(close, pd.Series) or not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise ValueError("Inputs must be pandas.Series for close, high, and low prices.")
    
    if len(close) != len(high) or len(close) != len(low):
        raise ValueError("The series must have the same length.")

    low_min = low.rolling(window=K_length).min()
    high_max = high.rolling(window=K_length).max()
    
    # %K
    stoch_k = (close - low_min) / (high_max - low_min) * 100
    stoch_k_smooth = stoch_k.rolling(window=K_smoothing).mean()

    # %D
    stoch_d = stoch_k_smooth.rolling(window=D_smoothing).mean()
    
    stoch_df = pd.DataFrame({
        '%K': stoch_k_smooth,
        '%D': stoch_d
    }).dropna()
    
    return stoch_df

def atr(prices_close, prices_high, prices_low, window=14):
    if not isinstance(prices_close, pd.Series) or not isinstance(prices_close, pd.Series) \
        or not isinstance(prices_low, pd.Series):
        raise ValueError('Error, prices data must be pandas.Series.')
    # TR
    tr1 = prices_high - prices_low
    tr2 = (prices_high - prices_close.shift(1)).abs()
    tr3 = (prices_low - prices_close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR first value
    atr_sma = tr.rolling(window=window).mean()
    atr = pd.Series(index=tr.index, dtype='float64')
    atr.iloc[window - 1] = atr_sma.iloc[window - 1]
    
    # ATR values
    for i in range(window, len(tr)):
        atr.iloc[i] = (atr.iloc[i-1] * (window - 1) + tr.iloc[i]) / window
    
    return atr

def cmf(prices_close, prices_high, prices_low, volumes, window=21):
    if not isinstance(prices_close, pd.Series) or not isinstance(prices_high, pd.Series) \
        or not isinstance(prices_low, pd.Series) or not isinstance(volumes, pd.Series):
        raise ValueError('Error, prices data and volumes must be pandas.Series (not dataframes).')
    
    if not (len(prices_close) == len(prices_high) == len(prices_low) == len(volumes)):
        raise ValueError("Error, prices data must have same length.")

    # Close Location Value (CLV)
    clv = (prices_close - prices_low) - (prices_high - prices_close)
    clv /= (prices_high - prices_low)

    # Chaikin Money Flow
    cmf = np.array([0.0] * len(prices_close))

    for i in range(window - 1, len(prices_close)):
        sum_clv_volume = np.sum(clv[i - window + 1:i + 1] * volumes[i - window + 1:i + 1])
        sum_volume = np.sum(volumes[i - window + 1:i + 1])
        cmf[i] = sum_clv_volume / sum_volume if sum_volume != 0 else 0

    cmf = pd.Series(cmf, index=prices_close.index, name = 'CMF')
    
    return cmf
