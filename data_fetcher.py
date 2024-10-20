import yfinance as yf
import pandas as pd

def fetch_market_data(ticker, start_date='2020-01-01', end_date=None, interval='1d'):
    try:
        market_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if market_data.empty:
            raise ValueError("No data available")
        return market_data
    except Exception as e:
        print(f"Error while fetching yfinance data: {e}")
        return pd.DataFrame()