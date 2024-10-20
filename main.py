from data_fetcher import fetch_market_data
from indicators import *
from pandasgui import show
import pandas as pd

ticker = 'BTC-USD'
start_date = None
end_date = None
market_data = fetch_market_data(ticker, start_date=start_date, end_date=end_date)
