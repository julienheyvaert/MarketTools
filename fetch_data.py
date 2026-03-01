import yfinance as yf

def fetch_data(ticker, start="2024-01-01", end=None, period=None):
    if period:
        data = yf.download(ticker, period=period)
    else:
        data = yf.download(ticker, start=start, end=end)

    if data.empty:
        print(f"No data found for {ticker}.")
        return None

    print(f"Fetched {len(data)} days of data for {ticker}.")
    return data


if __name__ == "__main__":
    data = fetch_data("BTC-USD", period="6mo")
    print(data.tail())