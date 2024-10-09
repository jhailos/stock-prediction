import yfinance as yf
import pandas as pd
import numpy as np
import datetime
"""

"""
def download_data(ticker, interval='15m'):
    """Fetch data from yfinance
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=59)

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    return data

def compute_features(data):
    # EMA
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()

    # MACD
    data['MACD'] = data['EMA12'] + data['EMA26']

    #MACD signal
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    
    return data

def main():
    # Main function to run the analysis
    ticker = 'NVDX'
    
    # Download data
    data = download_data(ticker)

    print(compute_data(data))

if __name__ == "__main__":
    main()
