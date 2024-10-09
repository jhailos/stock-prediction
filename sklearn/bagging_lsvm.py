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
    start_date = end_date - datetime.timedelta(days=60)

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    return data

def main():
    # Main function to run the analysis
    ticker = 'NVDX'
    
    # Download data
    data = download_data(ticker)

if __name__ == "__main__":
    main()
