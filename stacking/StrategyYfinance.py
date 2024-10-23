from StrategyData import StrategyData

import yfinance as yf
import datetime

class StrategyYfinance(StrategyData):

    def download_data(self, ticker, days, interval):
        """Fetch data from yfinance
        """
        if days is None:
            days = 59
        if interval is None:
            interval = "5m"
            
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)

        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        data = data.ffill() #* replace NaNs with the previous valid data
        return data