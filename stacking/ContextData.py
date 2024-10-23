from StrategyData import StrategyData

import datetime
import pandas as pd
import os

class ContextData:

    def __init__(self, ticker, strategy: StrategyData, days=None, interval=None, in_market_hours=True):
        self.strategy = strategy
        self.ticker = ticker
        self.days = days
        self.interval = interval
        self.data = self.strategy.download_data(ticker=ticker, days=days, interval=interval)
        if in_market_hours : self.delete_after_hours()
        self.compute_features()

    def read_csv(self):
        return pd.read_csv(f'stock_data\{self.ticker}.csv', index_col='Datetime', parse_dates=True)

    def write_csv(self):
        outdir = "stock_data"
        outfile = f"{self.ticker}.csv"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.data.to_csv(os.path.join(outdir, outfile))

    def compute_features(self):
        # EMA
        # daily_closes = self.data['Close'].resample('1D').last()
        # print(daily_closes)
        # daily_closes = daily_closes.ffill()
        self.data['EMA12'] = self.data['Close'].ewm(span=12*6.5*60/2).mean() # 12 days * 6.5 hours per day * min per hour
        self.data['EMA26'] = self.data['Close'].ewm(span=26*6.5*60/2).mean()
        # print(self.data['EMA12'])
        # MACD
        self.data['MACD'] = self.data['EMA12'] + self.data['EMA26']

        # MACD signal
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()

        # Price change
        self.data['price_change'] = self.data['Close'].pct_change()

        # Previous closing price
        self.data['previous_close'] = self.data['Close'].shift(1)

    def delete_after_hours(self):
        """Delete after market hours from data set
        """
        market_hours = self.data.between_time(datetime.time(hour=9, minute=30), datetime.time(hour=16))
        market_hours = market_hours[market_hours.index.dayofweek < 5]