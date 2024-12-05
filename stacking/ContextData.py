from StrategyData import StrategyData

import datetime
import pandas as pd
import os
import math

class ContextData:
    def __init__(self, ticker, strategy: StrategyData, days=None, interval=None, market_hours_only=True):
        self.strategy = strategy
        self.ticker = ticker
        self.days = days
        self.interval = interval
        self.data = self.strategy.download_data(ticker=ticker, days=days, interval=interval)
        self.data.sort_index(inplace=True)
        if market_hours_only : self.delete_after_hours()
        self.compute_features()
        self.delete_outliers()

    def read_csv(self):
        return pd.read_csv(f'stock_data\\{self.ticker}.csv', index_col='Datetime', parse_dates=True)

    def write_csv(self):
        outdir = "stock_data"
        outfile = f"{self.ticker}.csv"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.data.to_csv(os.path.join(outdir, outfile))

    def compute_features(self):
        # EMA
        self.data['EMA12'] = self.data['Close'].ewm(span=math.ceil(12*6.5*60/2)).mean() # 12 days * 6.5 hours per day * min per hour
        self.data['EMA26'] = self.data['Close'].ewm(span=math.ceil(26*6.5*60/2)).mean()

        # MACD
        self.data['MACD'] = self.data['EMA12'] + self.data['EMA26']

        # MACD signal
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()

        #RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=math.ceil(14*6.5*60/2)).mean() # 14 days * 6.5 hours per day * min per hour
        loss = (-delta.where(delta < 0, 0)).rolling(window=math.ceil(14*6.5*60/2)).mean()
        relative_strength = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + relative_strength))

        # Price change
        self.data['price_change'] = self.data['Close'].pct_change()

        # Previous closing price
        self.data['previous_close'] = self.data['Close'].shift(1)

    def delete_after_hours(self):
        """Delete after market hours from data set
        """
        market_hours = self.data.between_time(datetime.time(hour=9, minute=30), datetime.time(hour=16))
        market_hours = market_hours[market_hours.index.dayofweek < 5]
        self.data = market_hours
    
    def delete_outliers(self):
        # upper = self.data['Close'].mean() + 3 * self.data['Close'].std()
        # lower = self.data['Close'].mean() - 3 * self.data['Close'].std()

        # data = (self.data[self.data['Close'] > upper]) and (self.data[self.data['Close'] < lower])
        pass