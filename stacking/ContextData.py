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
        # self.handle_missing_date()
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

    def handle_missing_date(self):
        """Adds the missing data points
        """
        # Resample to 1-minute frequency (in case of gaps)
        data_resampled = self.data.resample('D').asfreq()
        # Fill missing values using linear interpolation
        data_filled = data_resampled.interpolate(method='linear')

        self.data = data_filled

    def compute_features(self):
        # EMA
        self.data['EMA12'] = self.data['Close'].ewm(span=math.ceil(12*6.5*60)).mean() # 12 days * 6.5 hours per day * min per hour
        self.data['EMA26'] = self.data['Close'].ewm(span=math.ceil(26*6.5*60)).mean()

        # MACD
        self.data['MACD'] = self.data['EMA12'] + self.data['EMA26']

        # MACD signal
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()

        #RSI
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=math.ceil(14*6.5*60)).mean() # 14 days * 6.5 hours per day * min per hour
        loss = (-delta.where(delta < 0, 0)).rolling(window=math.ceil(14*6.5*60)).mean()
        relative_strength = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + relative_strength))

        # Price change
        self.data['price_change'] = self.data['Close'].pct_change()

        # Previous closing price
        self.data['previous_close'] = self.data['Close'].shift(1)

        # self.data.ffill()

    def delete_after_hours(self):
        """Delete after market hours from data set
        """
        market_hours = self.data.between_time(datetime.time(hour=9, minute=30), datetime.time(hour=16))
        market_hours = market_hours[market_hours.index.dayofweek < 5]
        self.data = market_hours
    
    def delete_outliers(self, threshold=0.013): #! CHICKEM_Empanada<3
        """
        Delete outliers from dataset using a threshold on percent change in closing price.
        Threshold should be higher for higher intervals.

        Args:
        threshold (float): The threshold at which the data will be deleted

        """
        # 3 Standard Deviations away from mean
        upper_bound = self.data['price_change'].mean() + 3 * self.data['price_change'].std()
        lower_bound = self.data['price_change'].mean() - 3 * self.data['price_change'].std()
        
        self.data = self.data[self.data['price_change'] < upper_bound]
        self.data = self.data[self.data['price_change'] > lower_bound]

        # self.handle_missing_date()