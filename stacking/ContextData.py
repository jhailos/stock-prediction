from StrategyData import StrategyData

import pandas as pd
import os

class ContextData:

    def __init__(self, ticker, strategy: StrategyData, days=None, interval=None):
        self.strategy = strategy
        self.ticker = ticker
        self.days = days
        self.interval = interval
        self.data = self.strategy.download_data(ticker=ticker, days=days, interval=interval)

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
        self.data['EMA12'] = self.data['Close'].ewm(span=12).mean()
        self.data['EMA26'] = self.data['Close'].ewm(span=26).mean()

        # MACD
        self.data['MACD'] = self.data['EMA12'] + self.data['EMA26']

        # MACD signal
        self.data['MACD_signal'] = self.data['MACD'].ewm(span=9).mean()

        # Price change
        self.data['price_change'] = self.data['Close'].pct_change()

        # Previous closing price
        self.data['previous_close'] = self.data['Close'].shift(1)