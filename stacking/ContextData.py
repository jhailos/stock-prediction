from StrategyData import StrategyData

import concurrent.futures
import time
import requests
import pandas as pd
import datetime
import yaml
import json
import os

class ContextData:

    def __init__(self, ticker, strategy: StrategyData, days=None, interval=None):
        self.strategy = strategy
        self.ticker = ticker
        self.days = days
        self.interval = interval
        self.data = self.strategy.download_data(ticker=ticker, days=days, interval=interval)

    def read_csv(self):
        return pd.read_csv(f'stock_data\{self.ticker}.csv', index_col='Datetime', parse_dates=True)

    def write_csv(self):
        outdir = "stock_data"
        outfile = f"{self.ticker}.csv"
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        self.data.to_csv(os.path.join(outdir, outfile))