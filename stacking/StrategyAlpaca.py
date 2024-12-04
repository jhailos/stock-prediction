from StrategyData import StrategyData

import datetime
import yaml

from StrategyData import StrategyData
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

class StrategyAlpaca(StrategyData):
    
    def download_data(self, ticker, days, interval):

        with open("api_key.yml", "r") as file:
            api_key = yaml.safe_load(file)
            client = StockHistoricalDataClient(api_key=api_key['alpaca'], secret_key=api_key['alpaca_secret'])

            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)

            request_params = StockBarsRequest(
                symbol_or_symbols = f'{ticker}',
                timeframe = self.translate_interval(interval),
                start = start_date,
                end = end_date
            )

            bars = client.get_stock_bars(request_params)
            df = bars.df
            print(df)
            df.columns = map(str.capitalize, df.columns)
            return df
    
    def translate_interval(self, interval):
        """Converts the past in `interval` string to an alpaca TimeFrame interval

        Raises:
            ValueError: If the interval is not found

        Returns:
            alpaca TimeFrame: The converted timedelta
        """
        interval_mapping = {
            '1d': TimeFrame.Day,
            '1h': TimeFrame.Hour,
            '1m': TimeFrame.Minute
        }

        if interval in interval_mapping:
            return interval_mapping[interval]
        else:
            raise ValueError(f"Interval mapping not found for the specified interval: {interval}")
