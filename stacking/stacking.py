import yfinance as yf
import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
"""

"""
class StackingModel:
    def __init__(self, ticker: str, interval: str, estimators=None, final_estimator=None):
        self.ticker = ticker
        self.interval = interval
        self.estimators = estimators
        self.final_estimator = final_estimator
        if self.estimators == None:
            self.estimators = [
                ('rf', RandomForestRegressor(n_estimators=100)),
                ('svr', SVR(kernel='linear')),
                ('ada', AdaBoostRegressor(n_estimators=100)),
                ('xgb', XGBRegressor(n_estimators=100))
            ]
        if self.final_estimator == None:
            self.final_estimator = RandomForestRegressor()

    def download_data(self):
        """Fetch data from yfinance
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=59)

        data = yf.download(self.ticker, start=start_date, end=end_date, interval=self.interval)
        data.dropna(inplace=True)
        return data

    def compute_features(self, data):
        # EMA
        data['EMA12'] = data['Close'].ewm(span=12).mean()
        data['EMA26'] = data['Close'].ewm(span=26).mean()

        # MACD
        data['MACD'] = data['EMA12'] + data['EMA26']

        # MACD signal
        data['MACD_signal'] = data['MACD'].ewm(span=9).mean()

        # Price change
        data['price_change'] = data['Close'].pct_change()

        # Previous closing price
        data['previous_close'] = data['Close'].shift(1)
        
        data.dropna(inplace=True)

        return data

    def train_model(self, x_train, y_train):
        model = StackingRegressor(
            estimators=self.estimators, final_estimator=self.final_estimator
        )
        
        # Train Stacking Model
        model.fit(x_train, y_train)
        
        return model

    def data_preprocessing(self, data):
        y = data['Close'].shift(-1) # Y is value to predict (price in the next interval)
        data['y'] = y
        data.dropna(inplace=True) #! Added drop NA but paper says to : "use the method of imputing with the prior existing values to handle missing values"
        x = data[['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']]

        return x, data['y']

    def scale_data(self, x_train, x_test, scaler):
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled

    def model_eval(self, model, x_test, y_test):
        prediction = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, prediction)
        r_squared = r2_score(y_test, prediction)

        return rmse, r_squared

    def next_closing(self, data, model, scaler):
        most_recent = data.iloc[-1][['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']].values.reshape(1, -1)
        scaled = scaler.transform(most_recent)
        
        prediction = model.predict(scaled)

        return prediction[0]
    
    def run(self):
        # Download data
        data = self.download_data()

        # Compute features
        data = self.compute_features(data)

        # Pre proc
        x, y = self.data_preprocessing(data)

        # Scaler
        scaler = StandardScaler()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        x_train_scaled, x_test_scaled = self.scale_data(x_train, x_test, scaler)

        model = self.train_model(x_train_scaled, y_train)

        rmse, r2 = self.model_eval(model, x_test_scaled, y_test)
        print("RMSE: ", rmse)
        print("R2: ", r2)
        print(data.index[-1] + datetime.timedelta(minutes=5))
        print("Price in next interval: ", self.next_closing(data, model, scaler))

def main():
    model = StackingModel("NVDX", "15m")
    model.run()

if __name__ == "__main__":
    main()
