import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import concurrent.futures

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class StackingModel:
    """
    
    """

    def __init__(self, ticker: str, interval: str, estimators=None, final_estimator=None):
        self.ticker = ticker
        self.interval = interval
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.data = self.download_data()
        if self.estimators == None:
            self.estimators = [
                ('rf', RandomForestRegressor(n_estimators=100)),
                ('svr', SVR(kernel='linear')),
                ('ada', AdaBoostRegressor(n_estimators=100)),
                ('xgb', XGBRegressor(n_estimators=100))
            ]
        if self.final_estimator == None:
            self.final_estimator = RandomForestRegressor()
        self.rmse = 0
        self.r2 = 0

    def download_data(self):
        """Fetch data from yfinance
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=59)

        data = yf.download(self.ticker, start=start_date, end=end_date, interval=self.interval)
        return data

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

    def train_model(self, x_train, y_train):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', StackingRegressor(
                estimators=self.estimators, final_estimator=self.final_estimator
            ))
        ])
        
        # Train Stacking Model
        pipeline.fit(x_train, y_train)
        
        return pipeline

    def data_preprocessing(self, data):
        y = self.data['Close'].shift(-1) # Y is value to predict (price in the next interval)
        self.data['y'] = y
        self.data.dropna(inplace=True) #! Added drop NA but paper says to : "use the method of imputing with the prior existing values to handle missing values"
        x = self.data[['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']]

        return x, self.data['y']

    def scale_data(self, x_train, x_test, scaler):
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return x_train_scaled, x_test_scaled

    def model_eval(self, model, x_test, y_test):
        prediction = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, prediction)
        r_squared = r2_score(y_test, prediction)

        return rmse, r_squared

    def next_closing(self, model, scaler):
        most_recent = self.data.iloc[-1][['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']].values.reshape(1, -1)
        scaled = scaler.transform(most_recent)
        
        prediction = model.predict(scaled)

        return prediction[0]
    
    def run(self):
        # Compute features
        self.compute_features()

        # Pre process
        x, y = self.data_preprocessing()

        # Scaler
        scaler = StandardScaler()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        x_train_scaled, x_test_scaled = self.scale_data(x_train, x_test, scaler)

        # Train model
        model = self.train_model(x_train_scaled, y_train)

        # Evaluate model
        self.rmse, self.r2 = self.model_eval(model, x_test_scaled, y_test)

        # Inference
        prediction_time = self.data.index[-1] + datetime.timedelta(minutes=15)
        predicted_price = self.next_closing(model, scaler)

        print("RMSE: ", self.rmse)
        print("R2: ", self.r2)
        print(prediction_time)
        print("Price in next interval: ", predicted_price)

        # return [rmse, r2, prediction_time, predicted_price]
        # return predicted_price
