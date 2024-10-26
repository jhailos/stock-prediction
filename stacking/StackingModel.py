import pandas as pd
from joblib import parallel_backend

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor

class StackingModel:
    """
    
    """

    def __init__(self, data, interval, estimators=None, final_estimator=None):
        """_summary_

        Args:
            ticker (str): Ticker of the stock
            interval (str): Interval at which the data is taken
            estimators (list, optional): List of estimators. Defaults to random forest, svr, adaboost, xgboost.
            final_estimator (_type_, optional): Final estimator. Defaults to sklearn.linear_model.LinearRegression.
        """
        self.data = data
        self.interval = interval
        self.estimators = estimators or [
            ('rf', RandomForestRegressor(n_estimators=100, n_jobs=-1)),
            ('bag', BaggingRegressor(estimator=LinearSVR(max_iter=1000000), n_estimators=100, n_jobs=-1)),
            ('ada', AdaBoostRegressor(n_estimators=100)),
            ('xgb', XGBRegressor(n_estimators=100, n_jobs=-1))
        ]
        self.final_estimator = final_estimator or LinearRegression()
        self.rmse = 0
        self.r2 = 0
        print("Data size: ", self.data.shape[0])

    def train_model(self, x_train, y_train):
        print('> Training model')
        pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', StackingRegressor(
                estimators=self.estimators, final_estimator=self.final_estimator
            ))
        ])
        
        # Train Stacking Model
        with parallel_backend('threading'):
            pipeline.fit(x_train, y_train)
        
        return pipeline

    def data_preprocessing(self):
        print('> Processing data')
        y = self.data['Close'].shift(-1) # Y is value to predict (price in the next interval)
        self.data['y'] = y
        self.data.dropna(inplace=True)
        x = self.data[['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']]
        return x, self.data['y']

    def scale_data(self, x_train, x_test, scaler):
        print('> Scaling data')
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

        return x_train_scaled, x_test_scaled

    def model_eval(self, model, x_test, y_test):
        print('> Evaluating model')
        prediction = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, prediction)
        r_squared = r2_score(y_test, prediction)

        return rmse, r_squared

    def timedelta_interval(self):
        """Converts the `self.interval` string to pd.Timedelta 

        Raises:
            ValueError: If the interval is not found

        Returns:
            pd.Timedelta: The converted timedelta
        """
        interval_mapping = {
            '1d': pd.Timedelta(days=1),
            '1h': pd.Timedelta(hours=1),
            '30m': pd.Timedelta(minutes=30),
            '15m': pd.Timedelta(minutes=15),
            '5m': pd.Timedelta(minutes=5),
            '2m': pd.Timedelta(minutes=2),
            '1m': pd.Timedelta(minutes=1)
        }

        if self.interval in interval_mapping:
            return interval_mapping[self.interval]
        else:
            raise ValueError(f"Interval mapping not found for the specified interval: {self.interval}")


    def next_closing(self, model, scaler, steps=1):
        """Predicts the closing price at future intervals.

        Parameters:
        - model: Trained model for prediction
        - scaler: StandardScaler object for data scaling
        - steps (int): Number of future intervals to predict. Defaults to 1 step

        Returns:
        - future_timestamps (list): Timestamps of the predicted closing prices
        - predicted_closing_price (float): Predicted closing price at the last future interval
        """
        print('> Calculating next closing price')
        most_recent = pd.DataFrame([self.data.iloc[-1][['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']]])
        
        most_recent_scaled = pd.DataFrame(scaler.transform(most_recent), columns=most_recent.columns)
        
        # Predict closing price at future interval
        future_data = []
        future_timestamps = [self.data.index[-1] + self.timedelta_interval()]  # Timestamps of the most recent data
        
        for _ in range(steps):
            prediction = model.predict(most_recent_scaled)
            future_data.append(prediction[0])
            future_timestamps.append(future_timestamps[-1] + self.timedelta_interval())
            
            # Update the scaled data with the new prediction
            most_recent_scaled.iloc[0, -1] = prediction

            previous_close = prediction[0]  # predicted closing price is the new previous_close

            most_recent_scaled.iloc[0, 4] = previous_close - most_recent.iloc[0]['previous_close']  # 5th column is `price_change`
        
        return future_timestamps, future_data[-1]
    
    def run(self):

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
        prediction_time, predicted_price = self.next_closing(model, scaler, steps=1)
        rrmse = self.rmse/self.data['Close'].mean() * 100
        print("RMSE: ", self.rmse)
        print(f"RRMSE: {rrmse:.4f}%")
        print("R2: ", self.r2)
        print(f"Price in next interval ({prediction_time[-1]}): {predicted_price}")

        return [self.rmse, rrmse, self.r2, predicted_price, self.data['Close'].iloc[-1]]
        # return predicted_price
