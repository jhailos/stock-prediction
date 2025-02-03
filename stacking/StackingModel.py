import pandas as pd
from joblib import parallel_backend

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
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
        print(self.data)
        print("Data size: ", self.data.shape[0])

    def train_model(self, x_train, y_train):
        print('> Training model', end='\r')
        pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', StackingRegressor(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                n_jobs=-1,
                verbose=1
            ))
        ])
        
        # Train Stacking Model
        with parallel_backend('threading'):
            pipeline.fit(x_train, y_train)
        
        return pipeline

    def data_preprocessing(self):
        print('> Processing data', end='\r')
        # Create lagged features for time series
        self.data['Close_lag1'] = self.data['Close'].shift(1)
        self.data['Close_lag2'] = self.data['Close'].shift(2)
        self.data['Close_lag3'] = self.data['Close'].shift(3)
        self.data.dropna(inplace=True)

        # Define features and target
        x = self.data[['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close', 'Close_lag1', 'Close_lag2', 'Close_lag3']]
        y = self.data['Close']
        return x, y

    def scale_data(self, x_train, x_test, scaler):
        print('> Scaling data', end='\r')
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

        return x_train_scaled, x_test_scaled

    def model_eval(self, model, x_test, y_test):
        print('> Evaluating model', end='\r')
        prediction = model.predict(x_test)
        rmse = root_mean_squared_error(y_test, prediction)
        rrmse = root_mean_squared_error(y_test, prediction) / self.data['Close'].mean()
        r2 = r2_score(y_test, prediction)
        
        return rmse, rrmse, r2

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
        most_recent = pd.DataFrame([self.data.iloc[-1][['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close', 'Close_lag1', 'Close_lag2', 'Close_lag3']]])
        most_recent_scaled = pd.DataFrame(scaler.transform(most_recent), columns=most_recent.columns)

        future_timestamps = [self.data.index[-1] + self.timedelta_interval()]
        future_data = []

        for _ in range(steps):
            prediction = model.predict(most_recent_scaled)
            future_data.append(prediction[0])

            # Update the future timestamps
            future_timestamps.append(future_timestamps[-1] + self.timedelta_interval())

            # Update the scaled data with the new prediction
            most_recent_scaled.iloc[0, -1] = prediction  # Update 'Close_lag1'
            most_recent_scaled.iloc[0, -2] = most_recent_scaled.iloc[0, -1]  # Update 'Close_lag2'
            most_recent_scaled.iloc[0, -3] = most_recent_scaled.iloc[0, -2]  # Update 'Close_lag3'

        return future_timestamps[1:], future_data[-1]
        
    def run(self):

        # Pre process
        x, y = self.data_preprocessing()

        tss = TimeSeriesSplit(n_splits=5)

        for train_index, test_index in tss.split(self.data):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale data
        scaler = StandardScaler()
        x_train_scaled, x_test_scaled = self.scale_data(x_train, x_test, scaler)

        # Train model
        model = self.train_model(x_train_scaled, y_train)

        # Evaluate model
        rmse, rrmse, r2 = self.model_eval(model, x_test_scaled, y_test)

        # Inference
        prediction_time, predicted_price = self.next_closing(model, scaler, steps=1)

        return rmse, rrmse, r2, predicted_price, self.data['Close'].iloc[-1], prediction_time[-1]
