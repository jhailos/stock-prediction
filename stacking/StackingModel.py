import pandas as pd
from joblib import parallel_backend

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
import numpy as np

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

    def train_model(self, x, y):
        print('> Training model', end='\r')

        scoring = {
            "RMSE": 'neg_root_mean_squared_error',
            "R2": 'r2',
        }

        pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stacking', StackingRegressor(
                estimators=self.estimators,
                final_estimator=self.final_estimator,
                n_jobs=-1,
                verbose=1
            ))
        ])
        
        tss = TimeSeriesSplit(n_splits=5)

        scores = dict()

        for score in scoring:
            cv_scores = cross_val_score(
                pipeline,
                x,
                y,
                cv=tss,
                scoring=scoring[score],
                n_jobs=-1
            )
            scores[score] = cv_scores

        # Train Stacking Model
        with parallel_backend('threading'):
            pipeline.fit(x, y)
        
        return pipeline, scores

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

        # Train model
        model, cv_scores = self.train_model(x, y)

        # Inference
        prediction_time, predicted_price = self.next_closing(model, model.named_steps['scaler'], steps=1)

        scores = dict()

        for score in cv_scores:
            scores[score] = np.array(cv_scores[score]).mean()
        
        print(cv_scores)

        return scores, predicted_price, self.data['Close'].iloc[-1], prediction_time[-1]
