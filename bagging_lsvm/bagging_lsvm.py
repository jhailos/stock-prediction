import yfinance as yf
import pandas as pd
import numpy as np
import datetime

import sklearn
from sklearn.svm import LinearSVR
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
"""

"""
def download_data(ticker, interval='5m'):
    """Fetch data from yfinance
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=59)

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    data.dropna(inplace=True)
    return data

def compute_features(data):
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

def train_model(x_train, y_train):
    linear_SVR = LinearSVR(max_iter=1000000) #! Unsure max_iter
    bagging_model = BaggingRegressor(estimator=linear_SVR, n_estimators=100)
    bagging_model.fit(x_train, y_train)

    return bagging_model

def data_preprocessing(data):
    y = data['Close'].shift(-1) # Y is value to predict (price in the next interval)
    data['y'] = y
    data.dropna(inplace=True) #! Added drop NA but paper says to : "use the method of imputing with the prior existing values to handle missing values"
    x = data[['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']]

    return x, data['y']

def scale_data(x_train, x_test, scaler):
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def model_eval(model, x_test, y_test):
    prediction = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r_squared = r2_score(y_test, prediction)

    return rmse, r_squared

def next_closing(data, model, scaler):
    most_recent = data.iloc[-1][['EMA12', 'EMA26', 'MACD', 'MACD_signal', 'price_change', 'previous_close']].values.reshape(1, -1)
    scaled = scaler.transform(most_recent)
    
    prediction = model.predict(scaled)

    return prediction[0]

def main():
    # Main function to run the analysis
    ticker = 'NVDX'
    
    # Download data
    data = download_data(ticker)

    # Compute features
    data = compute_features(data)

    # Pre proc
    x, y = data_preprocessing(data)


    # Scaler
    scaler = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test, scaler)

    model = train_model(x_train_scaled, y_train)

    rmse, r2 = model_eval(model, x_test_scaled, y_test)
    print("RMSE: ", rmse)
    print("R2: ", r2)
    print(data.index[-1] + datetime.timedelta(minutes=5))
    print("Price in next interval: ", next_closing(data, model, scaler))

if __name__ == "__main__":
    main()
