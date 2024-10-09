import yfinance as yf
import pandas as pd
import numpy as np
import sklearn
from sklearn.svm import LinearSVR
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime
"""

"""
def download_data(ticker, interval='15m'):
    """Fetch data from yfinance
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=59)

    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    return data

def compute_features(data):
    # EMA
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()

    # MACD
    data['MACD'] = data['EMA12'] + data['EMA26']

    #MACD signal
    data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
    
    return data

def train_model(x, y):
    linearSVR = LinearSVR(max_iter=100) #! Unsure max_iter
    bagging_model = BaggingClassifier(estimator=linearSVR, n_estimators=100)
    bagging_model.fit(x, y)

    return bagging_model

def data_preprocessing(data):
    y = data['Close'].shift(1).dropna() # Y is value to preduct (price in the next 5min)
    data['y'] = y
    #! Added drop NA but paper says to : "use the method of imputing with the prior existing values to handle missing values"
    x = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'EMA12', 'EMA26', 'MACD', 'MACD_signal']].dropna()

    return x, y

def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled

def main():
    # Main function to run the analysis
    ticker = 'NVDX'
    
    # Download data
    data = download_data(ticker)

    # Compute features
    data = compute_features(data)

    # Pre proc
    x, y = data_preprocessing(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    x_train_scaled, x_train_scaled = scale_data(x_train, x_test)

    model = train_model(x_train_scaled, y_train)


if __name__ == "__main__":
    main()
