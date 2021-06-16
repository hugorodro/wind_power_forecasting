import numpy as np

# normalize data
def normalize_array(arr):
    amax = np.max(arr)
    amin = np.min(arr)
    return (arr - amin)/ (amax - amin)

# split data into test size of 14, and train of the remaining values
def test_train_split(x, y, forecast_period):
    n = len(x)
    mid = n - forecast_period
    x_train = x[:mid]
    y_train = y[:mid]
    x_test = x[mid:]
    y_test = y[mid:]

    return x_train, y_train, x_test, y_test

# calculates the difference between true and predicted values
def calc_delta(y_true, y_predict):
    delta= y_true - y_predict
    return delta