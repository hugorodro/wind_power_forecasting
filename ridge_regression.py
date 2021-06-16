import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ridge regression class
class Ridge_Regression:
    def __init__(self, X, y, n_lambdas):
        self.lambda_range = np.arange(0, 1, 1/n_lambdas)
        self.X = X
        self.y = y
        self.n_lambdas = n_lambdas
        self.arr_B = None
        self.predictions = None

    # fits a ridge regrssion to each lambda
    def fit(self):
        x = self.X
        y = self.y

        # print(x)
        # print(y)

        n_lambda = self.n_lambdas
        arr_B = np.zeros((n_lambda, x.shape[1]))
        right = x.T@y
        A = x.T@x
        I = np.eye(A.shape[0])
        for i in range(n_lambda):
            lambda_matrix = self.lambda_range[i] * I
            left = A + lambda_matrix
            # Bi = np.linalg.lstsq(left, right)
            Bi = np.linalg.solve(left, right)
            arr_B[i] = Bi
            # print(Bi)
        self.arr_B = arr_B
        # print(arr_B[0])
        # print(arr_B)

    # makes a prediction for each set of parameters, one for each lambda
    def predict(self, x):
        n_lambda = self.n_lambdas
        n_samples = x.shape[0]
        predicted_power_at_each_day = np.zeros((n_lambda, n_samples))
    
        for i in range(n_lambda):
            Bi = self.arr_B[i]
            # print(Bi)
            for j in range(n_samples):
                xi = x[j]
                # print(xi.shape)
                y_predict = Bi.T@xi
                predicted_power_at_each_day[i, j] = y_predict
        # exit()
        # print(predicted_power_at_each_day)

        predicted_power_at_forecast_period = np.sum(
            predicted_power_at_each_day, axis=1)
        # print(predicted_power_at_forecast_period)
        # exit()
        return predicted_power_at_forecast_period, predicted_power_at_each_day