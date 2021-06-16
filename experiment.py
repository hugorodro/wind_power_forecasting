import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions import normalize_array, test_train_split, calc_delta
from ridge_regression import Ridge_Regression
import math 

# import data, create x and y arrays
data = pd.read_csv('clean_data.csv').values
y = data[:,1]
date = np.reshape(data[:,0], (len(y), 1))
other_attributes = data[:,2:]
x = np.concatenate((date, other_attributes), axis=1)

# normlize values x and y values
for j in range(x.shape[1]):
    new_arr = normalize_array(x[:,j])
    x[:,j] = new_arr

y = normalize_array(y)

## time series cross validation
# instantiate variables
k = 20
n_lambdas = 20
forcast_period = 14
fold_end = [(821//k)*(i+1) for i in range(k)]
delta = np.zeros((k, n_lambdas))
r2_each_lambda = np.zeros((k,n_lambdas))

# loop through each fold
for i in range(k):

    # retrieves the end of the fold, and splices data set
    val = fold_end[i]
    xk = x[:val+1,:]
    yk = y[:val+1]

    # retrieves the test train split
    x_train, y_train, x_test, y_test = test_train_split(xk, yk, forcast_period)

    # fit the data at each lambda
    # return the aggregate sum (1) and the power at each day (14)   
    rreg = Ridge_Regression(x_train,y_train, n_lambdas)
    rreg.fit()
    y_predict_at_each_lambda, predicted_power_at_each_day  = rreg.predict(x_test)

    # total sum of squares
    sst = np.sum(np.square(y_test - np.mean(y_test)))

    for j in range(0, n_lambdas):
        # r2 = r2_score(predicted_power_at_each_day[j], y_test)
        ssr = np.sum(np.square(calc_delta(predicted_power_at_each_day[j], y_test)))
        r2 = 1 - (ssr / sst)
        # plt.plot(range(14),predicted_power_at_each_day[j])
        # plt.plot(range(14), y_test)

        # plt.title('R2: ' + str(r2) + " vs. Lambda: " + str(.05*j))
        # plt.xlabel('Lambda')
        # plt.ylabel('Active power')
        # plt.show()

        r2_each_lambda[i,j] = r2
        # print(r2_each_lambda)

    y_true = np.sum(y_test)

    delta_at_each_lambda = calc_delta(y_true, y_predict_at_each_lambda)
    delta[i] = delta_at_each_lambda

    # print(rreg.lambda_range.shape)
    # print(rreg.arr_B.shape)
    # print(prediction.shape)
    # print(y_predict_at_each_lambda.shape)
    # print(y_true)
    # print(delta_at_each_lambda.shape)


mse_lambda = np.mean(np.square(delta), axis=0)


plt.plot(np.arange(0,1, 1/n_lambdas), mse_lambda, 'o')
plt.title('MSE vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.show()
print(delta.shape)

mse_fold = np.mean(np.square(delta), axis=1)

plt.plot(np.arange(1,k+1), mse_fold, 'o')
plt.title('MSE vs K-Fold')
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()

mr2 = np.mean(r2_each_lambda, axis=0)

plt.plot(np.arange(0,1, 1/n_lambdas), mr2, 'o')
plt.title('R^2 vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('R^2')
plt.show()

