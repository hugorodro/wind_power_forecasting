import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

### import clean data ###
data = pd.read_csv('clean_data_2.csv').values
ap = data[:, 1]

## histogram of average power
# w=1000
# n = math.ceil((ap.max()-ap.min())/w)
# plt.hist(ap, bins = n)
# plt.show()
# exit()

# make dates circular
dates = data[:, 0]
circular = np.zeros((len(dates), 2))
for i in range(len(dates)):
    circular[i] = np.array(
        [math.cos(2*math.pi*i/365), math.cos(2*math.pi*i/365)])

# reformat arr
data = np.delete(data, 0, 1)
data = np.concatenate((data, circular), axis=1)
# print(data.shape)

# normalize
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

year1 = data[:364]
year2 = data[364:]

df_year2 = pd.DataFrame(year2)
df_year2.to_csv('year2.csv', index=False)

# plt.plot(year1)
# plt.show()

print(year1.shape, year2.shape)

forcastPeriods = range(14, 84+1,14)
interval = .01
lambdas = np.arange(0, 1 + interval, interval)
start = 84

avg_r2 = []
avg_mse = []

### loop for forecast period ###
for fp in forcastPeriods:

    sum_r2 = np.zeros(len(lambdas))
    sum_mse = np.zeros(len(lambdas))

    # test train split
    for i in range(20):
        ind = start + 14 * i
        train = year1[ind-fp:ind]
        test = year1[ind:ind+14]

        x_train = train[:, 6:]
        x_test = test[:, 6:]
        y_train = train[:, 0]
        y_test = test[:, 0]

        # print(fp, x_train.shape, x_test.shape, y_train.shape, y_test.shape)


        for j in range(len(lambdas)):
            ridge = Ridge(alpha=lambdas[j])
            ridge.fit(x_train, y_train)
            y_pred = ridge.predict(x_test)

            # metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = ridge.score(x_train, y_train)
            sum_mse[j] += mse
            sum_r2[j] += r2

    avg_mse.append(sum_mse / 20)
    avg_r2.append(sum_r2 / 20)             

avg_mse = np.array(avg_mse)
avg_r2 = np.array(avg_r2)

# print(avg_mse)
# avg_mse_fp = np.mean(avg_mse, axis=1)
# avg_r2_fp = np.mean(avg_r2, axis=1)

avg_mse_lam = np.mean(avg_mse, axis=0)
avg_r2_lam = np.mean(avg_r2, axis=0)

# plt.plot(forcastPeriods, avg_mse_fp)
# plt.show()

# plt.plot(forcastPeriods, avg_r2_fp)
# plt.show()

plt.plot(lambdas, avg_mse_lam, 'o')
plt.title('MSE vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('MSE')
plt.show()
plt.show()

plt.plot(lambdas, avg_r2_lam, 'o')
plt.title('R^2 vs Lambda')
plt.xlabel('Lambda')
plt.ylabel('R^2')
plt.show()

# print("Best Forecast Period = " + str(forcastPeriods[np.argmin(avg_mse_fp)]))
print("Best Lambda = " + str(lambdas[np.argmin(avg_mse_lam)]) )
