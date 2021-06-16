import math
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

data = pd.read_csv('year2.csv').values
ap = data[:, 0]
sum_weeks = []
weeks = range(26)

for i in weeks:
    ind = i*14
    days = ap[ind:ind+14]
    weeki = sum(days)
    sum_weeks.append(weeki)

ma4_range= range(2, 26)
ma8_range= range(4, 26)
ma12_range= range(6, 26)

ma4 = []
ma8 = []
ma12 = []

for i in ma4_range:
    mai = np.mean(sum_weeks[i-2:i])
    ma4.append(mai)

for i in ma8_range:
    mai = np.mean(sum_weeks[i-4:i])
    ma8.append(mai)

for i in ma12_range:
    mai = np.mean(sum_weeks[i-6:i])
    ma12.append(mai)



# plt.plot(ma4_range,ma4)
# plt.plot(ma8_range,ma8)


fp  =  84
lam = .05
start = 84

rr12 = []
for i in range(20):
    ind = start + 14 * i
    train = data[ind-fp:ind]
    test = data[ind:ind+14]

    x_train = train[:, 1:]
    x_test = test[:, 1:]
    y_train = train[:, 0]
    y_test = test[:, 0]

    ridge = Ridge(lam)
    ridge.fit(x_train, y_train)
    y_pred = ridge.predict(x_test)
    weeki = np.sum(y_pred)
    rr12.append(weeki)

plt.plot(ma12_range,rr12)

# rr12_time = []

# fp = 84
# start = 84

# for i in range(20):
#     ind = start + 14 * i
#     train = data[ind-fp:ind]
#     test = data[ind:ind+14]

#     x_train = train[:, 6:]
#     x_test = test[:, 6:]
#     y_train = train[:, 0]
#     y_test = test[:, 0]

#     ridge = Ridge(.31)
#     ridge.fit(x_train, y_train)
#     y_pred = ridge.predict(x_test)
#     weeki = np.sum(y_pred)
#     rr12_time.append(weeki)



# plt.plot(weeks, sum_weeks, label="True Power Produced")
# plt.plot(ma12_range,ma12, label="12-Week Moving Average")

# plt.legend()
# plt.show()

# plt.plot(weeks, sum_weeks, label="True Power Produced")
# plt.plot(ma12_range, rr12_time, label="Ridge Regression - Time")
# plt.plot(ma12_range,rr12, label="Ridge Regression - Sensor + Time")

# plt.legend()
# plt.show()

plt.plot(weeks, sum_weeks, label="True Power Produced")
plt.plot(ma4_range, ma4, label="4-Week Moving Average")
# plt.plot(ma12_range, rr12_time, label="Ridge Regression - Time")
plt.plot(ma12_range,rr12, label="Ridge Regression - Sensor + Time")

plt.legend()
# plt.show()

arr_true = np.array(sum_weeks[6:]).reshape((-1,1))
arr_4week = np.array(ma4[4:]).reshape((-1,1))
arr_rr12 = np.array(rr12).reshape((-1,1))

# print(arr_true.shape, arr_4week.shape, arr_rr12.shape)

arr_final = np.concatenate((arr_true, arr_4week, arr_rr12), 1)
# print(arr_final.shape)
columns = ['True', '4 Week Moving Average', 'Ridge Regression Senors + Time']
df_final = pd.DataFrame(data=arr_final, columns=columns)
df_final.to_csv('simulation_data.csv', index=False)



