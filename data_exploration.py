import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math

#Initial import
df_data = pd.read_csv('Turbine_Data.csv')
df_data_columns = df_data.columns
df_data_null_perc = df_data.isnull().sum(axis=0) / df_data.shape[0]

# print('Shape after import:' + str(df_data.shape))
# print('Columns after import: ' + str(df_data_columns))
# print('Null Count after import:'+ str(df_data_null_perc))

## bar graph of null values
# df_data_null_perc.plot(kind="barh")
# plt.title('Percentage of Null Values in Each Feature Columns')
# plt.xlabel('Features')
# plt.show()


# Drop Control Box Temp (Values are 0 or blank), WTG (1 unique value), Columns with %null > 40%
df_data = df_data.drop(['ControlBoxTemperature', 'WTG'], axis=1)

df_data = df_data.drop(
    ['Blade1PitchAngle', 'Blade2PitchAngle', 'Blade3PitchAngle'], axis=1)
df_data = df_data.drop(['BearingShaftTemperature',
                        'GearboxBearingTemperature', 'GearboxOilTemperature'], axis=1)
df_data = df_data.drop(
    ['GeneratorRPM', 'GeneratorWinding1Temperature', 'GeneratorWinding2Temperature'], axis=1)
df_data = df_data.drop(['HubTemperature', 'MainBoxTemperature'], axis=1)
df_data = df_data.drop(['RotorRPM', 'TurbineStatus'], axis=1)

shape_post_drop = df_data.shape
columns_post_drop = df_data.columns

# print('Shape after drop:' + str(shape_post_drop))
# print('Columns after import: ' + str(columns_post_drop))

# Convert time stamp to date
df_data['Time'] = df_data['Time'].str[:10]
df_data = df_data.rename(columns={'Time': 'Day'})

## Average feautues in day excluding nulls
# Initial Variables
arr_data = df_data.values
n_days = len(np.unique(arr_data[:, 0]))
n_samples, n_cols = arr_data.shape[0], arr_data.shape[1]
arr_daily = np.zeros((n_days, n_cols))
day = 0
sum_valid = np.zeros(n_cols)
n_valid = np.zeros(n_cols)


# Outer Loop over each sample
for i in range(n_samples):

    # Inner Loop over each feature expect day
    for j in range(1, n_cols):
        val = float(arr_data[i, j])

        if math.isnan(val) == False:
            sum_valid[j] += val
            n_valid[j] += 1

    # condition: algorithm acts differently at the last sample
    if i < n_samples - 1:
        day_current_row = arr_data[i, 0]
        day_next_row = arr_data[i+1, 0]

        # condition: next samples in different date, therefore must aggregate
        if day_current_row != day_next_row:

            day_avg = sum_valid / n_valid
            day_avg[0] = day
            day_avg[1] = day_avg[1] * 144
            arr_daily[day] = day_avg

            day += 1
            sum_valid = np.zeros(n_cols)
            n_valid = np.zeros(n_cols)


df_daily = pd.DataFrame(arr_daily, columns=df_data.columns)
df_daily_null_perc = df_daily.isnull().sum(axis=0)
# print(sum(df_daily_null_perc / (df_daily.shape[0] * df_daily.shape[1])))

# # Fill null values with feature means
# feature_means = np.nanmean(arr_daily, axis=0)
# ind_null = np.where(np.isnan(arr_daily))
# arr_daily[ind_null] = np.take(feature_means, ind_null[1])
# final_data = arr_daily
# final_shape = final_data.shape
# final_columns = df_data.columns

# # print(final_shape)
# # print(len(final_columns))
# # print(final_columns)

# df_clean = pd.DataFrame(final_data, columns=final_columns)
# df_clean_null_perc = df_clean.isnull().sum(axis=0) / df_data.shape[0]
# # print(df_clean_null_perc)

# # Save data 
# # df_clean.to_csv('clean_data.csv', index=False)

# Get Two years worth of data fill data with means and save
daily_2 = arr_daily[93:, :]

# # Fill null values with feature means
feature_means = np.nanmean(daily_2, axis=0)
ind_null = np.where(np.isnan(daily_2))
daily_2[ind_null] = np.take(feature_means, ind_null[1])
final_data_2 = daily_2
final_shape_2 = final_data_2.shape
final_columns_2= df_data.columns

# print(final_shape_2)
# print(len(final_columns_2))
# print(final_columns_2)

df_clean_2 = pd.DataFrame(final_data_2, columns=final_columns_2)
df_clean_2_null = df_clean_2.isnull().sum(axis=0)
print(df_clean_2_null)

df_clean_2.to_csv('clean_data_2.csv', index=False)


