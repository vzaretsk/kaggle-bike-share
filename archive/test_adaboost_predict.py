import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree
import sklearn.ensemble
import sklearn.externals
import os.path

# if true, will overwrite existing files
overwrite = True

fig_num = 1

window_days = 3

# load the trained adaboost classifier
adaboost_regress = sklearn.externals.joblib.load("adaboost_regress.pkl")

# load the cleaned train data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_clean_df = pd.read_csv("train_clean.csv", parse_dates=["datetime"], index_col="datetime")

# load the saved workday and weekend trends
workday_trend = np.loadtxt("workday_trend.csv", delimiter=",")
weekend_trend = np.loadtxt("weekend_trend.csv", delimiter=",")

# group the train data by month and year
by_month_train_gp = train_set_clean_df.groupby([train_set_clean_df.index.year, train_set_clean_df.index.month])

# calculate a rolling sum of the count for the last 3 days
window_days = 3
rolling_sum = lambda x: pd.stats.moments.rolling_sum(x, window=window_days*24, min_periods=window_days*24)
train_set_clean_df["3day_sum"] = by_month_train_gp["count"].transform(rolling_sum)

# load the test data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
test_set_orig_df = pd.read_csv("test.csv", parse_dates=["datetime"], index_col="datetime")

# group the test data by month and year
by_month_orig_test_gp = test_set_orig_df.groupby([test_set_orig_df.index.year, test_set_orig_df.index.month])

# resample at 1 hour intervals, missing data has value NaN
test_set_list = []
for month, group in by_month_orig_test_gp:
    test_set_list.append(group.resample("H"))
test_set_clean_df = pd.tools.merge.concat(test_set_list)

# group the new data frame by month and year
by_month_test_gp = test_set_clean_df.groupby([test_set_clean_df.index.year, test_set_clean_df.index.month])

# find all the missing values and create a series of them
missing = test_set_clean_df["season"].isnull()
missing_series = test_set_clean_df["season"][missing == True].fillna(0)
missing_gp = missing_series.groupby([missing_series.index.year, missing_series.index.dayofyear])
print("count of missing values by date")
print(missing_gp.count())

# backfill missing workingday information for previously interpolated values
test_set_clean_df["workingday_clean"] = by_month_test_gp["workingday"].fillna(method="bfill")

# create convenience columns indicating which data set the data came from
# this may also be possible with hierarchical indexing
# using int instead of a string label as that avoids the .values becoming object type
TRAIN_SET, TEST_SET = 0, 1
train_set_clean_df["data_set"] = TRAIN_SET
test_set_clean_df["data_set"] = TEST_SET

# combine the train and test sets and shift the 3day_sum by 1 hour forward
# shift the data 1 hour forward, i.e. the sum at the current time period shouldn't include the current
# doing this seems to break timerange access, i.e. the data is the first data set followed by the second
# the index is not sorted
# this may be why some of the shifting operations were having weird behavior
# not tested, solution maybe to call pandas.DataFrame.sort_index
combined_df = pd.concat([train_set_clean_df, test_set_clean_df])

# including the freq="H" is important, otherwise the behavior is strange
# for example, data is shifted to the start of the next month
combined_df["3day_sum"] = combined_df["3day_sum"].shift(periods=1, freq="H")

# combined_df["3day_sum"].plot()

# concatenating with keys (example below) doesn't work
# later shifting the 3day_sum shifts it to the subsequent month of test, not 1 day ahead into train
# may be due to the method i'm using to concat
# solutions at
# http://stackoverflow.com/questions/23198053/how-do-you-shift-pandas-dataframe-with-a-multiindex
# http://stackoverflow.com/questions/13030245/how-to-shift-a-pandas-multiindex-series
# combined_df = pd.concat([train_set_clean_df, test_set_clean_df], keys=["train", "test"])

by_data_set_gp = combined_df.groupby("data_set")


# does trend prediction on the test data
# modifies the count and 3day_sum columns in place
def test_trend_predict(row):

    time_current = row.name
    time_1h_ago = row.name - pd.offsets.DateOffset(hours=1)
    time_73h_ago = row.name - pd.offsets.DateOffset(hours=24*window_days+1)

    # if 3day_sum isn't known, calculate it using previous sum and count 73 hours ago
    if np.isnan(combined_df.loc[time_current, "3day_sum"]):
        new_3day_sum = combined_df.loc[time_1h_ago, "3day_sum"] + combined_df.loc[time_1h_ago, "count"] \
                        - combined_df.loc[time_73h_ago, "count"]
        combined_df.loc[time_current, "3day_sum"] = new_3day_sum
    else:
        new_3day_sum = row["3day_sum"]

    # predict the current value by scaling the corresponding trend value for that hour by
    # one third of the 3 day sum, i.e. normalize the trend daily sum by using the daily sum for the last
    # 3 days
    if row["workingday_clean"] == 1:  # if it's a work day
        combined_df.loc[time_current, "count"] = workday_trend[time_current.hour] * new_3day_sum / 3
    elif row["workingday_clean"] == 0:  # weekend or holiday
        combined_df.loc[time_current, "count"] = weekend_trend[time_current.hour] * new_3day_sum / 3
    return

# this is very slow for some reason
# maybe faster to extract values, do the computation, and overwrite the values
print("prediction unknown count values using the trend")
by_data_set_gp.get_group(TEST_SET).apply(test_trend_predict, axis=1)

# create month, dayofweek, and hour columns, used for ababoost prediction in place of season
month_f = lambda x: x.index.month
dayofweek_f = lambda x: x.index.dayofweek
hour_f = lambda x: x.index.hour
# could use any column besides "weather", just need it to be a single column so result is a series, not a data frame
combined_df["month"] = combined_df.apply(month_f)["weather"]
combined_df["day_of_week"] = combined_df.apply(dayofweek_f)["weather"]
combined_df["hour"] = combined_df.apply(hour_f)["weather"]

# the columns remaining in X and used for adaboost prediction are
x_columns_list = ["holiday", "weather", "temp", "atemp", "humidity", "windspeed",
                  "workingday_clean", "month", "day_of_week", "hour"]


# use the adaboost regression to predict ratio for the test set
def test_adaboost_predict(row):
    if not np.isnan(row["weather"]):  # check if feature data is available
        ratio = adaboost_regress.predict(row.loc[x_columns_list].values)[0]
        return ratio
    else:  # no feature data available
        return 0  # return 0, i.e. use the trend predicted value

# predict the ratio using adaboost
print("predicting the ratio and count on the test set with adaboost")
combined_df["ratio_adaboost"] = by_data_set_gp.get_group(TEST_SET).apply(test_adaboost_predict, axis=1)
combined_df["count_adaboost"] = ((np.exp(combined_df["ratio_adaboost"])) * (combined_df["count"] + 1)) - 1

# for plotting convenience
combined_df.loc[combined_df["data_set"] == TRAIN_SET, "count_adaboost"] = \
    combined_df.loc[combined_df["data_set"] == TRAIN_SET, "count"]

by_data_set_gp["count_adaboost"].plot()

trend_submit_df = pd.DataFrame(by_data_set_gp.get_group(TEST_SET)["weather"])
trend_submit_df["count"] = by_data_set_gp.get_group(TEST_SET)["count_adaboost"].apply(int)
trend_submit_df = trend_submit_df.dropna()

# save data and predicted trend to a new csv
if not os.path.exists("adaboost_submit.csv") or overwrite:
    trend_submit_df.to_csv("adaboost_submit.csv", columns=["count"])

plt.show()
