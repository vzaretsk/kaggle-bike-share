import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

# SETTINGS AND GLOBAL CONSTANTS

# if true, will overwrite existing files
overwrite = True

# if true, will mark certain days as workdays that are not marked as such in the training data
# for example tax day
fix_workdays = True

TRAIN_SET, TEST_SET = 0, 1
fig_num = 1

# LOAD SAVED RESULTS AND DATA SETS

# load the cleaned train data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_clean_df = pd.read_csv("train_clean.csv", parse_dates=["datetime"], index_col="datetime")
# no longed needed, a remnant of the old moving average predictor
train_set_clean_df.drop("ncount", axis=1, inplace=True)

# load the saved workday and weekend trends
workday_trend = np.loadtxt("workday_trend_gp.csv", delimiter=",")
weekend_trend = np.loadtxt("weekend_trend_gp.csv", delimiter=",")

# load the saved log sum prediction and calculate the daily sum
daily_sum_gp_df = pd.read_csv("daily_log_sum_pred_gp.csv", parse_dates=["datetime"], index_col="datetime")
daily_sum_gp_df["daily_sum_gp"] = np.exp(daily_sum_gp_df["daily_log_sum_gp"]) - 1

# load the test data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
test_set_orig_df = pd.read_csv("test.csv", parse_dates=["datetime"], index_col="datetime")

# CLEAN UP AND PREPROCESS DATA

# group the train data by month and year
by_month_train_gp = train_set_clean_df.groupby([train_set_clean_df.index.year, train_set_clean_df.index.month])

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

# mark Thanksgiving as holiday
if fix_workdays:
    test_set_clean_df["workingday_clean"][pd.datetime(2011, 11, 25, 0):pd.datetime(2011, 11, 25, 23)] = 0
    test_set_clean_df["workingday_clean"][pd.datetime(2012, 11, 23, 0):pd.datetime(2012, 11, 23, 23)] = 0

# create convenience columns indicating which data set the data came from
# this may also be possible with hierarchical indexing
# using int instead of a string label as that avoids the .values becoming object type
train_set_clean_df["data_set"] = TRAIN_SET
test_set_clean_df["data_set"] = TEST_SET

# combine the train and test sets
# sorting afterward is needed otherwise shift operations don't behave properly
# the is because the index is no longer time sorted but two groups in the order of the concat
combined_df = pd.concat([train_set_clean_df, test_set_clean_df])
combined_df.sort_index(inplace=True)

# including the freq="H" is important, otherwise the behavior is strange
# for example, data is shifted to the start of the next month
# combined_df["example"] = combined_df["example"].shift(periods=1, freq="H")

# concatenating with keys (example below) doesn't work
# solutions at
# http://stackoverflow.com/questions/23198053/how-do-you-shift-pandas-dataframe-with-a-multiindex
# http://stackoverflow.com/questions/13030245/how-to-shift-a-pandas-multiindex-series
# combined_df = pd.concat([train_set_clean_df, test_set_clean_df], keys=["train", "test"])

# PREDICT THE DAILY COUNT USING THE DAILY TRENDS AND GP PREDICTED LOG SUM


# does trend prediction on the test data
# uses the gp trend and workday/weekend day patterns
def test_trend_predict(row):
    row_datetime = row.name
    # predict the count by scaling the corresponding daily trend value for that hour by
    # the gp predicted trend
    if row["workingday_clean"] == 1:  # if it's a work day
        return workday_trend[row_datetime.hour] * daily_sum_gp_df["daily_sum_gp"][row_datetime.date()]
    elif row["workingday_clean"] == 0:  # weekend or holiday
        return weekend_trend[row_datetime.hour] * daily_sum_gp_df["daily_sum_gp"][row_datetime.date()]

combined_df["count_gp"] = combined_df.apply(test_trend_predict, axis=1)
combined_df["log_count"] = np.log(combined_df["count"] + 1)
combined_df["log_count_gp"] = np.log(combined_df["count_gp"] + 1)

by_data_set_gp = combined_df.groupby("data_set")

# plt.figure(fig_num)
# plt.title("gp predicted count and actual count")
# by_data_set_gp["count_gp"].plot()
# combined_df["count"].plot()
# fig_num += 1

# plt.figure(fig_num)
# plt.title("gp predicted log count and actual log count")
# by_data_set_gp["log_count_gp"].plot()
# combined_df["log_count"].plot()
# fig_num += 1

score = np.sqrt(((by_data_set_gp.get_group(TRAIN_SET)["log_count"] -
                  by_data_set_gp.get_group(TRAIN_SET)["log_count_gp"])**2).mean())
print("training data score: {:0.5f}".format(score))

# CREATE THE SUBMISSION FILE

# need weather to keep track of which dates to drop, could also use reindex
# by first storing the original index
trend_submit_df = pd.DataFrame(by_data_set_gp.get_group(TEST_SET)["weather"])
trend_submit_df["count"] = by_data_set_gp.get_group(TEST_SET)["count_gp"].apply(int)
trend_submit_df = trend_submit_df.dropna()

# save data and predicted trend to a new csv
if not os.path.exists("trend_submit_gp.csv") or overwrite:
    trend_submit_df.to_csv("trend_submit_gp.csv", columns=["count"])
# gp submission score 0.52292
# 3 day moving average submission score 0.54306

# save the cleaned combined file
if not os.path.exists("combined_gp.csv") or overwrite:
    combined_df.to_csv("combined_gp.csv")

plt.show()
