import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

fig_num = 1

# load the cleaned data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_clean_df = pd.read_csv("train_clean.csv", parse_dates=["datetime"], index_col="datetime")

# load the saved workday and weekend trends
workday_trend = np.loadtxt("workday_trend.csv", delimiter=",")
weekend_trend = np.loadtxt("weekend_trend.csv", delimiter=",")

# group the data by month and year
by_month_gp = train_set_clean_df.groupby([train_set_clean_df.index.year, train_set_clean_df.index.month])

# calculate a rolling sum of the count for the last 3 days
window_days = 3
rolling_sum = lambda x: pd.stats.moments.rolling_sum(x, window=window_days*24, min_periods=window_days*24)
train_set_clean_df["3day_sum"] = by_month_gp["count"].transform(rolling_sum)
# shift the data 1 hour forward, i.e. the sum at the current time period shouldn't include the current
# time period itself
train_set_clean_df["3day_sum"] = train_set_clean_df["3day_sum"].shift(periods=1)

# create hour column, useful for trend predicting
# this is inefficient, code below is better
# hour = lambda x: x.index.hour
# # could use any column besides "count", just need it to be a single column so result is a series, not a data frame
# train_set_clean_df["hour"] = train_set_clean_df.apply(hour)["count"]

# alternatively may access index by doing (pseudocode, don't uncomment)
# hour = lambda x: x.name.hour
# df.apply(hour, axis=1)

# train_set_clean_df["3day_sum"].plot()


def train_trend_predict(row):
    # predict the current value by scaling the corresponding trend value for that hour by
    # one third of the 3 day sum, i.e. normalize the trend daily sum by using the daily sum for the last
    # 3 days
    if row["workingday_clean"] == 1 and not np.isnan(row["3day_sum"]):  # if it's a work day
        return workday_trend[row.name.hour] * row["3day_sum"] / 3
    elif row["workingday_clean"] == 0 and not np.isnan(row["3day_sum"]):  # weekend or holiday
        return weekend_trend[row.name.hour] * row["3day_sum"] / 3
    else:  # something else, should only occur when 3day_sum is is nan
        return np.nan


# calculate the trend predicted values for all available data
train_set_clean_df["trend"] = train_set_clean_df.apply(train_trend_predict, axis=1)

plt.figure(fig_num)
plt.title("count vs trend prediction for training data")
train_set_clean_df["count"].plot(style=".-b", label="count")
train_set_clean_df["trend"].plot(style="x-r", label="trend")
plt.legend()
fig_num += 1

plt.show()