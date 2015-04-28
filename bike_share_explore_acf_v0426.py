import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts

fig_num = 1

# load the data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_orig_df = pd.read_csv("train.csv", parse_dates=["datetime"], index_col="datetime")

# group the data by month and year
by_month_orig_gp = train_set_orig_df.groupby([train_set_orig_df.index.year, train_set_orig_df.index.month])

# resample at 1 hour intervals, missing data has value NaN
train_set_list = []
for month, group in by_month_orig_gp:
    train_set_list.append(group.resample("H"))
train_set_clean_df = pd.tools.merge.concat(train_set_list)

# group the new data frame by month and year
by_month_gp = train_set_clean_df.groupby([train_set_clean_df.index.year, train_set_clean_df.index.month])

# find all the missing values and create a series of them
missing = train_set_clean_df["count"].isnull()
missing_series = train_set_clean_df["count"][missing == True].fillna(0)
missing_gp = missing_series.groupby([missing_series.index.year, missing_series.index.dayofyear])
print("count of missing values by date")
print(missing_gp.count())

# other than a single day in Jan 2011, missing values are either a single hour or two consecutive hours
# missing value imputation by linear interpolation seems simplest
train_set_clean_df["count"] = by_month_gp["count"].apply(pd.Series.interpolate, method="time")

# obsolete, replaced by missing value imputation
# # fill missing values with the mean of the count for that month
# mean_fill = lambda x: x.fillna(x.mean())
# # try rolling_mean here
# train_set_clean_df["count"] = by_month_gp["count"].transform(mean_fill)

# copy the count column for normalization
train_set_clean_df["ncount"] = train_set_clean_df["count"]
# normalize the data by the total count in each month
train_set_clean_df["ncount"] = by_month_gp["ncount"].transform(lambda x: x / x.sum())

print("difference between interpolated and original sum")
print(by_month_orig_gp["count"].sum()[:5])
print(by_month_gp["count"].sum()[:5] - by_month_orig_gp["count"].sum()[:5])

print("difference between interpolated and original mean")
print(by_month_orig_gp["count"].mean()[:5])
print(by_month_gp["count"].mean()[:5] - by_month_orig_gp["count"].mean()[:5])

plt.figure(fig_num)
plt.title("original and interpolated values")
train_set_orig_df["count"][:pd.datetime(2011, 1, 20)].plot(style="or")
train_set_clean_df["count"][:pd.datetime(2011, 1, 20)].plot(style="-+b")
fig_num += 1

plt.show()
