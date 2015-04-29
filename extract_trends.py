import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

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

# plt.figure(fig_num)
# plt.title("original and interpolated values")
# train_set_orig_df["count"][:pd.datetime(2011, 1, 20)].plot(style="or")
# train_set_clean_df["count"][:pd.datetime(2011, 1, 20)].plot(style="-+b")
# fig_num += 1

# plt.show()

# backfill missing workingday information for previously interpolated values
train_set_clean_df["workingday_clean"] = by_month_gp["workingday"].fillna(method="bfill")

# group the data by work day or not
by_workingday_gp = train_set_clean_df.groupby("workingday_clean")

# this seems to have a few not usually celebrated holidays, such as tax day
# plt.figure(fig_num)
# by_workingday_gp["ncount"].plot()
# fig_num += 1

weekend_count, workday_count = by_workingday_gp["ncount"].count()

weekends = by_workingday_gp["ncount"].get_group(0).values
workdays = by_workingday_gp["ncount"].get_group(1).values

weekends = weekends.reshape((weekend_count//24, 24))
workdays = workdays.reshape((workday_count//24, 24))

# average to obtain a typical workday and weekend, normalize to a sum of 1
workday_trend = workdays.mean(axis=0)
print("workday sum before normalization: {:0.6f}".format(workday_trend.sum()))
workday_trend = workday_trend/workday_trend.sum()

weekend_trend = weekends.mean(axis=0)
print("weekend sum before normalization: {:0.6f}".format(weekend_trend.sum()))
weekend_trend = weekend_trend/weekend_trend.sum()

# may be able to obtain the same result by grouping by workday+hour and averaging across hours
# also I ignored daylight savings, might need to check what effect this had

plt.figure(fig_num)
plt.title("workday and weekend trends")
plt.plot(range(24), workday_trend, ".-", label="workday")
plt.plot(range(24, 48), weekend_trend, ".-", label="weekend")
plt.legend()
fig_num += 1

# save cleaned data to a new csv
if not os.path.exists("train_clean.csv"):
    train_set_clean_df.to_csv("train_clean.csv")

# save workday and weekend trends to csv
if not os.path.exists("workday_trend.csv"):
    np.savetxt("workday_trend.csv", workday_trend, delimiter=",")
if not os.path.exists("weekend_trend.csv"):
    np.savetxt("weekend_trend.csv", workday_trend, delimiter=",")

plt.show()
