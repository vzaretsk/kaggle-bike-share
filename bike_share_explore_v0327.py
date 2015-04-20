import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig_num = 1

# load the data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_df = pd.read_csv("train.csv", parse_dates=["datetime"], index_col="datetime")
# train_set_df = pd.read_csv("train.csv")

# train_set_df["count"].plot()
# train_set_df["casual"].plot()
# # secondary_y=True

# copy the count column for later normalization
train_set_df["nmoncount"] = train_set_df["count"]
train_set_df["nweekcount"] = train_set_df["count"]
# group the data by month and year
by_month_gp = train_set_df.groupby([train_set_df.index.year, train_set_df.index.month])
# group the data by week and year
by_week_gp = train_set_df.groupby([train_set_df.index.year, train_set_df.index.week])
# normalize the data by the total count in each month
train_set_df["nmoncount"] = by_month_gp["nmoncount"].transform(lambda x: x / x.sum())
# normalize the data by the total count in each week
train_set_df["nweekcount"] = by_week_gp["nweekcount"].transform(lambda x: x / x.sum())

# CONCLUSION
# some within month trend still remains

# plt.figure(fig_num)
# fig_num += 1
# by_month_gp["count"].plot(title="original counts")
# plt.figure(fig_num)
# fig_num += 1
# by_month_gp["nmoncount"].plot(title="counts normalized by month")

# plt.figure(fig_num)
# plt.title("original week stack")
# fig_num += 1
# for week, group in by_week_gp:
#     plt.plot(group["count"])

# plt.figure(fig_num)
# plt.title("week stack, normalized by month")
# fig_num += 1
weekly_df = pd.DataFrame()
for week, group in by_week_gp:
    year_offset = int(week[0] - 2011)
    week_offset = int(week[1] - 1)
    # before = group.index[0].weekday()
    group.index = group.index - pd.offsets.DateOffset(weeks=week_offset + 52*year_offset)
    # after = group.index[0].weekday()
    # if before-after != 0:
    #     print("before {}, after {}".format(before, after))
    weekly_df[week] = group["nmoncount"]
    # replaced all of the code below with a DataFrame containing weekly data
    # # in some cases the number of hours is significantly different than a multiple of 24
    # # this is due to missing data), I skip plotting those weeks, very crude code
    # missing_hours = len(group["nmoncount"]) % 24
    # if missing_hours not in (0, 23):
    #     print("week {} has {} missing hours, skipping".format(week, missing_hours))
    # else:
    #     plt.plot(group["nmoncount"])

plt.figure(fig_num)
plt.title("weekly average, normalized by month")
fig_num += 1
weekly_df.mean(axis=1).plot()
# for week in weekly_df:
#     weekly_df[week].plot()

# normalizing by week is giving poor results, possibly due to partial weeks
# plt.figure(fig_num)
# plt.title("normalized by week, week stack")
# fig_num += 1
# for week, group in by_week_gp:
#     plt.plot(group["nweekcount"])

monthly_df_list = []
monthly_df = pd.DataFrame()
for month, group in by_month_gp:
    # offset to the same year, month, and to start on Monday for plotting convenience
    year_offset = group.index[0].year - 2011
    month_offset = group.index[0].month - 1
    original_weekday = group.index[0].weekday()
    group.index = group.index - pd.offsets.DateOffset(years=year_offset, months=month_offset)
    new_weekday = group.index[0].weekday()
    group.index = group.index + pd.offsets.DateOffset(days=original_weekday - new_weekday)
    # print(original_weekday, group.index[0].weekday())
    monthly_df_list.append(group["nmoncount"])
    monthly_df[month] = group["nmoncount"]

plt.figure(fig_num)
plt.title("month stack, aligned by weekday")
fig_num += 1
for month in monthly_df_list:
    month.plot()

plt.figure(fig_num)
plt.title("monthly average, normalized by month")
fig_num += 1
monthly_df.mean(axis=1).plot()

plt.show()
