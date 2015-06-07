import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gpr
import sklearn.cross_validation as cv
import sklearn.grid_search as gs
import sklearn.metrics as metrics
import os.path

# SETTINGS AND GLOBAL CONSTANTS

# if true, will overwrite existing files
overwrite = True

fig_num = 1
TRAIN_SET, TEST_SET = 0, 1

# LOAD THE DATA

# load the data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

# PREPROCESS THE DATA

# get the counts for good days in the training set
train_set_df = combined_df.loc[(combined_df["data_set"] == TRAIN_SET) & (combined_df["bad_day"] == 0),
                                ["workingday", "casual", "registered", "count"]]

# impute missing values by linear interpolation
train_set_df = train_set_df.interpolate(method="time")

# calculate the sum for each day
daily_sum_df = pd.DataFrame(train_set_df["count"].resample("D", how="sum"))
daily_sum_df["casual"] = train_set_df["casual"].resample("D", how="sum")
daily_sum_df["registered"] = train_set_df["registered"].resample("D", how="sum")
daily_sum_df["workingday"] = train_set_df["workingday"].resample("D", how=None)

# plt.figure(fig_num)
# plt.title("daily usage by weekday and user type")
# daily_sum_df[daily_sum_df["workingday"] == 1]["registered"].plot(style="o-", label="registered weekday")
# daily_sum_df[daily_sum_df["workingday"] == 1]["casual"].plot(style="x-", label="casual weekday")
# daily_sum_df[daily_sum_df["workingday"] == 0]["registered"].plot(style="o--", label="registered weekend")
# daily_sum_df[daily_sum_df["workingday"] == 0]["casual"].plot(style="x--", label="casual weekend")
# plt.legend(loc="upper left")
# fig_num += 1

# get array of elapsed days since jan 1, 2011 for both weekday and weekend
weekdays_X = np.array((daily_sum_df[daily_sum_df["workingday"] == 1].index - daily_sum_df.index[0]) /
                      np.timedelta64(1, "D"))
weekends_X = np.array((daily_sum_df[daily_sum_df["workingday"] == 0].index - daily_sum_df.index[0]) /
                      np.timedelta64(1, "D"))
# change shape from (456,) to (456,1) as that is what gp.fit expects
weekdays_X = np.reshape(weekdays_X, (weekdays_X.shape[0], 1))
weekends_X = np.reshape(weekends_X, (weekends_X.shape[0], 1))

# get arrays of log sums of registered and casual for both weekday and weekend
log_registered_weekday_y = np.log(daily_sum_df[daily_sum_df["workingday"] == 1]["registered"].values + 1)
log_casual_weekday_y = np.log(daily_sum_df[daily_sum_df["workingday"] == 1]["casual"].values + 1)
log_registered_weekend_y = np.log(daily_sum_df[daily_sum_df["workingday"] == 0]["registered"].values + 1)
log_casual_weekend_y = np.log(daily_sum_df[daily_sum_df["workingday"] == 0]["casual"].values + 1)

# create array of all possible elapsed days, this includes the fact that 2012 is a leap year
all_days_X = np.array(range(2*365+1))
all_days_X = np.reshape(all_days_X, (all_days_X.shape[0], 1))

# plt.plot(weekdays_X, log_registered_weekday_y, "o-",
#          weekdays_X, log_casual_weekday_y, "o-",
#          weekends_X, log_registered_weekend_y, "o-",
#          weekends_X, log_casual_weekend_y, "o-")
# plt.show()

# INSTANTIATE AND FIT A GAUSSIAN PROCESS TO EACH DATA SUBSET

# will treat each of {registered, casual} x {weekday, weekend} separately

# instantiate the guassian process and use maximum likelihood estimation to optimize theta0
# corr options are
# "absolute_exponential", "squared_exponential", "generalized_exponential", "cubic", "linear"
# trying a these by hand suggests that "squared_exponential" gives the best looking result
# the rest are too wiggly, i.e. appear to overfit or are too sensitive to outliers,
# increasing the nugget to prevent this causes
# the fit to miss important features, for example overshoot jan 2011
# regr options are "constant", "linear", "quadratic", they don't seem to make a big difference
# random_start=10 seems enough to prevent the algorithm from finding the wrong minimum
gp_registered_weekday = gpr.GaussianProcess(regr="constant", corr="squared_exponential", nugget=0.5,
                                            theta0=1, thetaL=0.001, thetaU=100, verbose=False, random_start=10)
gp_registered_weekend = gpr.GaussianProcess(regr="constant", corr="squared_exponential", nugget=0.5,
                                            theta0=1, thetaL=0.001, thetaU=100, verbose=False, random_start=10)
gp_casual_weekday = gpr.GaussianProcess(regr="constant", corr="squared_exponential", nugget=2.0,
                                        theta0=1, thetaL=0.001, thetaU=100, verbose=False, random_start=10)
gp_casual_weekend = gpr.GaussianProcess(regr="constant", corr="squared_exponential", nugget=2.0,
                                        theta0=1, thetaL=0.001, thetaU=100, verbose=False, random_start=10)

# train on the log of the daily sum instead of the sum itself
# this better targets the kaggle goal
# additionally, since my assumption is that there is a baseline for a particular month and the
# weather multiplicatively modifies the usage, log better reflects this
# observations on gaussian process fitting
# small theta gives a wiggly, overfit curve
# large theta may miss features
# large nugget also causes features to be missed, such as overestimating jan 2011
print("fitting gp for gp_registered_weekday")
gp_registered_weekday.fit(weekdays_X, log_registered_weekday_y)
print("fitting gp for gp_registered_weekend")
gp_registered_weekend.fit(weekends_X, log_registered_weekend_y)
print("fitting gp for gp_casual_weekday")
gp_casual_weekday.fit(weekdays_X, log_casual_weekday_y)
print("fitting gp for gp_casual_weekend")
gp_casual_weekend.fit(weekends_X, log_casual_weekend_y)
print("optimized theta for gp_registered_weekday: {:.3f}".format(gp_registered_weekday.theta_[0, 0]))
print("optimized theta for gp_registered_weekend: {:.3f}".format(gp_registered_weekend.theta_[0, 0]))
print("optimized theta for gp_casual_weekday: {:.3f}".format(gp_casual_weekday.theta_[0, 0]))
print("optimized theta for gp_casual_weekend: {:.3f}".format(gp_casual_weekend.theta_[0, 0]))

# predict the log daily sum, can also predict sigma by setting eval_MSE=True
log_registered_weekday_pred = gp_registered_weekday.predict(weekdays_X)
log_registered_weekend_pred = gp_registered_weekend.predict(weekends_X)
log_casual_weekday_pred = gp_casual_weekday.predict(weekdays_X)
log_casual_weekend_pred = gp_casual_weekend.predict(weekends_X)

print("RMSE of registered weekday sum: {:.4f}".format(np.sqrt(metrics.mean_squared_error(
                                log_registered_weekday_y, log_registered_weekday_pred))))
print("RMSE of registered weekend sum: {:.4f}".format(np.sqrt(metrics.mean_squared_error(
                                log_registered_weekend_y, log_registered_weekend_pred))))
print("RMSE of casual weekday sum: {:.4f}".format(np.sqrt(metrics.mean_squared_error(
                                log_casual_weekday_y, log_casual_weekday_pred))))
print("RMSE of casual weekend sum: {:.4f}".format(np.sqrt(metrics.mean_squared_error(
                                log_casual_weekend_y, log_casual_weekend_pred))))

# redo the log daily sum predictions on the entire interval
log_registered_weekday_pred = gp_registered_weekday.predict(all_days_X)
log_registered_weekend_pred = gp_registered_weekend.predict(all_days_X)
log_casual_weekday_pred = gp_casual_weekday.predict(all_days_X)
log_casual_weekend_pred = gp_casual_weekend.predict(all_days_X)

plt.figure(fig_num)
plt.title("log daily sum vs log predicted sum for registered weekday")
plt.plot(weekdays_X[:, 0], log_registered_weekday_y, ".-k")
plt.plot(all_days_X[:, 0], log_registered_weekday_pred, "-r", linewidth=2)
fig_num += 1
plt.figure(fig_num)
plt.title("log daily sum vs log predicted sum for registered weekend")
plt.plot(weekends_X[:, 0], log_registered_weekend_y, ".-k")
plt.plot(all_days_X[:, 0], log_registered_weekend_pred, "-r", linewidth=2)
fig_num += 1
plt.figure(fig_num)
plt.title("log daily sum vs log predicted sum for casual weekday")
plt.plot(weekdays_X[:, 0], log_casual_weekday_y, ".-k")
plt.plot(all_days_X[:, 0], log_casual_weekday_pred, "-r", linewidth=2)
fig_num += 1
plt.figure(fig_num)
plt.title("log daily sum vs log predicted sum for casual weekend")
plt.plot(weekends_X[:, 0], log_casual_weekend_y, ".-k")
plt.plot(all_days_X[:, 0], log_casual_weekend_pred, "-r", linewidth=2)
fig_num += 1

# obsolete, used to plot upper and lower sigma bounds
# plt.fill_between(daily_sum_full_X[:, 0], log_daily_sum_pred - 1.96 * daily_sum_sigma,
#                  log_daily_sum_pred + 1.96 * daily_sum_sigma, facecolor="b", alpha=0.5)

# NORMALIZE BY THE PREDICTED TREND AND EXTRACT DAILY PATTERNS

# normalize the hourly data by the predicted daily sum, stack all the days within the four groups,
# and average together to obtain a typical daily pattern

# reconstruct the predicted daily sums for registered and casual
working_day_indicator = combined_df["workingday"].resample("D", how=None).values
daily_pred_df = pd.DataFrame({"registered_pred":
                                working_day_indicator * (np.exp(log_registered_weekday_pred) - 1) +
                                (1 - working_day_indicator) * np.exp(log_registered_weekend_pred) - 1,
                              "casual_pred":
                                working_day_indicator * (np.exp(log_casual_weekday_pred) - 1) +
                                (1 - working_day_indicator) * np.exp(log_casual_weekend_pred) - 1},
                             index=pd.date_range(pd.datetime(2011, 1, 1), pd.datetime(2012, 12, 31), freq="D"))

# resample them at 1 hour intervals and forward fill
hourly_pred_df = daily_pred_df.reindex(combined_df.index).fillna(method="ffill")

# normalize the registered and casual data by the predicted daily sums
combined_df["registered_gp_norm"] = combined_df["registered"] / hourly_pred_df["registered_pred"]
combined_df["casual_gp_norm"] = combined_df["casual"] / hourly_pred_df["casual_pred"]

# plt.figure(fig_num)
# plt.title("gp normalized registered and casual log counts")
# combined_df["registered_gp_norm"].plot(label="registered")
# combined_df["casual_gp_norm"].plot(label="casual")
# plt.legend(loc="best")
# fig_num += 1

# group by working day and hour
by_day_and_hour_gp = combined_df.groupby(["workingday", combined_df.index.hour])

# average to obtain a typical workday and weekend
registered_hourly_mean = by_day_and_hour_gp["registered_gp_norm"].mean()
casual_hourly_mean = by_day_and_hour_gp["casual_gp_norm"].mean()

# alternative approach, uses the geometric mean (mean of log) to obtain the typical workday and weekend
# not used since it gave worse submission score as compared to arithmetic mean, see end of file
use_geometric_mean = False
if use_geometric_mean:
    # create temporary columns
    combined_df["log_registered_gp_norm"] = combined_df["log_registered"] - np.log(hourly_pred_df["registered_pred"] + 1)
    combined_df["log_casual_gp_norm"] = combined_df["log_casual"] - np.log(hourly_pred_df["casual_pred"] + 1)

    # average the log counts
    registered_hourly_gmean = by_day_and_hour_gp["log_registered_gp_norm"].mean()
    casual_hourly_gmean = by_day_and_hour_gp["log_casual_gp_norm"].mean()

    # calculate actual count
    registered_hourly_mean = np.exp(registered_hourly_gmean)
    casual_hourly_mean = np.exp(casual_hourly_gmean)

    # drop temporary columns
    combined_df.drop(["log_registered_gp_norm", "log_casual_gp_norm"], axis=1, inplace=True)

registered_weekend = registered_hourly_mean[0].values
registered_weekday = registered_hourly_mean[1].values
casual_weekend = casual_hourly_mean[0].values
casual_weekday = casual_hourly_mean[1].values

print("workday registered sum before normalization: {:0.6f}".format(registered_weekday.sum()))
print("weekend registered sum before normalization: {:0.6f}".format(registered_weekend.sum()))
print("workday casual sum before normalization: {:0.6f}".format(casual_weekday.sum()))
print("weekend casual sum before normalization: {:0.6f}".format(casual_weekend.sum()))

# normalize to a sum of 1
registered_weekend = registered_weekend/registered_weekend.sum()
registered_weekday = registered_weekday/registered_weekday.sum()
casual_weekend = casual_weekend/casual_weekend.sum()
casual_weekday = casual_weekday/casual_weekday.sum()

# plt.figure(fig_num)
# plt.title("workday and weekend trends")
# plt.plot(range(24), registered_weekday, ".-b", label="registered workday")
# plt.plot(range(24, 48), registered_weekend, ".-b", label="registered weekend")
# plt.plot(range(24), casual_weekday, ".-r", label="casual workday")
# plt.plot(range(24, 48), casual_weekend, ".-r", label="casual weekend")
# plt.legend(loc="best")
# fig_num += 1


# helper function to predict the registered and casual log counts
def gp_predict(row):
    hour = row.name.hour
    day = row["workingday"]
    registered = day * registered_weekday[hour] + (1 - day) * registered_weekend[hour]
    casual = day * casual_weekday[hour] + (1 - day) * casual_weekend[hour]
    return pd.Series([registered, casual])

# partial result, column names are placeholders
combined_df[["log_registered_gp_pred", "log_casual_gp_pred"]] = combined_df.apply(gp_predict, axis=1)

# create "registered_gp_pred" and "casual_gp_pred" for submission and plotting
combined_df["registered_gp_pred"] = combined_df["log_registered_gp_pred"] * hourly_pred_df["registered_pred"]
combined_df["casual_gp_pred"] = combined_df["log_casual_gp_pred"] * hourly_pred_df["casual_pred"]
combined_df["count_gp_pred"] = (combined_df["registered_gp_pred"] + combined_df["casual_gp_pred"]).apply(int)

# overwrite placeholder columns with predicted log counts
combined_df["log_registered_gp_pred"] = np.log(combined_df["registered_gp_pred"] + 1)
combined_df["log_casual_gp_pred"] = np.log(combined_df["casual_gp_pred"] + 1)

# drop "registered_gp_norm" and "casual_gp_norm" as they are no longer needed
combined_df.drop(["registered_gp_norm", "casual_gp_norm"], axis=1, inplace=True)

# plt.figure(fig_num)
# combined_df["registered"].plot()
# combined_df["registered_gp_pred"].plot()
# plt.legend(loc="best")
# fig_num += 1
#
# plt.figure(fig_num)
# combined_df["casual"].plot(label="casual")
# combined_df["casual_gp_pred"].plot()
# plt.legend(loc="best")
# fig_num += 1

# create data frame for submission
gp_submit_df = combined_df.loc[(combined_df["data_set"] == TEST_SET) & (combined_df["submit"] == 1),
                               ["count_gp_pred"]]

# drop the remaining unneeded columns
combined_df.drop(["registered_gp_pred", "casual_gp_pred", "count_gp_pred"], axis=1, inplace=True)

# create the log difference columns, these will be the regression classifier targets
combined_df["log_registered_gp_diff"] = combined_df["log_registered"] - combined_df["log_registered_gp_pred"]
combined_df["log_casual_gp_diff"] = combined_df["log_casual"] - combined_df["log_casual_gp_pred"]

# save submission file
if not os.path.exists("gp_submit.csv") or overwrite:
    gp_submit_df.to_csv("gp_submit.csv", columns=["count_gp_pred"], header=["count"])

# save the modified data, overwrite the original file
if not os.path.exists("combined_data.csv") or overwrite:
    combined_df.to_csv("combined_data.csv")

# plt.show()

# submission score history
# gp with registered casual split 0.52064
# gp without registered casual split 0.52292
# gp with registered casual split, geometric averaging 0.52653
# 3 day moving average 0.54306
