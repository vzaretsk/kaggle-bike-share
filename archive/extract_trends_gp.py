import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.gaussian_process as gpr
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.metrics
import os.path

# if true, will mark certain days as workdays that are not marked as such in the training data
# for example tax day
fix_workdays = True

# if true, will overwrite existing files
overwrite = True

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

# mark several rarely celebrated holidays, such as tax day, as work days
if fix_workdays:
    train_set_clean_df["workingday_clean"][pd.datetime(2011, 4, 15, 0):pd.datetime(2011, 4, 15, 23)] = 1
    train_set_clean_df["workingday_clean"][pd.datetime(2012, 4, 16, 0):pd.datetime(2012, 4, 16, 23)] = 1

# calculate the sum for each day
daily_sum_sr = train_set_clean_df["count"].resample("D", how="sum").dropna()

# get arrays of sum and corresponding elapsed days index
# could also subtract daily_sum_sr.index[0]
daily_sum_X = np.array((daily_sum_sr.index - pd.datetime(2011, 1, 1)) / np.timedelta64(1, "D"))
# change shape from (456,) to (456,1) as that is what gp.fit expects
daily_sum_X = np.reshape(daily_sum_X, (daily_sum_X.shape[0], 1))
daily_sum_y = daily_sum_sr.values
log_daily_sum_y = np.log(daily_sum_y + 1)

# this includes the fact that 2012 is a leap year
daily_sum_full_X = np.array(range(2*365+1))
daily_sum_full_X = np.reshape(daily_sum_full_X, (daily_sum_full_X.shape[0], 1))

# for debug only
# plt.plot(daily_sum_X, daily_sum_y,".-")
# plt.show()

# instantiate the guassian process and use maximum likelihood estimation to optimize theta0
# corr options are
# "absolute_exponential", "squared_exponential", "generalized_exponential", "cubic", "linear"
# trying a these by hand suggests that "squared_exponential" gives the best looking result
# the rest are too wiggly, i.e. appear to overfit or are too sensitive to outliers,
# increasing the nugget to prevent this causes
# the fit to miss important features, for example overshoot jan 2011
# regr options are "constant", "linear", "quadratic", they don't seem to make a big difference
# random_start=10 seems enough to prevent it from finding the wrong minimum
# using nugget=0.5 found by grid search below
gaussian_proc_regress = gpr.GaussianProcess(regr="constant", corr="squared_exponential", nugget=0.5,
                                            theta0=1, thetaL=0.001, thetaU=100, verbose=False, random_start=10)
# train on the log of the daily sum instead of the sum itself
# this better targets the kaggle goal
# additionally, since my assuption is that there is a baseline for a particular month and the
# weather multiplicatively modifies the usage, log better reflects this
#gaussian_proc_regress.fit(daily_sum_X, daily_sum_y)
gaussian_proc_regress.fit(daily_sum_X, log_daily_sum_y)
print("optimized theta: {:.3f}".format(gaussian_proc_regress.theta_[0, 0]))

# predict the daily sum and sigma
log_daily_sum_pred, MSE = gaussian_proc_regress.predict(daily_sum_full_X, eval_MSE=True)
#daily_sum_pred, MSE = gaussian_proc_regress.predict(daily_sum_full_X, eval_MSE=True)
daily_sum_sigma = np.sqrt(MSE)

print("root mean squared error of dialy sum: {:.3f}".format(np.sqrt(
                                                sklearn.metrics.mean_squared_error(
                                                log_daily_sum_y, gaussian_proc_regress.predict(daily_sum_X)))))

# initial attempt looks good but will need improvement, probably using cv, to find
# optimal corr, nugget, and maybe theta0
# observations so far
# small theta gives a wiggly, overfit curve
# large theta may miss features
# large nugget also causes features to be missed, such as overestimating jan 2011
plt.figure(fig_num)
plt.title("log daily sum vs log predicted sum")
plt.plot(daily_sum_X[:, 0], log_daily_sum_y, ".-k")
plt.plot(daily_sum_full_X[:, 0], log_daily_sum_pred, "-r", linewidth=2)
plt.fill_between(daily_sum_full_X[:, 0], log_daily_sum_pred - 1.96 * daily_sum_sigma,
                 log_daily_sum_pred + 1.96 * daily_sum_sigma, facecolor="b", alpha=0.5)
fig_num += 1

# take a detour calculate the histogram of the log daily sum - predicted sum
log_daily_diff = log_daily_sum_y - gaussian_proc_regress.predict(daily_sum_X)

plt.figure(fig_num)
plt.title("histogram of log daily sum - log predicted sum")
plt.hist(log_daily_diff, bins=50)
fig_num += 1

# use cross validation to find optimal nugget
# use mean squared error since y values are already log of sum, no need to write custom score

# search_space = {"nugget": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1e0, 2e0, 5e0]}
search_space = {"nugget": [0.2, 0.5, 1.0]}

ss_cv = sklearn.cross_validation.ShuffleSplit(len(log_daily_sum_y), n_iter=1, train_size=0.6)

# attempted to run on two cores with n_jobs=2 but gave errors/strange behavior
# full search takes ~6 min, short 2 nugget value search ~45 sec
gp_grid_search = sklearn.grid_search.GridSearchCV(gpr.GaussianProcess(regr="constant",
                    corr="squared_exponential", nugget=0.1, theta0=1, thetaL=0.001, thetaU=100,
                    verbose=False, random_start=10),
                    search_space, cv=ss_cv, scoring="mean_squared_error", verbose=1)

print("running grid search for optimal nugget value")
gp_grid_search.fit(daily_sum_X, log_daily_sum_y)

# i ran the above grid search for a total of 140 iterations per nugget value, see summary at the end
# of the file, the conclusion is that the optimum is shallow and wide, nugget=0.5 is a good choice
for params, mean_score, scores in gp_grid_search.grid_scores_:
        print("mean: {:.3f}, stdev: {:.3f} MSE for {:}".format(-mean_score, scores.std(), params))

daily_log_full_pred_ts = pd.Series(log_daily_sum_pred,
                              index=pd.date_range(pd.datetime(2011, 1, 1), pd.datetime(2012, 12, 31), freq="D"))

daily_sum_pred = np.exp(log_daily_sum_pred) - 1
daily_sum_full_pred_ts = pd.Series(daily_sum_pred,
                              index=pd.date_range(pd.datetime(2011, 1, 1), pd.datetime(2012, 12, 31), freq="D"))

daily_sum_pred_ts = daily_sum_full_pred_ts.reindex(train_set_clean_df.index)
daily_sum_pred_ts = daily_sum_pred_ts.fillna(method="ffill")

# normalize the count data by the predicted daily sum
train_set_clean_df["ncount_gp"] = train_set_clean_df["count"]/daily_sum_pred_ts

plt.figure(fig_num)
plt.title("daily sum vs predicted sum")
# plt.plot(daily_sum_X[:, 0], daily_sum_y, ".-k")
# plt.plot(daily_sum_full_X[:, 0], daily_sum_pred, "-r", linewidth=2)
daily_sum_sr.plot(style=".-k")
daily_sum_full_pred_ts.plot(style="-r", linewidth=2)
fig_num += 1

plt.figure(fig_num)
plt.title("gaussian process normalized daily count")
train_set_clean_df["ncount_gp"].plot()
fig_num += 1

# group the data by work day or not
by_workingday_gp = train_set_clean_df.groupby("workingday_clean")

# this seems to have a few not usually celebrated holidays, such as tax day
# plt.figure(fig_num)
# by_workingday_gp["ncount"].plot()
# fig_num += 1

weekend_count, workday_count = by_workingday_gp["ncount_gp"].count()

weekends = by_workingday_gp["ncount_gp"].get_group(0).values
workdays = by_workingday_gp["ncount_gp"].get_group(1).values

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
if not os.path.exists("train_clean_gp.csv") or overwrite:
    train_set_clean_df.to_csv("train_clean_gp.csv")

# save gaussian process predicted daily sum to csv
if not os.path.exists("daily_log_sum_pred_gp.csv") or overwrite:
    daily_log_full_pred_ts.to_csv("daily_log_sum_pred_gp.csv", header=["daily_log_sum_gp"], index_label=["datetime"])

# save workday and weekend trends to csv
if not os.path.exists("workday_trend_gp.csv") or overwrite:
    np.savetxt("workday_trend_gp.csv", workday_trend, delimiter=",")
if not os.path.exists("weekend_trend_gp.csv") or overwrite:
    np.savetxt("weekend_trend_gp.csv", weekend_trend, delimiter=",")

plt.show()

# summary of 140 iteration grid search
# nugget	mean MSE
# 0.001	0.058
# 0.002	0.058
# 0.005	0.056
# 0.01	0.060
# 0.02	0.057
# 0.05	0.061
# 0.1	0.057
# 0.2	0.053
# 0.5	0.055
# 1	0.054
# 2	0.057
# 5	0.062
