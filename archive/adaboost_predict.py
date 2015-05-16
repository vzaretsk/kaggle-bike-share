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

# including the freq="H" is important, otherwise the behavior is strange
# for example, data is shifted to the start of the next month
train_set_clean_df["3day_sum"] = train_set_clean_df["3day_sum"].shift(periods=1, freq="H")

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

# create month, dayofweek, and hour columns, used for ababoost training in place of season
month_f = lambda x: x.index.month
dayofweek_f = lambda x: x.index.dayofweek
hour_f = lambda x: x.index.hour
# could use any column besides "count", just need it to be a single column so result is a series, not a data frame
train_set_clean_df["month"] = train_set_clean_df.apply(month_f)["count"]
train_set_clean_df["day_of_week"] = train_set_clean_df.apply(dayofweek_f)["count"]
train_set_clean_df["hour"] = train_set_clean_df.apply(hour_f)["count"]

# natural log of the ratio of the actual counts and the trend, use for checking results and training
# i originally through linear would be simpler but then realized that the log handles much better
# the symmetry between, for example a ratio of 2 and 0.5
train_set_clean_df["ratio"] = np.log((train_set_clean_df["count"] + 1) / (train_set_clean_df["trend"] + 1))

adaboost_train_set_df = train_set_clean_df.copy()
adaboost_train_set_df = adaboost_train_set_df.drop(["season", "workingday","casual", "registered",
                                                    "count", "ncount", "3day_sum", "trend"], axis=1)
adaboost_train_set_df = adaboost_train_set_df.dropna()

# will use all the remaining columns to predict ratio
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# y : array-like of shape = [n_samples]

adaboost_X = adaboost_train_set_df.drop("ratio", axis=1).values
adaboost_y = adaboost_train_set_df["ratio"].values

# the columns remaining in X and used for adaboost prediction are
x_columns_list = ["holiday", "weather", "temp", "atemp", "humidity", "windspeed",
                  "workingday_clean", "month", "day_of_week", "hour"]

adaboost_regress = sklearn.ensemble.AdaBoostRegressor(
    sklearn.tree.DecisionTreeRegressor(max_depth=13), n_estimators=200)

print("training using {0:} samples with {1:} features each".format(*adaboost_X.shape))
adaboost_regress.fit(adaboost_X, adaboost_y)


# use the adaboost regression to predict ratio for the training set
def train_adaboost_predict(row):
    if not np.isnan(row["weather"]):  # check if feature data is available
        ratio = adaboost_regress.predict(row.loc[x_columns_list].values)[0]
        return ratio
    else:  # no feature data available
        return np.nan

# predict the ratio using adaboost
print("predicting the ratio, count, and new ratio on the training set with adaboost")
train_set_clean_df["ratio_pred_adaboost"] = train_set_clean_df.apply(train_adaboost_predict, axis=1)
train_set_clean_df["count_adaboost"] = ((np.exp(train_set_clean_df["ratio_pred_adaboost"])) * \
                                       (train_set_clean_df["trend"] + 1)) - 1

train_set_clean_df["ratio_adaboost"] = np.log((train_set_clean_df["count"] + 1) / (train_set_clean_df["count_adaboost"] + 1))

score_trend = np.sqrt((train_set_clean_df["ratio"]**2).mean())
print("training trend data score: {:0.4f}".format(score_trend))

score_adaboost = np.sqrt((train_set_clean_df["ratio_adaboost"]**2).mean())
print("training adaboost data score: {:0.4f}".format(score_adaboost))

plt.figure(fig_num)
plt.title("count vs adaboost prediction for training data")
train_set_clean_df["count"].plot(style=".-k", label="count")
train_set_clean_df["count_adaboost"].plot(style="x-r", label="trend")
plt.legend()
fig_num += 1

# some of the biggest outliers are due to tax day marked as not a workday and other rarely celebrated holidays
# fixed by modifying those days in the cleaned training data
plt.figure(fig_num)
plt.title("log ratio_adaboost of count+1/trend+1")
train_set_clean_df["ratio_adaboost"].plot(style=".-b", label="ratio")
fig_num += 1

# pickle the trained adaboost classifier
print("saving adaboost_regress to disk")
sklearn.externals.joblib.dump(adaboost_regress, "adaboost_regress.pkl", compress=9)

plt.show()
