import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as ens
import sklearn.cross_validation as cv
import sklearn.grid_search
import sklearn.metrics as metrics
import os.path
from utility import *

# needed for python multiprocessing
if __name__ == '__main__':

    # SETTINGS AND GLOBAL CONSTANTS

    # if false, will skip grid search, values used in classifier are still optimal since i chose them
    # after running this and also based on submission scores
    do_optimize = False

    # if true, will overwrite existing files
    overwrite = True

    fig_num = 1
    TRAIN_SET, TEST_SET = 0, 1

    # LOAD THE DATA

    # load the data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

    # PREPROCESS THE DATA

    # calculate the proper daily representation of each column
    # interpolate the registered, casual, and count columns, all other columns have already been interpolated
    # by the clean_combine script
    # need to remember to skip bad days and later interpolate again, also discard interpolated values of
    # count, registered, casual in the test set
    daily_combined_df = pd.DataFrame(combined_df[["registered", "casual", "count"]]
                                     .interpolate(method='time').resample("D", how="sum"))
    daily_combined_df[["data_set", "workingday", "holiday", "bad_day",
                       "weekend", "day_of_week", "month"]] = \
        combined_df[["data_set", "workingday", "holiday", "bad_day",
                     "weekend", "day_of_week", "month"]].resample("D", how=None)
    daily_combined_df[["atemp", "humidity", "temp", "weather", "windspeed"]] = \
        combined_df[["atemp", "humidity", "temp", "weather", "windspeed"]].resample("D", how="mean")

    # create elapse day column column
    daily_combined_df["day"] = combined_df["day_of_year"] + 365 * combined_df["year"]

    # replace bad_days with NaN and then interpolate
    daily_combined_df.loc[daily_combined_df["bad_day"] == 1,
                    ["registered", "casual", "count", "atemp", "humidity", "temp", "weather", "windspeed"]] = np.nan
    daily_combined_df.interpolate(method='time', inplace=True)

    # create log registered, casual, count columns
    daily_combined_df[["log_registered", "log_casual", "log_count"]] = \
        np.log(daily_combined_df[["registered", "casual", "count"]] + 1)

    # create shifted weather, etc columns and backfill
    daily_combined_df[["atemp_1d_ago", "humidity_1d_ago", "temp_1d_ago", "weather_1d_ago", "windspeed_1d_ago"]] = \
        daily_combined_df[["atemp", "humidity", "temp", "weather", "windspeed"]].shift(periods=1, freq="D")
    # i briefly tested this and it didn't have much additional predictive value
    # the importances of all the 2d_ago features were about the same as the 1d_ago features
    # daily_combined_df[["atemp_2d_ago", "humidity_2d_ago", "temp_2d_ago", "weather_2d_ago", "windspeed_2d_ago"]] = \
    #     daily_combined_df[["atemp", "humidity", "temp", "weather", "windspeed"]].shift(periods=2, freq="D")
    daily_combined_df.fillna(method="bfill", inplace=True)

    # split the data into the train and test sets
    train_set_df = daily_combined_df.loc[daily_combined_df["data_set"] == TRAIN_SET].copy()
    test_set_df = daily_combined_df.loc[daily_combined_df["data_set"] == TEST_SET].copy()
    # set test set log_registered, log_casual, and log_count to nan, alternatively could drop them
    test_set_df.loc[:, ["log_registered", "log_casual", "log_count"]] = np.nan

    # create the training and test X and training y arrays
    drop_columns = ["registered", "casual", "count", "data_set", "holiday", "bad_day"]
    train_set_df.drop(drop_columns, axis=1, inplace=True)
    test_set_df.drop(drop_columns, axis=1, inplace=True)

    train_X = train_set_df.drop(["log_registered", "log_casual", "log_count"], axis=1).values
    test_X = test_set_df.drop(["log_registered", "log_casual", "log_count"], axis=1).values
    train_y = train_set_df[["log_registered", "log_casual"]].values

    features_lst = list(train_set_df.drop(["log_registered", "log_casual", "log_count"], axis=1).columns)

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram

    if do_optimize:
        search_space = {"max_features": [None],
                        "max_depth": [10],
                        "max_leaf_nodes": [None],
                        "n_estimators": [500]}

        # cv_gs = cv.ShuffleSplit(len(train_y), n_iter=4, test_size=0.35)

        # 4 folds without shuffling gives better cross validation scores
        # cv_gs = cv.KFold(len(train_y), n_folds=4, shuffle=False)

        # custom split based on predicting the first/last 5 days from the remaining of each month
        split_arr = np.array([0, 0, 0, 0, 0,
                              -1, -1, -1, -1, -1, -1, -1, -1, -1,
                              1, 1, 1, 1, 1])
        split_f = lambda x: split_arr[x.index.day - 1]
        test_split = train_set_df.apply(split_f)["atemp"].values
        cv_gs = cv.PredefinedSplit(test_split)
        # split_f = lambda x: (x.index.day  - 1)// 15

        grid_search_gs = sklearn.grid_search.GridSearchCV(ens.RandomForestRegressor(
                            ),
                            search_space, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=1, pre_dispatch="2*n_jobs", verbose=3)

        print("running grid search")
        grid_search_gs.fit(train_X, train_y)

        with open("gs_score_log.txt", "a") as gs_score_log:
            gs_score_log.write("search space:\n")
            gs_score_log.write(str(search_space)+"\n")
            gs_score_log.write("cross validation:\n")
            gs_score_log.write(str(cv_gs)+"\n")
            gs_score_log.write("grid search:\n")
            gs_score_log.write(str(grid_search_gs)+"\n")

            for params, mean_score, scores in grid_search_gs.grid_scores_:
                line = "mean: {:.4f}, stdev: {:.4f} MSE for {:}".format(-mean_score, scores.std(), params)
                print(line)
                gs_score_log.write(line+"\n")

            gs_score_log.write("\n")

        print("best score (RMSE): {:.5f}".format(np.sqrt(-grid_search_gs.best_score_)))

    # INSTANTIATE AND FIT A RANDOM FOREST TO THE REGISTERED AND CASUAL DATA SETS

    # will use a single random forest with multi-output as that may pick up on correlations between the two

    # create the classifier
    rnd_forest_regress = ens.RandomForestRegressor(
        n_estimators=10000, max_features=0.2, max_depth=20, max_leaf_nodes=None, n_jobs=1)

    print("training using {0:} samples with {1:} features each using classifier:".format(*train_X.shape))
    print(rnd_forest_regress)
    rnd_forest_regress.fit(train_X, train_y)

    print("predicting log registered and log casual daily sums on the training set with random forest")
    train_set_df["log_registered_rf"], train_set_df["log_casual_rf"] = \
        rnd_forest_regress.predict(train_X).T

    print("RMSE of registered daily sums: {:.4f}".format(
        np.sqrt(metrics.mean_squared_error(train_y[:, 0], train_set_df["log_registered_rf"].values))))
    print("RMSE of casual daily sums: {:.4f}".format(
        np.sqrt(metrics.mean_squared_error(train_y[:, 1], train_set_df["log_casual_rf"].values))))

    feature_import_lst = list(zip(features_lst, rnd_forest_regress.feature_importances_))
    feature_import_lst.sort(key=lambda x: x[1], reverse=True)
    # print("len features_lst ", len(features_lst))
    # print("len importances ", len(rnd_forest_regress.feature_importances_))
    print("feature importance")
    for feature, value in feature_import_lst:
        print("{}: {:.2f}".format(feature, 100*value))

    # do the log registered and casual daily sum predictions on the test set
    print("predicting log registered and log casual daily sums on the test set with random forest")
    test_set_df["log_registered_rf"], test_set_df["log_casual_rf"] = \
        rnd_forest_regress.predict(test_X).T

    # put both into a single data frame
    daily_pred_df = pd.concat([train_set_df, test_set_df])
    daily_pred_df.sort_index(inplace=True)

    plt.figure(fig_num)
    plt.title("registered log daily sum vs log predicted sum")
    daily_pred_df["log_registered"].plot(style=".-k")
    daily_pred_df["log_registered_rf"].plot(style="-r", linewidth=1)
    fig_num += 1
    plt.figure(fig_num)
    plt.title("casual log daily sum vs log predicted sum")
    daily_pred_df["log_casual"].plot(style=".-k")
    daily_pred_df["log_casual_rf"].plot(style="-r", linewidth=1)
    fig_num += 1

    # NORMALIZE BY THE PREDICTED TREND AND EXTRACT DAILY PATTERNS

    # normalize the hourly data by the predicted daily sum, stack all the days within the four groups,
    # and average together to obtain a typical daily pattern

    # reconstruct the predicted daily sums for registered and casual
    daily_pred_df["registered_rf"] = np.exp(daily_pred_df["log_registered_rf"]) - 1
    daily_pred_df["casual_rf"] = np.exp(daily_pred_df["log_casual_rf"]) - 1

    # resample at 1 hour intervals and forward fill
    hourly_pred_df = daily_pred_df.reindex(combined_df.index).fillna(method="ffill")

    # normalize the registered and casual data by the predicted daily sums
    # reuse the _gp_ names instead of _rf_ for backward compatibility
    combined_df["registered_gp_norm"] = combined_df["registered"] / hourly_pred_df["registered_rf"]
    combined_df["casual_gp_norm"] = combined_df["casual"] / hourly_pred_df["casual_rf"]

    # plt.figure(fig_num)
    # plt.title("rf normalized registered and casual counts")
    # combined_df["registered_gp_norm"].plot(label="registered")
    # combined_df["casual_gp_norm"].plot(label="casual")
    # plt.legend(loc="best")
    # fig_num += 1

    # group by working day and hour
    by_day_and_hour_gp = combined_df.groupby(["workingday", combined_df.index.hour])

    # average to obtain a typical workday and weekend
    registered_hourly_mean = by_day_and_hour_gp["registered_gp_norm"].mean()
    casual_hourly_mean = by_day_and_hour_gp["casual_gp_norm"].mean()

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

    plt.figure(fig_num)
    plt.title("workday and weekend trends")
    plt.plot(range(24), registered_weekday, ".-b", label="registered workday")
    plt.plot(range(24, 48), registered_weekend, ".-b", label="registered weekend")
    plt.plot(range(24), casual_weekday, ".-r", label="casual workday")
    plt.plot(range(24, 48), casual_weekend, ".-r", label="casual weekend")
    plt.legend(loc="best")
    fig_num += 1

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
    combined_df["registered_gp_pred"] = combined_df["log_registered_gp_pred"] * hourly_pred_df["registered_rf"]
    combined_df["casual_gp_pred"] = combined_df["log_casual_gp_pred"] * hourly_pred_df["casual_rf"]
    combined_df["count_gp_pred"] = (combined_df["registered_gp_pred"] +
                                    combined_df["casual_gp_pred"]).apply(np.round).apply(int)

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
    if not os.path.exists("rf_daily_submit.csv") or overwrite:
        gp_submit_df.to_csv("rf_daily_submit.csv", columns=["count_gp_pred"], header=["count"])

    # save the modified data, overwrite the original file
    if not os.path.exists("combined_data.csv") or overwrite:
        combined_df.to_csv("combined_data.csv")

    plt.show()
