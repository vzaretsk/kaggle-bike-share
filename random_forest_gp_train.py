import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.ensemble as ens
import sklearn.externals
import sklearn.cross_validation as cv
import sklearn.grid_search
import sklearn.metrics
import os.path


# needed for python multiprocessing
if __name__ == '__main__':

    # SETTINGS AND GLOBAL CONSTANTS

    # if false, will skip grid search, values used in classifier are still optimimal since
    # i already ran grid search and hard coded the results into the code
    # this is also VERY slow, takes about half a day on my laptop
    # i may be overdoing it by using too many estimators
    do_optimize = False

    # if true, will overwrite existing files
    overwrite = True

    TRAIN_SET, TEST_SET = 0, 1
    fig_num = 1

    # LOAD SAVED RESULTS AND DATA SETS

    # load the cleaned data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_gp.csv", parse_dates=["datetime"], index_col="datetime")

    # CLEAN UP AND PREPROCESS DATA

    # these columns are not needed
    combined_df.drop(["season", "workingday_clean", "casual", "registered"], axis=1, inplace=True)

    # create year, month, dayofweek, and hour columns, used for classifier training in place of season
    year_f = lambda x: x.index.year
    month_f = lambda x: x.index.month
    dayofweek_f = lambda x: x.index.dayofweek
    # dayofyear_f = lambda x: x.index.dayofyear
    hour_f = lambda x: x.index.hour
    # could use any column besides "count", just need it to be a single column so result is a series, not a data frame
    combined_df["year"] = combined_df.apply(month_f)["count"]
    combined_df["month"] = combined_df.apply(month_f)["count"]
    combined_df["day_of_week"] = combined_df.apply(dayofweek_f)["count"]
    # combined_df["day_of_year"] = combined_df.apply(dayofyear_f)["count"]
    combined_df["hour"] = combined_df.apply(hour_f)["count"]

    # create 24h shifted versions of humidity, weather, atemp as potentially useful features
    # other than a two outliers of many consecutive missing hours,
    # missing values are either a single hour or two consecutive hours
    # missing value imputation by linear interpolation seems simplest
    combined_df["humidity_1day_ago"] =\
        combined_df["humidity"].interpolate(method='time').shift(periods=24, freq="H")
    combined_df["weather_1day_ago"] =\
        combined_df["weather"].interpolate(method='time').apply(int).shift(periods=24, freq="H")
    combined_df["atemp_1day_ago"] =\
        combined_df["atemp"].interpolate(method='time').shift(periods=24, freq="H")

    # combined_df["atemp_1day_ago"].plot()
    # combined_df["atemp"].interpolate(method='time').plot()
    # plt.show()

    # OPTIMIZE AND TRAIN THE CLASSIFIER

    by_data_set_gp = combined_df.groupby("data_set")

    rnd_forest_train_set_df = by_data_set_gp.get_group(TRAIN_SET).copy()
    rnd_forest_test_set_df = by_data_set_gp.get_group(TEST_SET).copy()

    # create log count difference column, this will be the regression target
    rnd_forest_train_set_df["log_count_diff"] = rnd_forest_train_set_df["log_count"] - \
                                              rnd_forest_train_set_df["log_count_gp"]

    # drop all remaining unneeded columns
    rnd_forest_train_set_df = rnd_forest_train_set_df.drop(["count", "data_set", "count_gp"], axis=1)
    rnd_forest_train_set_df = rnd_forest_train_set_df.dropna()

    rnd_forest_test_set_df = rnd_forest_test_set_df.drop(["count", "data_set", "count_gp"], axis=1)
    # test set doesn't have log_count values
    rnd_forest_test_set_df = rnd_forest_test_set_df.drop(["log_count"], axis=1)
    rnd_forest_test_set_df = rnd_forest_test_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples]
    rnd_forest_X = rnd_forest_train_set_df.drop(["log_count_diff", "log_count", "log_count_gp"], axis=1).values
    rnd_forest_test_X = rnd_forest_test_set_df.drop(["log_count_gp"], axis=1).values
    features_lst = list(rnd_forest_test_set_df.drop(["log_count_gp"], axis=1).columns)
    rnd_forest_y = rnd_forest_train_set_df["log_count_diff"].values

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram
    if do_optimize:
        search_space = {"max_features": [None],
                        "max_depth": [15],
                        "max_leaf_nodes": [None],
                        "n_estimators": [200]}

        # search_space = {"max_features": [None, 3],
        #                 "max_depth": [None],
        #                 "max_leaf_nodes": [None],
        #                 "n_estimators": [500]}

        # search_space = {"max_features": [0.6, 0.8, 1.0],
        #                 "max_depth": [10, 50, 100],
        #                 "max_leaf_nodes": [None],
        #                 "n_estimators": [500]}

        # cv_gs = cv.ShuffleSplit(len(rnd_forest_y), n_iter=5, test_size=0.35)
        cv_gs = cv.KFold(len(rnd_forest_y), n_folds=4, shuffle=False)

        gp_grid_search = sklearn.grid_search.GridSearchCV(ens.RandomForestRegressor(
                            ),
                            search_space, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=2, pre_dispatch="2*n_jobs", verbose=3)

        print("running grid search")
        gp_grid_search.fit(rnd_forest_X, rnd_forest_y)

        with open("gs_score_log.txt", "a") as gs_score_log:
            gs_score_log.write("search space:\n")
            gs_score_log.write(str(search_space)+"\n")
            gs_score_log.write("cross validation:\n")
            gs_score_log.write(str(cv_gs)+"\n")
            gs_score_log.write("grid search:\n")
            gs_score_log.write(str(gp_grid_search)+"\n")

            for params, mean_score, scores in gp_grid_search.grid_scores_:
                line = "mean: {:.4f}, stdev: {:.4f} MSE for {:}".format(-mean_score, scores.std(), params)
                print(line)
                gs_score_log.write(line+"\n")

        print("best score (RMSE): {:.5f}".format(np.sqrt(-gp_grid_search.best_score_)))

    # TRAIN THE OPTIMUM CLASSIFIER

    # create the classifier
    rnd_forest_regress = ens.RandomForestRegressor(
        n_estimators=200, max_features=None, max_depth=15, max_leaf_nodes=None, n_jobs=3)

    print("training using {0:} samples with {1:} features each using classifier:".format(*rnd_forest_X.shape))
    print(rnd_forest_regress)
    rnd_forest_regress.fit(rnd_forest_X, rnd_forest_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the random forest regression to predict ratio for the training set
    # def train_rnd_forest_predict(row):
    #     return rnd_forest_regress.predict(row.loc[x_columns_list].values)[0]

    # predict the log count diff using random forest
    print("predicting the log count difference, log count, and count on the training set with random forest")
    rnd_forest_train_set_df["pred_log_count_diff"] = rnd_forest_regress.predict(rnd_forest_X)
    rnd_forest_train_set_df["pred_log_count"] = rnd_forest_train_set_df["pred_log_count_diff"] + \
                                              rnd_forest_train_set_df["log_count_gp"]
    rnd_forest_train_set_df["pred_count"] = np.exp(rnd_forest_train_set_df["pred_log_count"]) - 1

    score = np.sqrt(((rnd_forest_train_set_df["log_count"] -
                      rnd_forest_train_set_df["pred_log_count"])**2).mean())
    print("training data score (RMSE): {:0.5f}".format(score))

    # rnd_forest_train_set_df["count"] = np.exp(rnd_forest_train_set_df["log_count"]) - 1
    # plt.figure(fig_num)
    # plt.title("log count vs predicted log count for training data")
    # rnd_forest_train_set_df["count"].plot(style=".-k", label="count")
    # rnd_forest_train_set_df["pred_count"].plot(style="x-r", label="prediction")
    # plt.legend()
    # fig_num += 1

    plt.figure(fig_num)
    plt.title("log count - predicted log count for training data")
    (rnd_forest_train_set_df["log_count"] - rnd_forest_train_set_df["pred_log_count"]).plot(style=".-k")
    fig_num += 1

    # # pickle the trained random forest classifier
    # print("saving rnd_forest_regress to disk")
    # sklearn.externals.joblib.dump(rnd_forest_regress, "rnd_forest_gp_regress.pkl", compress=9)

    # PREDICT THE TEST SET

    # predict the log count diff using random forest, calculate count
    print("predicting the log count difference, log count, and count on the test set with random forest")
    rnd_forest_test_set_df["pred_log_count_diff"] = rnd_forest_regress.predict(rnd_forest_test_X)
    rnd_forest_test_set_df["pred_log_count"] = rnd_forest_test_set_df["pred_log_count_diff"] + \
                                              rnd_forest_test_set_df["log_count_gp"]
    rnd_forest_test_set_df["count"] = (np.exp(rnd_forest_test_set_df["pred_log_count"]) - 1).apply(int)

    feature_import_lst = list(zip(features_lst, rnd_forest_regress.feature_importances_))
    feature_import_lst.sort(key=lambda x: x[1], reverse=True)
    print("feature importance")
    for feature, value in feature_import_lst:
        print("{}: {:.2f}".format(feature, 100*value))

    # save data and predicted trend to a new csv
    if not os.path.exists("rnd_forest_gp_submit.csv") or overwrite:
        rnd_forest_test_set_df.to_csv("rnd_forest_gp_submit.csv", columns=["count"])

    plt.show()

# rnd forest max_depth=100, max_features=1.0, max_leaf_nodes=None, n_estimators=500 score 0.42547
# adaboost max_depth=500, n_estimators=2000, loss="exponential" submission score 0.43043
# adaboost max_depth=50, n_estimators=500, loss="exponential" submission score 0.43066
# adaboost max_depth=500, n_estimators=2000, loss="linear" submission score 0.43089
# adaboost max_depth=100, n_estimators=2000, loss="exponential" submission score 0.43154
# adaboost max_depth=50, n_estimators=2000, learning_rate=0.4, loss="exponential" submission score 0.43167
# adaboost max_depth=10, n_estimators=2000, loss="exponential" submission score 0.43461

# gaussian process submission score 0.52292
# 3 day moving average submission score 0.54306

