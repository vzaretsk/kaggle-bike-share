import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.ensemble as ens
import sklearn.externals
import sklearn.cross_validation
import sklearn.grid_search
import sklearn.metrics
import os.path


# define an ens.AdaBoostRegressor derived class that takes max_depth as a decision tree parameter
# some of the functions of ens.AdaBoostRegressor no longer properly work
# for example, any parameter i want to grid search over needs to be explicitly listed in the __init__
class AdaBoostRegressorDTree(ens.AdaBoostRegressor):
    def __init__(self, max_depth=None, n_estimators=50, loss="linear", learning_rate=1.0, **kwargs):
        super(AdaBoostRegressorDTree, self).__init__(
            base_estimator=tree.DecisionTreeRegressor(max_depth=max_depth), **kwargs)

# needed for python multiprocessing
if __name__ == '__main__':

    # SETTINGS AND GLOBAL CONSTANTS

    # if true, will overwrite existing files
    overwrite = True

    TRAIN_SET, TEST_SET = 0, 1
    fig_num = 1

    # if false, will skip grid search, values used in classifier are still optimimal since
    # i already ran grid search and hard coded the results into the code
    # this is also VERY slow, takes about half a day on my laptop
    # i may be overdoing it by using too many estimators
    do_optimize = False

    # LOAD SAVED RESULTS AND DATA SETS

    # load the cleaned data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_gp.csv", parse_dates=["datetime"], index_col="datetime")

    # CLEAN UP AND PREPROCESS DATA

    # these columns are not needed
    combined_df.drop(["season", "workingday", "casual", "registered"], axis=1, inplace=True)

    # create year, month, dayofweek, and hour columns, used for classifier training in place of season
    year_f = lambda x: x.index.year
    month_f = lambda x: x.index.month
    dayofweek_f = lambda x: x.index.dayofweek
    dayofyear_f = lambda x: x.index.dayofyear
    hour_f = lambda x: x.index.hour
    # could use any column besides "count", just need it to be a single column so result is a series, not a data frame
    combined_df["year"] = combined_df.apply(month_f)["count"]
    combined_df["month"] = combined_df.apply(month_f)["count"]
    combined_df["day_of_week"] = combined_df.apply(dayofweek_f)["count"]
    combined_df["day_of_year"] = combined_df.apply(dayofyear_f)["count"]
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

    adaboost_train_set_df = by_data_set_gp.get_group(TRAIN_SET).copy()
    adaboost_test_set_df = by_data_set_gp.get_group(TEST_SET).copy()

    # create log count difference column, this will be the regression target
    adaboost_train_set_df["log_count_diff"] = adaboost_train_set_df["log_count"] - \
                                              adaboost_train_set_df["log_count_gp"]

    # drop all remaining unneeded columns, holiday is very redundant with workingday_clean
    adaboost_train_set_df = adaboost_train_set_df.drop(["count", "data_set", "holiday", "count_gp"], axis=1)
    adaboost_train_set_df = adaboost_train_set_df.dropna()

    adaboost_test_set_df = adaboost_test_set_df.drop(["count", "data_set", "holiday", "count_gp"], axis=1)
    # test set doesn't have log_count values
    adaboost_test_set_df = adaboost_test_set_df.drop(["log_count"], axis=1)
    adaboost_test_set_df = adaboost_test_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples]
    adaboost_X = adaboost_train_set_df.drop(["log_count_diff", "log_count", "log_count_gp"], axis=1).values
    adaboost_test_X = adaboost_test_set_df.drop(["log_count_gp"], axis=1).values
    features_lst = list(adaboost_test_set_df.drop(["log_count_gp"], axis=1).columns)
    adaboost_y = adaboost_train_set_df["log_count_diff"].values

    # the columns remaining in X and used for adaboost prediction are
    # x_columns_list = ["atemp", "humidity", "temp", "weather", "windspeed", "workingday_clean", "year", "month",
    #                   "day_of_week", "hour", "humidity_1day_ago", "weather_1day_ago", "atemp_1day_ago"]

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram
    if do_optimize:
        search_space = {"max_depth": [50],
                "n_estimators": [200],
                "learning_rate": [0.4],
                "loss": ["exponential"]}
        # search_space = {"max_depth": [50],
        #         "n_estimators": [2000],
        #         "learning_rate": [0.2, 0.4, 0.6, 0.8, 1.0],
        #         "loss": ["exponential"]}
        # this one takes many hours to run
        # search_space = {"max_depth": [2, 5, 10, 20, 50, 100, 200, 500],
        #                 "n_estimators": [10, 20, 50, 100, 200, 500, 1000, 2000],
        #                 "loss": ["exponential"]}

        ss_cv = sklearn.cross_validation.ShuffleSplit(len(adaboost_y), n_iter=5, train_size=0.65)

        gp_grid_search = sklearn.grid_search.GridSearchCV(AdaBoostRegressorDTree(
                            max_depth=13, n_estimators=200, loss="exponential"),
                            search_space, cv=ss_cv, scoring="mean_squared_error",
                            refit=False, n_jobs=2, pre_dispatch="2*n_jobs", verbose=3)

        print("running grid search for optimal max_depth and n_estimators values")
        gp_grid_search.fit(adaboost_X, adaboost_y)

        with open("gs_score_log.txt", "a") as gs_score_log:
            gs_score_log.write("search space:\n")
            gs_score_log.write(str(search_space)+"\n")
            gs_score_log.write("cross validation:\n")
            gs_score_log.write(str(ss_cv)+"\n")
            gs_score_log.write("grid search:\n")
            gs_score_log.write(str(gp_grid_search)+"\n")

            for params, mean_score, scores in gp_grid_search.grid_scores_:
                line = "mean: {:.4f}, stdev: {:.4f} MSE for {:}".format(-mean_score, scores.std(), params)
                print(line)
                gs_score_log.write(line+"\n")

        # the following code is a messy way to get the grid search results into an array for plotting
        # only works when search is across max_depth and n_estimators
        plot_gs = False
        if plot_gs:
            max_d_dct = dict(zip(search_space["max_depth"], range(len(search_space["max_depth"]))))
            n_iter_dct = dict(zip(search_space["n_estimators"], range(len(search_space["n_estimators"]))))

            gs_values = np.zeros((len(search_space["max_depth"]), len(search_space["n_estimators"])))

            for params, mean_score, scores in gp_grid_search.grid_scores_:
                gs_values[max_d_dct[params["max_depth"]],
                          n_iter_dct[params["n_estimators"]]] = mean_score

            plt.figure(fig_num)
            plt.title("grid search MSE scores")
            plt.xlabel("n_estimators")
            plt.ylabel("max_depth")
            plt.pcolor(gs_values)
            plt.xticks(np.arange(0.5, len(n_iter_dct), 1), search_space["n_estimators"])
            plt.yticks(np.arange(0.5, len(max_d_dct), 1), search_space["max_depth"])
            plt.colorbar()
            plt.tight_layout()
            plt.savefig("gs_ada_scores.png")
            fig_num += 1

            # save the grid search scores to disk
            np.savetxt("gs_ada_scores.csv", gs_values, delimiter=",")
            pd.Series(search_space["n_estimators"], name="n_estimators").to_csv("gs_ada_n_estimators.csv", header=True)
            pd.Series(search_space["max_depth"], name="max_depth").to_csv("gs_ada_max_depth.csv", header=True)

    # TRAIN THE OPTIMUM CLASSIFIER

    # create the classifier
    adaboost_regress = ens.AdaBoostRegressor(
        tree.DecisionTreeRegressor(max_depth=50), n_estimators=200, learning_rate=0.4, loss="exponential")

    print("training using {0:} samples with {1:} features each using classifier:".format(*adaboost_X.shape))
    print(adaboost_regress)
    adaboost_regress.fit(adaboost_X, adaboost_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the adaboost regression to predict ratio for the training set
    # def train_adaboost_predict(row):
    #     return adaboost_regress.predict(row.loc[x_columns_list].values)[0]

    # predict the log count diff using adaboost
    print("predicting the log count difference, log count, and count on the training set with adaboost")
    adaboost_train_set_df["pred_log_count_diff"] = adaboost_regress.predict(adaboost_X)
    adaboost_train_set_df["pred_log_count"] = adaboost_train_set_df["pred_log_count_diff"] + \
                                              adaboost_train_set_df["log_count_gp"]
    adaboost_train_set_df["pred_count"] = np.exp(adaboost_train_set_df["pred_log_count"]) - 1

    score = np.sqrt(((adaboost_train_set_df["log_count"] -
                      adaboost_train_set_df["pred_log_count"])**2).mean())
    print("training data score: {:0.5f}".format(score))

    adaboost_train_set_df["count"] = np.exp(adaboost_train_set_df["log_count"]) - 1
    plt.figure(fig_num)
    plt.title("log count vs predicted log count for training data")
    adaboost_train_set_df["count"].plot(style=".-k", label="count")
    adaboost_train_set_df["pred_count"].plot(style="x-r", label="prediction")
    plt.legend()
    fig_num += 1

    plt.figure(fig_num)
    plt.title("log count - predicted log count for training data")
    (adaboost_train_set_df["log_count"] - adaboost_train_set_df["pred_log_count"]).plot(style=".-k")
    fig_num += 1

    # # pickle the trained adaboost classifier
    # print("saving adaboost_regress to disk")
    # sklearn.externals.joblib.dump(adaboost_regress, "adaboost_gp_regress.pkl", compress=9)

    # PREDICT THE TEST SET

    # predict the log count diff using adaboost, calculate count
    print("predicting the log count difference, log count, and count on the test set with adaboost")
    adaboost_test_set_df["pred_log_count_diff"] = adaboost_regress.predict(adaboost_test_X)
    adaboost_test_set_df["pred_log_count"] = adaboost_test_set_df["pred_log_count_diff"] + \
                                              adaboost_test_set_df["log_count_gp"]
    adaboost_test_set_df["count"] = (np.exp(adaboost_test_set_df["pred_log_count"]) - 1).apply(int)

    feature_import_lst = list(zip(features_lst, adaboost_regress.feature_importances_))
    feature_import_lst.sort(key=lambda x: x[1], reverse=True)
    print("feature importance")
    for feature, value in feature_import_lst:
        print("{}: {:.2f}".format(feature, 100*value))

    # save data and predicted trend to a new csv
    if not os.path.exists("adaboost_gp_submit.csv") or overwrite:
        adaboost_test_set_df.to_csv("adaboost_gp_submit.csv", columns=["count"])

    plt.show()

# adaboost max_depth=500, n_estimators=2000, loss="exponential" submission score 0.43043
# adaboost max_depth=50, n_estimators=500, loss="exponential" submission score 0.43066
# adaboost max_depth=500, n_estimators=2000, loss="linear" submission score 0.43089
# adaboost max_depth=100, n_estimators=2000, loss="exponential" submission score 0.43154
# adaboost max_depth=50, n_estimators=2000, learning_rate=0.4, loss="exponential" submission score 0.43167
# adaboost max_depth=10, n_estimators=2000, loss="exponential" submission score 0.43461

# gaussian process submission score 0.52292
# 3 day moving average submission score 0.54306

