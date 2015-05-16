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

    # if false, will skip grid search, values used in classifier are still optimimal since i chose them
    # after running this and also based on submission scores
    do_optimize = False

    # if true, will overwrite existing files
    overwrite = True

    TRAIN_SET, TEST_SET = 0, 1
    fig_num = 1

    # LOAD SAVED RESULTS AND DATA SETS

    # load the cleaned data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

    # CLEAN UP AND PREPROCESS DATA

    # OPTIMIZE AND TRAIN THE CLASSIFIER

    # split the data into the train and test sets
    rnd_forest_train_set_df = combined_df.loc[combined_df["data_set"] == TRAIN_SET].copy()
    rnd_forest_test_set_df = combined_df.loc[(combined_df["data_set"] == TEST_SET) &
                                             (combined_df["submit"] == 1)].copy()

    # drop all remaining unneeded columns
    # dropping day_of_year because i don't think it will lead to good generalization
    rnd_forest_train_set_df = rnd_forest_train_set_df.drop([
                                "count", "data_set", "submit", "bad_day", "casual", "registered"], axis=1)
    # drop times without weather, etc, information due to time shifting
    # also drops hours with missing casual, registered values (since
    # no diff is availble there)
    rnd_forest_train_set_df = rnd_forest_train_set_df.dropna()

    rnd_forest_test_set_df = rnd_forest_test_set_df.drop([
                                "count", "data_set", "submit", "bad_day", "casual", "registered"], axis=1)
    # test set has no log_registered_gp_diff or log_casual_gp_diff values
    rnd_forest_test_set_df = rnd_forest_test_set_df.drop(["log_registered_gp_diff", "log_casual_gp_diff"], axis=1)
    # rnd_forest_test_set_df = rnd_forest_test_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples] or [n_samples, n_outputs]
    train_X = rnd_forest_train_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                            "log_registered_gp_diff", "log_casual_gp_diff",
                                            "log_count", "log_registered", "log_casual"], axis=1).values
    test_X = rnd_forest_test_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                          "log_count", "log_registered", "log_casual"], axis=1).values
    features_lst = list(rnd_forest_test_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                          "log_count", "log_registered", "log_casual"], axis=1).columns)
    train_y = rnd_forest_train_set_df[["log_registered_gp_diff", "log_casual_gp_diff"]].values

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram

    if do_optimize:
        search_space = {"max_features": [0.5],
                        "max_depth": [15],
                        "max_leaf_nodes": [15],
                        "n_estimators": [50]}

        # cv_gs = cv.ShuffleSplit(len(rnd_forest_y), n_iter=5, test_size=0.35)
        # 4 folds without shuffling gives better cross validation scores
        cv_gs = cv.KFold(len(train_y), n_folds=4, shuffle=False)

        gp_grid_search = sklearn.grid_search.GridSearchCV(ens.RandomForestRegressor(
                            ),
                            search_space, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=2, pre_dispatch="2*n_jobs", verbose=3)

        print("running grid search")
        gp_grid_search.fit(train_X, train_y)

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
        n_estimators=500, max_features=None, max_depth=25, max_leaf_nodes=None, n_jobs=3)

    print("training using {0:} samples with {1:} features each using classifier:".format(*train_X.shape))
    print(rnd_forest_regress)
    rnd_forest_regress.fit(train_X, train_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the random forest regression to predict ratio for the training set
    # def train_rnd_forest_predict(row):
    #     return rnd_forest_regress.predict(row.loc[x_columns_list].values)[0]

    # predict the log count diff using random forest
    print("predicting the log differences, log counts, and counts on the training set with random forest")
    rnd_forest_train_set_df["log_registered_rf_diff"], rnd_forest_train_set_df["log_casual_rf_diff"] = \
        rnd_forest_regress.predict(train_X).T

    rnd_forest_train_set_df["log_registered_rf_pred"] = rnd_forest_train_set_df["log_registered_rf_diff"] + \
                                              rnd_forest_train_set_df["log_registered_gp_pred"]
    rnd_forest_train_set_df["log_casual_rf_pred"] = rnd_forest_train_set_df["log_casual_rf_diff"] + \
                                              rnd_forest_train_set_df["log_casual_gp_pred"]

    rnd_forest_train_set_df["registered_rf_pred"] = \
        np.exp(rnd_forest_train_set_df["log_registered_rf_pred"]) - 1
    rnd_forest_train_set_df["casual_rf_pred"] = \
        np.exp(rnd_forest_train_set_df["log_casual_rf_pred"]) - 1

    rnd_forest_train_set_df["count_rf_pred"] = rnd_forest_train_set_df["registered_rf_pred"] +\
                                               rnd_forest_train_set_df["casual_rf_pred"]

    rnd_forest_train_set_df["log_count_rf_pred"] = np.log(rnd_forest_train_set_df["count_rf_pred"]+1)

    score = np.sqrt(((rnd_forest_train_set_df["log_count"] -
                      rnd_forest_train_set_df["log_count_rf_pred"])**2).mean())
    print("training data score (RMSE): {:0.5f}".format(score))

    feature_import_lst = list(zip(features_lst, rnd_forest_regress.feature_importances_))
    feature_import_lst.sort(key=lambda x: x[1], reverse=True)
    # print("len features_lst ", len(features_lst))
    # print("len importances ", len(rnd_forest_regress.feature_importances_))
    print("feature importance")
    for feature, value in feature_import_lst:
        print("{}: {:.2f}".format(feature, 100*value))

    # rnd_forest_train_set_df["count"] = np.exp(rnd_forest_train_set_df["log_count"]) - 1
    # plt.figure(fig_num)
    # plt.title("log count vs predicted log count for training data")
    # rnd_forest_train_set_df["count"].plot(style=".-k", label="count")
    # rnd_forest_train_set_df["pred_count"].plot(style="x-r", label="prediction")
    # plt.legend()
    # fig_num += 1

    plt.figure(fig_num)
    plt.title("log count - predicted log count for training data")
    (rnd_forest_train_set_df["log_count"] - rnd_forest_train_set_df["log_count_rf_pred"]).plot(style=".-k")
    fig_num += 1

    # PREDICT THE TEST SET

    # predict the log count diff using random forest, calculate count
    print("predicting the log differences, log counts, and counts on the test set with random forest")
    rnd_forest_test_set_df["log_registered_rf_diff"], rnd_forest_test_set_df["log_casual_rf_diff"] = \
        rnd_forest_regress.predict(test_X).T

    rnd_forest_test_set_df["log_registered_rf_pred"] = rnd_forest_test_set_df["log_registered_rf_diff"] + \
                                              rnd_forest_test_set_df["log_registered_gp_pred"]
    rnd_forest_test_set_df["log_casual_rf_pred"] = rnd_forest_test_set_df["log_casual_rf_diff"] + \
                                              rnd_forest_test_set_df["log_casual_gp_pred"]

    rnd_forest_test_set_df["registered_rf_pred"] = \
        np.exp(rnd_forest_test_set_df["log_registered_rf_pred"]) - 1
    rnd_forest_test_set_df["casual_rf_pred"] = \
        np.exp(rnd_forest_test_set_df["log_casual_rf_pred"]) - 1

    rnd_forest_test_set_df["count_rf_pred"] = rnd_forest_test_set_df["registered_rf_pred"] +\
                                               rnd_forest_test_set_df["casual_rf_pred"]
    rnd_forest_test_set_df["count_rf_pred"] = rnd_forest_test_set_df["count_rf_pred"].apply(np.round).apply(int)

    # save the predicted log_registered_rf_diff, log_casual_rf_diff
    rnd_forest_pred_df = pd.concat([rnd_forest_train_set_df[["log_registered_rf_pred", "log_casual_rf_pred"]],
                                    rnd_forest_test_set_df[["log_registered_rf_pred", "log_casual_rf_pred"]]])
    rnd_forest_pred_df.sort_index(inplace=True)

    if not os.path.exists("random_forest_pred.csv") or overwrite:
        rnd_forest_pred_df.to_csv("random_forest_pred.csv")

    # save data and predicted trend to a new csv
    if not os.path.exists("rnd_forest_gp_submit.csv") or overwrite:
        rnd_forest_test_set_df.to_csv("rnd_forest_gp_submit.csv", columns=["count_rf_pred"], header=["count"])

    # plt.show()

