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
from utility import *

# not able to find parameters for adaboost that match performance of rf or gbt
# cv may not have been working

# needed for python multiprocessing
if __name__ == '__main__':

    # SETTINGS AND GLOBAL CONSTANTS

    # if false, will skip grid search, values used in classifier are still optimimal since i chose them
    # after running this and also based on submission scores
    do_optimize = True

    # parameters for the hourly adaboost regression
    registered_ada_params = {"loss": "linear", "learning_rate": 0.01, "n_estimators": 500}
    registered_tree_params = {"max_depth": None, "max_features": None, "max_leaf_nodes": 2**7,
                              "min_samples_split": 8, "min_samples_leaf": 4}

    casual_ada_params = registered_ada_params
    casual_tree_params = registered_tree_params
    # casual_ada_params = {"loss": "linear", "learning_rate": 1.0, "n_estimators": 50}
    # casual_tree_params = {"max_depth": None, "max_features": None, "max_leaf_nodes": 2**10,
    #                       "min_samples_split": 8, "min_samples_leaf": 4}

    # prefixes needed for grid search to work with a classifier with embedded classifiers
    ada_p = ""
    tree_p = "base_estimator__"
    # loss may be "linear", "square", "exponential"
    # parameters for the gradient boosted trees hourly fit cross validation
    search_space_regist = {ada_p+"loss": ["linear"], ada_p+"learning_rate": [0.01], ada_p+"n_estimators": [500],
                           tree_p+"max_depth": [None], tree_p+"max_features": [None], tree_p+"max_leaf_nodes": [2**7],
                           tree_p+"min_samples_split": [8], tree_p+"min_samples_leaf": [4]}

    # search_space_casual = search_space_regist
    search_space_casual = {ada_p+"loss": ["linear"], ada_p+"learning_rate": [0.01], ada_p+"n_estimators": [500],
                           tree_p+"max_depth": [None], tree_p+"max_features": [None], tree_p+"max_leaf_nodes": [2**7],
                           tree_p+"min_samples_split": [8], tree_p+"min_samples_leaf": [4]}

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
    train_set_df = combined_df.loc[combined_df["data_set"] == TRAIN_SET].copy()
    test_set_df = combined_df.loc[(combined_df["data_set"] == TEST_SET) &
                                             (combined_df["submit"] == 1)].copy()

    # drop all remaining unneeded columns
    # i suspect day_of_year may cause overfitting, need to test both
    drop_columns = ["count", "data_set", "submit", "bad_day", "casual", "registered", "day_of_year"]

    train_set_df = train_set_df.drop(drop_columns, axis=1)
    # drop times without weather, etc, information due to time shifting
    # also drops hours with missing casual, registered values (since
    # no diff is availble there)
    train_set_df = train_set_df.dropna()

    test_set_df = test_set_df.drop(drop_columns, axis=1)
    # test set has no log_registered_gp_diff or log_casual_gp_diff values
    test_set_df = test_set_df.drop(["log_registered_gp_diff", "log_casual_gp_diff"], axis=1)
    # test_set_df = test_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples] or [n_samples, n_outputs]
    train_X = train_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                            "log_registered_gp_diff", "log_casual_gp_diff",
                                            "log_count", "log_registered", "log_casual"], axis=1).values
    test_X = test_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                          "log_count", "log_registered", "log_casual"], axis=1).values
    features_lst = list(test_set_df.drop(["log_registered_gp_pred", "log_casual_gp_pred",
                                          "log_count", "log_registered", "log_casual"], axis=1).columns)
    train_regist_y = train_set_df["log_registered_gp_diff"].values.reshape(-1)
    train_casual_y = train_set_df["log_casual_gp_diff"].values.reshape(-1)

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_ada_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram

    if do_optimize:
        # cv_gs = cv.ShuffleSplit(len(train_y), n_iter=4, test_size=0.35)

        # 4 folds without shuffling gives better cross validation scores
        # cv_gs = cv.KFold(len(train_y), n_folds=4, shuffle=False)

        # custom split based on predicting the first/last 5 days from the remaining of each month
        split_arr = np.array([0, 0, 0, 0, 0,
                              -1, -1, -1, -1, -1, -1, -1, -1, -1,
                              1, 1, 1, 1, 1])
        train_split_f = lambda x: split_arr[x.index.day - 1]

        train_split = train_set_df.apply(train_split_f)["atemp"].values
        cv_gs = cv.PredefinedSplit(train_split)

        regist_grid_search = sklearn.grid_search.GridSearchCV(ens.AdaBoostRegressor(
                                base_estimator=tree.DecisionTreeRegressor()),
                            search_space_regist, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=1, pre_dispatch="2*n_jobs", verbose=3)

        casual_grid_search = sklearn.grid_search.GridSearchCV(ens.AdaBoostRegressor(
                                base_estimator=tree.DecisionTreeRegressor()),
                            search_space_casual, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=1, pre_dispatch="2*n_jobs", verbose=3)

        print("running registered grid search")
        regist_grid_search.fit(train_X, train_regist_y)

        print("running casual grid search")
        casual_grid_search.fit(train_X, train_casual_y)

        gs_lst = [(search_space_regist, regist_grid_search), (search_space_casual, casual_grid_search)]

        for search_space, grid_search in gs_lst:
            with open("gs_score_log.txt", "a") as gs_score_log:
                gs_score_log.write("search space:\n")
                gs_score_log.write(str(search_space)+"\n")
                gs_score_log.write("cross validation:\n")
                gs_score_log.write(str(cv_gs)+"\n")
                gs_score_log.write("grid search:\n")
                gs_score_log.write(str(grid_search)+"\n")

                for params, mean_score, scores in grid_search.grid_scores_:
                    line = "mean: {:.4f}, stdev: {:.4f} MSE for {:}".format(-mean_score, scores.std(), params)
                    print(line)
                    gs_score_log.write(line+"\n")

                gs_score_log.write("\n")

        print("best score for registered (RMSE): {:.5f}".format(np.sqrt(-regist_grid_search.best_score_)))
        print("best score for casual (RMSE): {:.5f}".format(np.sqrt(-casual_grid_search.best_score_)))

    # TRAIN THE OPTIMUM CLASSIFIER

    # create the classifiers
    ada_regist_regr  = ens.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(**registered_tree_params),
                                             **registered_ada_params)
    ada_casual_regr  = ens.AdaBoostRegressor(base_estimator=tree.DecisionTreeRegressor(**casual_tree_params),
                                             **casual_ada_params)

    print("training using {0:} samples with {1:} features each".format(*train_X.shape))
    print("training registered adaboost using classifier:")
    print(ada_regist_regr)
    ada_regist_regr.fit(train_X, train_regist_y)
    print("training casual adaboost using classifier:")
    print(ada_casual_regr)
    ada_casual_regr.fit(train_X, train_casual_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the random forest regression to predict ratio for the training set
    # def train_rnd_forest_predict(row):
    #     return rnd_forest_regress.predict(row.loc[x_columns_list].values)[0]

    # predict the log count diff using adaboost
    print("predicting the log differences, log counts, and counts on the training set with adaboost")
    train_set_df["log_registered_gb_diff"] = ada_regist_regr.predict(train_X)
    train_set_df["log_casual_gb_diff"] = ada_casual_regr.predict(train_X)

    train_set_df["log_registered_gb_pred"] = train_set_df["log_registered_gb_diff"] + \
                                              train_set_df["log_registered_gp_pred"]
    train_set_df["log_casual_gb_pred"] = train_set_df["log_casual_gb_diff"] + \
                                              train_set_df["log_casual_gp_pred"]

    train_set_df["registered_gb_pred"] = np.exp(train_set_df["log_registered_gb_pred"]) - 1
    train_set_df["casual_gb_pred"] = np.exp(train_set_df["log_casual_gb_pred"]) - 1

    train_set_df["count_gb_pred"] = train_set_df["registered_gb_pred"] + train_set_df["casual_gb_pred"]

    train_set_df["log_count_gb_pred"] = np.log(train_set_df["count_gb_pred"]+1)

    score = np.sqrt(((train_set_df["log_count"] -
                      train_set_df["log_count_gb_pred"])**2).mean())
    print("training data score (RMSE): {:0.5f}".format(score))

    feature_import_regist_lst = list(zip(features_lst, ada_regist_regr.feature_importances_))
    feature_import_casual_lst = list(zip(features_lst, ada_casual_regr.feature_importances_))
    feature_import_regist_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import_casual_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import = zip(feature_import_regist_lst, feature_import_casual_lst)
    # print("len features_lst ", len(features_lst))
    # print("len importances ", len(ada_regress.feature_importances_))
    print("feature importance")
    print("registered\tcasual")
    for (reg_feat, reg_score), (cas_feat, cas_score) in feature_import:
        print("{}: {:.2f}\t{}: {:.2f}".format(reg_feat, 100*reg_score, cas_feat, 100*cas_score))

    # train_set_df["count"] = np.exp(train_set_df["log_count"]) - 1
    # plt.figure(fig_num)
    # plt.title("log count vs predicted log count for training data")
    # train_set_df["count"].plot(style=".-k", label="count")
    # train_set_df["pred_count"].plot(style="x-r", label="prediction")
    # plt.legend()
    # fig_num += 1

    plt.figure(fig_num)
    plt.title("log count - predicted log count for training data")
    (train_set_df["log_count"] - train_set_df["log_count_gb_pred"]).plot(style=".-k")
    fig_num += 1

    # PREDICT THE TEST SET

    # predict the log count diff using adaboost, calculate count
    print("predicting the log differences, log counts, and counts on the test set with adaboost")
    test_set_df["log_registered_gb_diff"] = ada_regist_regr.predict(test_X)
    test_set_df["log_casual_gb_diff"] = ada_casual_regr.predict(test_X)

    test_set_df["log_registered_gb_pred"] = test_set_df["log_registered_gb_diff"] + \
                                              test_set_df["log_registered_gp_pred"]
    test_set_df["log_casual_gb_pred"] = test_set_df["log_casual_gb_diff"] + \
                                              test_set_df["log_casual_gp_pred"]

    test_set_df["registered_gb_pred"] = np.exp(test_set_df["log_registered_gb_pred"]) - 1
    test_set_df["casual_gb_pred"] = np.exp(test_set_df["log_casual_gb_pred"]) - 1

    test_set_df["count_gb_pred"] = test_set_df["registered_gb_pred"] + test_set_df["casual_gb_pred"]
    test_set_df["count_gb_pred"] = test_set_df["count_gb_pred"].apply(np.round).apply(int)

    # save the predicted log_registered_gb_diff, log_casual_gb_diff
    pred_df = pd.concat([train_set_df[["log_registered_gb_pred", "log_casual_gb_pred"]],
                                    test_set_df[["log_registered_gb_pred", "log_casual_gb_pred"]]])
    pred_df.sort_index(inplace=True)

    if not os.path.exists("ada_gp_pred.csv") or overwrite:
        pred_df.to_csv("ada_gp_pred.csv")

    # save data and predicted trend to a new csv
    if not os.path.exists("ada_gp_submit.csv") or overwrite:
        test_set_df.to_csv("ada_gp_submit.csv", columns=["count_gb_pred"], header=["count"])

    # plt.show()
