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
    do_optimize = True

    # parameters for the hourly gradient boosted trees regression
    registered_gbt_params = {"loss": "ls", "learning_rate": 0.005, "max_depth": 10,
                             "max_features": None, "max_leaf_nodes": 100,
                             "subsample": 0.5, "n_estimators": 1000}

    casual_gbt_params = registered_gbt_params
    # casual_gbt_params = {"loss": "ls", "learning_rate": 0.01, "max_depth": 10,
    #                      "max_features": None, "max_leaf_nodes": 100,
    #                      "subsample": 0.7, "n_estimators": 500}

    # parameters for the gradient boosted trees hourly fit cross validation
    search_space_regist = {"loss": ["ls"], "learning_rate": [0.005],
                    "max_depth": [10], "max_features": [None], "max_leaf_nodes": [100],
                    "min_samples_split": [2], "min_samples_leaf": [1],
                    "subsample": [0.5], "alpha": [0.9], "n_estimators": [1000]}

    search_space_casual = search_space_regist
    # search_space_casual = {"loss": ["ls"], "learning_rate": [0.01],
    #                 "max_depth": [10], "max_features": [None], "max_leaf_nodes": [100],
    #                 "min_samples_split": [2], "min_samples_leaf": [1],
    #                 "subsample": [0.7], "alpha": [0.9], "n_estimators": [500]}

    # if true, will overwrite existing files
    overwrite = True

    TRAIN_SET, TEST_SET = 0, 1
    fig_num = 1

    # LOAD SAVED RESULTS AND DATA SETS

    # load the cleaned data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

    # load the predictions of the rf, gbt, and bagged ada classifiers
    gbt_pred_df = pd.read_csv("gb_trees_hourly_pred.csv", parse_dates=["datetime"], index_col="datetime")
    rf_pred_df = pd.read_csv("rf_hourly_pred.csv", parse_dates=["datetime"], index_col="datetime")
    ada_bag_pred_df = pd.read_csv("ada_bag_hourly_pred.csv", parse_dates=["datetime"], index_col="datetime")

    # CLEAN UP AND PREPROCESS DATA

    # OPTIMIZE AND TRAIN THE CLASSIFIER

    # combine everything into a single data frame, the axis=1 option was needed for correct behavior
    combined_df = pd.concat([combined_df, gbt_pred_df, rf_pred_df, ada_bag_pred_df], axis=1)

    # split the data into the train and test sets
    train_set_df = combined_df.loc[combined_df["data_set"] == TRAIN_SET].copy()
    test_set_df = combined_df.loc[(combined_df["data_set"] == TEST_SET) &
                                  (combined_df["submit"] == 1)].copy()

    # drop all remaining unneeded columns
    # i suspect day_of_year may cause overfitting, need to test both
    # also dropping all daily predict columns, will try to directly predict the hourly
    drop_columns = ["count", "data_set", "submit", "bad_day", "casual", "registered", "day_of_year",
                    "log_registered_gp_pred", "log_casual_gp_pred", "log_registered_gp_diff", "log_casual_gp_diff"]

    train_set_df = train_set_df.drop(drop_columns, axis=1)
    # drop times without weather, etc, information due to time shifting
    # also drops hours with missing casual, registered values (since
    # no diff is availble there)
    train_set_df = train_set_df.dropna()

    test_set_df = test_set_df.drop(drop_columns, axis=1)
    # test set has no log_registered_gp_diff or log_casual_gp_diff values
    # test_set_df = test_set_df.drop(["log_registered_gp_diff", "log_casual_gp_diff"], axis=1)
    # test_set_df = test_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples] or [n_samples, n_outputs]

    train_regist_X = train_set_df.drop(["log_count", "log_registered", "log_casual", "log_casual_gb_pred",
                                        "log_casual_rf_pred", "log_casual_ada_pred"], axis=1).values
    test_regist_X = test_set_df.drop(["log_count", "log_registered", "log_casual", "log_casual_gb_pred",
                                      "log_casual_rf_pred", "log_casual_ada_pred"], axis=1).values

    train_casual_X = train_set_df.drop(["log_count", "log_registered", "log_casual", "log_registered_gb_pred",
                                        "log_registered_rf_pred", "log_registered_ada_pred"], axis=1).values
    test_casual_X = test_set_df.drop(["log_count", "log_registered", "log_casual", "log_registered_gb_pred",
                                      "log_registered_rf_pred", "log_registered_ada_pred"], axis=1).values

    features_regist_lst = list(train_set_df.drop(["log_count", "log_registered", "log_casual", "log_casual_gb_pred",
                                                  "log_casual_rf_pred", "log_casual_ada_pred"], axis=1).columns)
    features_casual_lst = list(train_set_df.drop(["log_count", "log_registered", "log_casual", "log_registered_gb_pred",
                                                  "log_registered_rf_pred", "log_registered_ada_pred"], axis=1).columns)

    train_regist_y = train_set_df["log_registered"].values.reshape(-1)
    train_casual_y = train_set_df["log_casual"].values.reshape(-1)

    # USE GRID SEARCH AND CROSS VALIDATION TO FIND OPTIMAL CLASSIFIER HYPERPARAMETERS

    # try to keep n_estimators at 500 or less for performance reasons, over 2000 may exceed 8 GB ram

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

        regist_grid_search = sklearn.grid_search.GridSearchCV(ens.GradientBoostingRegressor(
                            ),
                            search_space_regist, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=4, pre_dispatch="2*n_jobs", verbose=3)

        casual_grid_search = sklearn.grid_search.GridSearchCV(ens.GradientBoostingRegressor(
                            ),
                            search_space_casual, cv=cv_gs, scoring="mean_squared_error",
                            refit=False, n_jobs=4, pre_dispatch="2*n_jobs", verbose=3)

        print("running registered grid search")
        regist_grid_search.fit(train_regist_X, train_regist_y)

        print("running casual grid search")
        casual_grid_search.fit(train_casual_X, train_casual_y)

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
    gb_trees_regist_regr = ens.GradientBoostingRegressor(**registered_gbt_params)
    gb_trees_casual_regr = ens.GradientBoostingRegressor(**casual_gbt_params)

    print("training using {0:} samples with {1:} features each".format(*train_regist_X.shape))
    print("training registered gb trees using classifier:")
    print(gb_trees_regist_regr)
    gb_trees_regist_regr.fit(train_regist_X, train_regist_y)
    print("training casual gb trees using classifier:")
    print(gb_trees_casual_regr)
    gb_trees_casual_regr.fit(train_casual_X, train_casual_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the random forest regression to predict ratio for the training set
    # def train_rnd_forest_predict(row):
    #     return rnd_forest_regress.predict(row.loc[x_columns_list].values)[0]

    # predict the log count diff using gb trees
    print("predicting the log counts and counts on the training set with gb trees")
    train_set_df["log_registered_gb_pred"] = gb_trees_regist_regr.predict(train_regist_X)
    train_set_df["log_casual_gb_pred"] = gb_trees_casual_regr.predict(train_casual_X)

    train_set_df["registered_gb_pred"] = np.exp(train_set_df["log_registered_gb_pred"]) - 1
    train_set_df["casual_gb_pred"] = np.exp(train_set_df["log_casual_gb_pred"]) - 1

    train_set_df["count_gb_pred"] = train_set_df["registered_gb_pred"] + train_set_df["casual_gb_pred"]

    train_set_df["log_count_gb_pred"] = np.log(train_set_df["count_gb_pred"]+1)

    score = np.sqrt(((train_set_df["log_count"] -
                      train_set_df["log_count_gb_pred"])**2).mean())
    print("training data score (RMSE): {:0.5f}".format(score))

    feature_import_regist_lst = list(zip(features_regist_lst, gb_trees_regist_regr.feature_importances_))
    feature_import_casual_lst = list(zip(features_casual_lst, gb_trees_casual_regr.feature_importances_))
    feature_import_regist_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import_casual_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import = zip(feature_import_regist_lst, feature_import_casual_lst)
    # print("len features_lst ", len(features_regist_lst))
    # print("len importances ", len(gb_trees_regist_regr.feature_importances_))
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

    # predict the log count diff using gb trees, calculate count
    print("predicting the log counts and counts on the test set with gb trees")
    test_set_df["log_registered_gb_pred"] = gb_trees_regist_regr.predict(test_regist_X)
    test_set_df["log_casual_gb_pred"] = gb_trees_casual_regr.predict(test_casual_X)

    test_set_df["registered_gb_pred"] = np.exp(test_set_df["log_registered_gb_pred"]) - 1
    test_set_df["casual_gb_pred"] = np.exp(test_set_df["log_casual_gb_pred"]) - 1

    test_set_df["count_gb_pred"] = test_set_df["registered_gb_pred"] + test_set_df["casual_gb_pred"]
    test_set_df["count_gb_pred"] = test_set_df["count_gb_pred"].apply(np.round).apply(int)

    # save data and predicted trend to a new csv
    if not os.path.exists("stack_gbt_submit.csv") or overwrite:
        test_set_df.to_csv("stack_gbt_submit.csv", columns=["count_gb_pred"], header=["count"])

    # plt.show()
