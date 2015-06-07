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


# create 1 to 24 hour shifted versions of various base features and then fit a random forest to it to determine feature
# importance, results for the 5 features are the the bottom of the email, plots are saved

# needed for python multiprocessing
if __name__ == '__main__':

    # SETTINGS AND GLOBAL CONSTANTS

    # if true, will overwrite existing files
    overwrite = True

    # one of atemp, humidity, temp, weather, windspeed
    base_feature = "windspeed"

    # parameters for the daily sum regression rf
    hourly_regist_rf_params = {"max_features": None, "max_depth": 20,
                        "max_leaf_nodes": 2**10, "min_samples_split": 8,
                        "min_samples_leaf": 4, "n_estimators": 500}

    hourly_casual_rf_params = hourly_regist_rf_params
    # hourly_casual_rf_params = {"max_features": None, "max_depth": 20,
    #                     "max_leaf_nodes": 2**10, "min_samples_split": 8,
    #                     "min_samples_leaf": 4, "n_estimators": 500}

    TRAIN_SET, TEST_SET = 0, 1
    fig_num = 1

    # LOAD SAVED RESULTS AND DATA SETS

    # load the cleaned data, parse "datetime" column as a date and use it as an index
    # if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
    combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

    # CLEAN UP AND PREPROCESS DATA

    # keep the base feature column and the target columns
    combined_df = combined_df[[base_feature, "log_registered_gp_diff", "log_casual_gp_diff", "data_set"]]

    # create 1 to 24 hour time shifted versions of the base column and then fit using all of them,
    # i will interpret feature importance as indicator of feature prediction quality
    for shift in range(1, 25):
        shifted_feature = "{:}_{:02d}h_ago".format(base_feature, shift)
        combined_df[shifted_feature] = combined_df[base_feature].shift(periods=shift, freq="H")

    # split out the training data
    train_set_df = combined_df.loc[combined_df["data_set"] == TRAIN_SET].copy()
    train_set_df.drop(["data_set"], axis=1, inplace=True)

    # drop times without weather, etc, information due to time shifting
    # also drops hours with missing casual, registered values (since
    # no diff is availble there)
    train_set_df = train_set_df.dropna()

    # will use all the remaining columns to predict log count difference
    # X : {array-like, sparse matrix} of shape = [n_samples, n_features]
    # y : array-like of shape = [n_samples] or [n_samples, n_outputs]
    train_X = train_set_df.drop(["log_registered_gp_diff", "log_casual_gp_diff"], axis=1).values
    features_lst = list(train_set_df.drop(["log_registered_gp_diff", "log_casual_gp_diff"], axis=1).columns)
    train_regist_y = train_set_df["log_registered_gp_diff"].values.reshape(-1)
    train_casual_y = train_set_df["log_casual_gp_diff"].values.reshape(-1)

    # TRAIN THE OPTIMUM HOURLY CLASSIFIER

    # create the classifier
    hourly_regist_regress = ens.RandomForestRegressor(n_jobs=2, **hourly_regist_rf_params)
    hourly_casual_regress = ens.RandomForestRegressor(n_jobs=2, **hourly_casual_rf_params)

    print("training using {0:} samples with {1:} features each".format(*train_X.shape))
    print("training registered using classifier:")
    print(hourly_regist_regress)
    hourly_regist_regress.fit(train_X, train_regist_y)
    print("training casual using classifier:")
    print(hourly_casual_regress)
    hourly_casual_regress.fit(train_X, train_casual_y)

    # not needed, can just predict on the entire data array at once, much faster than using apply
    # in general it seems apply is slow, a vectorized function done to the .values is much faster
    # # use the random forest regression to predict ratio for the training set
    # def train_rnd_forest_predict(row):
    #     return rnd_forest_regress.predict(row.loc[x_columns_list].values)[0]

    feature_import_regist_lst = list(zip(features_lst, hourly_regist_regress.feature_importances_))
    feature_import_casual_lst = list(zip(features_lst, hourly_casual_regress.feature_importances_))
    feature_import_regist_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import_casual_lst.sort(key=lambda x: x[1], reverse=True)
    feature_import = zip(feature_import_regist_lst, feature_import_casual_lst)
    # print("len features_lst ", len(features_lst))
    # print("len importances ", len(gb_trees_regress.feature_importances_))
    print("feature importance")
    print("registered\tcasual")
    for (reg_feat, reg_score), (cas_feat, cas_score) in feature_import:
        print("{}: {:.2f}\t{}: {:.2f}".format(reg_feat, 100*reg_score, cas_feat, 100*cas_score))

    cur_fig = plt.figure(fig_num)
    plt.title("shifted " + base_feature + " feature importance")
    plt.plot(range(0, 25), hourly_regist_regress.feature_importances_, ".-", label="registered")
    plt.plot(range(0, 25), hourly_casual_regress.feature_importances_, ".-", label="casual")
    plt.xticks(range(0, 25), features_lst, rotation="vertical")
    plt.legend(loc="best")
    plt.subplots_adjust(bottom=0.3)
    fig_num += 1
    cur_fig.savefig(base_feature + "_feature_importance.png", dpi=cur_fig.dpi, facecolor=cur_fig.get_facecolor())
    plt.show()

# feature importance
# registered	casual
#
# atemp: 8.63	atemp: 10.09
# atemp_24h_ago: 5.75	atemp_24h_ago: 5.68
# atemp_22h_ago: 4.27	atemp_07h_ago: 4.54
#
# humidity: 9.67	humidity: 8.81
# humidity_24h_ago: 6.23	humidity_24h_ago: 5.44
# humidity_23h_ago: 4.05	humidity_11h_ago: 4.61
#
# temp: 8.96	temp: 9.96
# temp_24h_ago: 5.75	temp_24h_ago: 5.37
# temp_21h_ago: 4.59	temp_07h_ago: 4.25
#
# weather_01h_ago: 21.38	weather_01h_ago: 16.50
# weather: 7.29	weather: 4.36
# weather_24h_ago: 3.54	weather_24h_ago: 4.28
#
# windspeed_02h_ago: 4.41	windspeed_11h_ago: 4.34
# windspeed_10h_ago: 4.38	windspeed_24h_ago: 4.34
# windspeed_12h_ago: 4.32	windspeed_10h_ago: 4.34
