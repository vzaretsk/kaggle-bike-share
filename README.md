# kaggle-bike-share

my attempt at the Kaggle bike sharing demand competition
http://www.kaggle.com/c/bike-sharing-demand

description of files and order of execution to generate results

1. clean_combine - cleans the test and train data sets and combines them into a single csv

2. add_features - adds additional features used by the regression algorithms

3. daily_trend_rf_split_predict - uses a random forest to fit the daily sums of the registered and casual users, then uses this to normalize the data, calculate typical workday and weekday hourly trends for the two categories, and combine it all together to make hourly predictions for the entire 2 year time span

4. the regression classifiers could be run in any order

4a. random_forest_hourly_predict - uses a random forest and features consisting of weather, etc. to predict the difference between the log of the rf daily trend prediction and the log of the actual counts

4b. gb_trees_hourly_predict - uses gradient boosted trees and features consisting of weather, etc. to predict the difference between the log of the rf daily trend prediction and the log of the actual counts

utility - file with function and class definitions

feature_selection - uses a random forest and a single weather, etc. time shifted feature to predict the difference between the log of the rf daily trend prediction and the log of the actual counts, output the feature importance to determine which time shifted feature should be included in the main results

ada_bag_hourly_predict, ada_hourly_predict - attempt to use adaboost or bagged adaboost to predict the hourly counts based on the rf daily trend prediction, didn't work as well as rf or gbt

daily_trend_gp_split_predict - uses a Gaussian process to fit the daily sums of the registered and casual users, then uses this to normalize the data, calculate typical workday and weekday hourly trends for the two categories, and combine it all together to make hourly predictions for the entire 2 year time span, replaced by rf trend predict

stack_gbt, stack_rf, stack_ridge - attempt to combine the gbt and rf hourly predictions using one of several classifiers, didn't work well