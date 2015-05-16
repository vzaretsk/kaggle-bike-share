# kaggle-bike-share

my attempt at the Kaggle bike sharing demand competition
http://www.kaggle.com/c/bike-sharing-demand

description of files and order of execution to generate results

1. clean_combine - cleans the test and train data sets and combines them into a single csv

2. add_features - adds additional features used by the regression algorithms

3. daily_trend_gp_predict - uses a Gaussian process to fit the daily sums of the registered and casual users, then uses this to normalize the data, calculate typical workday and weekday hourly trends for the two categories, and combine it all together to make hourly predictions for the entire 2 year time span, Kaggle submission score 0.52064

4. the regression classifiers could be run in any order

4a. random_forest_predict - uses a random forest and features consisting of weather, etc. to predict the difference between the log of the Guassian process trend prediction and the log of the actual counts, Kaggle submission score 0.41817

utility - file with function and class definitions