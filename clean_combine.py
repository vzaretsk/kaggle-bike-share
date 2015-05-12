import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

# SETTINGS AND GLOBAL CONSTANTS

# if true, will overwrite existing files
overwrite = True

# if true, will mark certain days as workdays that are not marked as such in the data,
# for example tax day, will also fix holidays, such as Thanksgiving
fix_workdays = True

TRAIN_SET, TEST_SET = 0, 1

# LOAD THE DATA

# load the data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
train_set_df = pd.read_csv("train.csv", parse_dates=["datetime"], index_col="datetime")
test_set_df = pd.read_csv("test.csv", parse_dates=["datetime"], index_col="datetime")

# CLEAN UP AND PREPROCESS DATA

# create convenience columns indicating which data set the data came from
# this may also be possible with hierarchical indexing
# using int instead of a string label as that avoids the .values becoming object type
train_set_df["data_set"] = TRAIN_SET
test_set_df["data_set"] = TEST_SET

# combine the train and test sets
# sorting afterward is needed otherwise shift operations don't behave properly
# the is because the index is no longer time sorted but two groups in the order of the concat
combined_df = pd.concat([train_set_df, test_set_df])
combined_df.sort_index(inplace=True)

# resample at 1 hour intervals, will later impute some of the missing values
combined_df = combined_df.resample("H")

# find all the missing values and create a series of them
missing_sr = combined_df["weather"][combined_df["weather"].isnull()].fillna(0)
missing_gp = missing_sr.groupby([missing_sr.index.date])
print("count of missing values by date")
print(missing_gp.count())

# except for a couple outliers, missing values are single or a few hours, linear interpolation should work well
# will impute missing values for weather, temp, atemp, humidity, windspeed, workingday, and holiday in case i
# want to create time shifted features

# interpolate weather, atemp, humidity
combined_df["weather"] = combined_df["weather"].interpolate(method='time').apply(np.round)
combined_df["temp"] = combined_df["temp"].interpolate(method='time')
combined_df["atemp"] = combined_df["atemp"].interpolate(method='time')
combined_df["humidity"] = combined_df["humidity"].interpolate(method='time').apply(np.round)
combined_df["windspeed"] = combined_df["windspeed"].interpolate(method='time')

# backfill missing workingday and holiday information
combined_df["workingday"] = combined_df["workingday"].fillna(method="bfill")
combined_df["holiday"] = combined_df["holiday"].fillna(method="bfill")

# mark several rarely celebrated holidays, such as tax day, as work days
# vice versa, mark thanksgiving as a holiday
if fix_workdays:
    # tax day
    combined_df["workingday"][pd.datetime(2011, 4, 15, 0):pd.datetime(2011, 4, 15, 23)] = 1
    combined_df["workingday"][pd.datetime(2012, 4, 16, 0):pd.datetime(2012, 4, 16, 23)] = 1
    combined_df["workingday"][pd.datetime(2011, 11, 25, 0):pd.datetime(2011, 11, 25, 23)] = 0
    combined_df["workingday"][pd.datetime(2012, 11, 23, 0):pd.datetime(2012, 11, 23, 23)] = 0
    # thanksgiving
    combined_df["holiday"][pd.datetime(2011, 4, 15, 0):pd.datetime(2011, 4, 15, 23)] = 0
    combined_df["holiday"][pd.datetime(2012, 4, 16, 0):pd.datetime(2012, 4, 16, 23)] = 0
    combined_df["holiday"][pd.datetime(2011, 11, 25, 0):pd.datetime(2011, 11, 25, 23)] = 1
    combined_df["holiday"][pd.datetime(2012, 11, 23, 0):pd.datetime(2012, 11, 23, 23)] = 1

# drop season, it's useless, can create better features from month, day of week, etc
combined_df.drop(["season"], axis=1, inplace=True)

# SAVE THE CLEANED DATA

if not os.path.exists("combined_data.csv") or overwrite:
    combined_df.to_csv("combined_data.csv")
