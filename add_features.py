import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

# SETTINGS AND GLOBAL CONSTANTS

# if true, will overwrite existing files
overwrite = True

TRAIN_SET, TEST_SET = 0, 1

# LOAD THE DATA

# load the data, parse "datetime" column as a date and use it as an index
# if index_col="datetime" is used, train_set_df["datetime"] doesn't work any more
combined_df = pd.read_csv("combined_data.csv", parse_dates=["datetime"], index_col="datetime")

# CREATE FEATURES USED BY THE CLASSIFIERS

# log versions of the various counts
# log is more resistant to outliers, better reflects ratios (rather than absolute differences),
# and is the evaluation metric for the challenge
combined_df["log_count"] = np.log(combined_df["count"] + 1)
combined_df["log_registered"] = np.log(combined_df["registered"] + 1)
combined_df["log_casual"] = np.log(combined_df["casual"] + 1)

# create year, month, dayofweek, dayofyear, and hour columns
year_f = lambda x: x.index.year - 2011
month_f = lambda x: x.index.month
dayofweek_f = lambda x: x.index.dayofweek
dayofyear_f = lambda x: x.index.dayofyear
hour_f = lambda x: x.index.hour
# could use any column besides "count", just need it to be a single column so result is a series, not a data frame
combined_df["year"] = combined_df.apply(year_f)["count"]
combined_df["month"] = combined_df.apply(month_f)["count"]
combined_df["day_of_week"] = combined_df.apply(dayofweek_f)["count"]
combined_df["day_of_year"] = combined_df.apply(dayofyear_f)["count"]
combined_df["hour"] = combined_df.apply(hour_f)["count"]

# create shifted versions of humidity, weather, atemp, temp, windspeed as potentially useful features
# optimal shift times found by running feature_selection
combined_df["atemp_07h_ago"] = combined_df["atemp"].shift(periods=7, freq="H")
combined_df["atemp_24h_ago"] = combined_df["atemp"].shift(periods=24, freq="H")

combined_df["humidity_11h_ago"] = combined_df["humidity"].shift(periods=11, freq="H")
combined_df["humidity_24h_ago"] = combined_df["humidity"].shift(periods=24, freq="H")

combined_df["temp_24h_ago"] = combined_df["temp"].shift(periods=24, freq="H")

combined_df["weather_01h_ago"] = combined_df["weather"].shift(periods=1, freq="H")
combined_df["weather_24h_ago"] = combined_df["weather"].shift(periods=24, freq="H")

combined_df["windspeed_11h_ago"] = combined_df["windspeed"].shift(periods=11, freq="H")
combined_df["windspeed_24h_ago"] = combined_df["windspeed"].shift(periods=24, freq="H")

# # create 24h shifted versions of humidity, weather, atemp as potentially useful features
# combined_df["humidity_24h_ago"] = combined_df["humidity"].shift(periods=24, freq="H")
# combined_df["weather_24h_ago"] = combined_df["weather"].shift(periods=24, freq="H")
# combined_df["atemp_24h_ago"] = combined_df["atemp"].shift(periods=24, freq="H")
# combined_df["windspeed_24h_ago"] = combined_df["windspeed"].shift(periods=24, freq="H")

# column "weekend" indicates if a day is a weekend and if so, how long the weekend is
combined_df["weekend"] = 1 - combined_df["workingday"]

weekend_values = combined_df["weekend"].values
begin_index = end_index = cur_index = 0
cur_val = prev_val = 0
weekend_iter = iter(weekend_values)

try:
    while True:
        cur_val = next(weekend_iter)
        if cur_val == 1 and prev_val == 0:
            # being weekend
            begin_index = cur_index
        elif cur_val == 0 and prev_val == 1:
            # end weekend
            end_index = cur_index
            weekend_values[begin_index:end_index] = [end_index-begin_index]*(end_index-begin_index)
            begin_index = cur_index
        else:
            # no change
            pass
        cur_index += 1
        prev_val = cur_val
except StopIteration:
    end_index = cur_index
    if prev_val == 1:
        weekend_values[begin_index:end_index] = [end_index-begin_index]*(end_index-begin_index)

combined_df["weekend"] = weekend_values // 24

# weekends = combined_df["weekend"].resample("D")
# print("single day weekends due to holidays falling in the middle of the week")
# print(weekends[weekends == 1].index)
# combined_df["weekend"].plot()
# plt.show()

# SAVE THE MODIFIED DATA

# save the modified data, overwrite the original file
if not os.path.exists("combined_data.csv") or overwrite:
    combined_df.to_csv("combined_data.csv")
