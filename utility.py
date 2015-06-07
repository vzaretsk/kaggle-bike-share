import pandas as pd
import sklearn.tree as tree
import sklearn.ensemble as ens

# defines utility functions and classes


def get_day(day_start: pd.datetime) -> pd.tseries.index.DatetimeIndex:
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="H")


# i need a quick hack to make grid search scorer keep multiple numbers but act as one
# make a custom class that inherits float and keeps additional values as a private variables
# reference
# http://stackoverflow.com/questions/2673651/inheritance-from-str-or-int
# i ended up now using this so it's not thoroughly tested
class MultiFloat(float):
    def __new__(cls, combined=0.0, registered=0.0, casual=0.0):
        obj = float.__new__(cls, combined)
        obj.registered = float(registered)
        obj.casual = float(casual)
        return obj
