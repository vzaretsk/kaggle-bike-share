import pandas as pd

# defines utility functions and classes


def get_day(day_start: pd.datetime) -> pd.tseries.index.DatetimeIndex:
    day_end = day_start + pd.offsets.DateOffset(hours=23)
    return pd.date_range(day_start, day_end, freq="H")
