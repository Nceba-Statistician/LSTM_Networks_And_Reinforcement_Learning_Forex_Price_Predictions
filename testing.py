import yfinance
import pandas
import numpy
import pickle
import json

Testing_data = pandas.DataFrame(yfinance.download(
    "EURUSD=X", start="2022-10-01", end="2024-06-01", interval="1h"
))

Testing_data.index = pandas.to_datetime(Testing_data.index).tz_localize(None).floor("s")

print(Testing_data.head(20))

