import yfinance
import pandas
import numpy
import pickle
import json

Historical_Forex_data = yfinance.download(
    "EURUSD=X", start="2022-10-01", end="2024-06-01", interval="1h"
)

Historical_Forex_data_Json = pandas.DataFrame(
    Historical_Forex_data
).to_json()


with open("EURUSD_Historical_Forex_2022_2024_Json.json", "w") as file:
    file.write(Historical_Forex_data_Json)
    
with open("EURUSD_Historical_Forex_2022_2024_Json.json", "r") as file:
    EURUSD_Historical_Forex_2022_2024_processed = json.load(file)
    
print(EURUSD_Historical_Forex_2022_2024_processed)

