# 1. LSTM Networks for Forex/Stock Price Prediction
# pip install yfinance pandas numpy matplotlib tensorflow scikit-learn
# Data Preparation and Feature Scaling

import yfinance as yf

# Download historical Forex data (EUR/USD)
# For stock data, you can use 'AAPL' for Apple, 'GOOG' for Google, etc.
df = yf.download('EURUSD=X', start='2018-01-01', end='2023-01-01')
print(df.head())

# For stock data (e.g., AAPL)
# df = yf.download('AAPL', start='2018-01-01', end='2023-01-01')


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Use the 'Close' price for prediction
data = df[['Close']].values

# Scaling data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Creating training data (e.g., 60 past steps to predict the next step)
sequence_length = 60
x_train, y_train = [], []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to 3D (samples, time steps, features) for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Splitting data into train and test sets (80% train, 20% test)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size-sequence_length:]

# Building and Training the LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Building the LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=25))
model.add(Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, batch_size=64, epochs=10)

# Test Data Prediction
x_test, y_test = [], []
for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
    y_test.append(test_data[i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the test data
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform([y_test])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(real_prices[0], color='black', label='Real Prices')
plt.plot(predicted_prices, color='blue', label='Predicted Prices')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# Explanation:
# This LSTM model uses past 60 days’ closing prices to predict the next price.
# You can adjust the sequence_length, units, and other parameters to improve the model.
