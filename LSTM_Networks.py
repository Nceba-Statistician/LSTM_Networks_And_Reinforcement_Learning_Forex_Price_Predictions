from Forex_data import Historical_Forex_data
import numpy
import pandas
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot

Close_records = Historical_Forex_data["Close"].values

scaler = StandardScaler() # The scaler expects a 2D array because it generally processes data where each row represents a sample, and each column represents a feature.
scaled_Closed_records = scaler.fit_transform(Close_records.reshape(-1,1))

# We use the window of past 24 time steps to predict the next value
sequence_length = 24
X_train, y_train = [], []

for i in range(sequence_length, len(scaled_Closed_records)):
    X_train.append(scaled_Closed_records[i - sequence_length:i, 0])
    y_train.append(scaled_Closed_records[i, 0])
    
X_train, y_train = numpy.array(X_train), numpy.array(y_train)

# print(scaled_Closed_records)
# print(scaled_Closed_records.shape) -- (10319, 1)
# print(X_train)
# print(X_train.shape) -- (10295, 24)
# print(y_train)
# print(y_train.shape) -- (10295,) -- as generating one target value for each sequence | y_train is the target for a particular sequence in X_train

# Reshape the data to 3D (sample, time steps, features) for LSTM input

X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# print(X_train) 
# print(X_train.shape) -- (10295, 24, 1)
# -- We have a time series dataset (X_train) where each sample (total 10295) is a sequence of 24 time steps, and each time step has 1 feature
# -- meaning Sequence 1 = 24 time steps = sample 1, Sequence 2 = 24 time steps = sample 2, ... , Sequence 10295 = 24 time steps = sample 10295

# Splitting data into train and test sets (80% train, 20% test)

train_size = int(len(scaled_Closed_records) * 0.8)
train_data = scaled_Closed_records[:train_size] # 8255
test_data = scaled_Closed_records[train_size - sequence_length:] # 8231

# Building and Training the LSTM Model
import tensorflow
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout, Input

model = Sequential()

model.add(Input(shape=(X_train.shape[1], 1))) # Input layer

model.add(LSTM(units=60, return_sequences=True)) # First LSTM layer with Dropout
model.add(Dropout(0.2))

model.add(LSTM(units=60, return_sequences=False)) # Second LSTM layer with Dropout
model.add(Dropout(0.2)) # 20% of the neurons are dropped out at each training step, helping the model generalize better

model.add(Dense(units=30, activation='relu')) # dense layer (fully connected layer) # Rectified Linear Unit introduces non-linearity into the network, allowing the model to learn more complex patterns.
model.add(Dense(units=1)) # Output layer

# LSTM Layer are great for learning and predicting sequential or time-dependent data. In this case we applied with 50 units/neurons,
# which should allow the model to learn patterns from the input sequence.

# To reduce/prevent overfitting in neural networks we randomly dropping 20% of the neurons (units) in the previous layer during the training step
# which forces the model to learn more robust patterns and prevents it from relying too heavily on any one set of neurons.
# Overfitting is common in deep learning models, especially when dealing with time series data that might have noise or variability.

# Activation Function: If no activation function is specified, it defaults to linear.
# This means the output of each neuron is just the linear combination of the inputs, without any non-linearity (such as ReLU or sigmoid). 

model.compile(optimizer='adam', loss='mean_squared_error') # Compiling the model # Adaptive Moment Estimation # MSE calculates the average squared difference between the predicted values (y_pred) and the true values (y_true)
model.fit(X_train, y_train, batch_size=64, epochs=10)  # Training the model # model will iterate over the dataset 10 times # epochs how many times the model will go through the entire training dataset
# batch_size number of samples the model will use in one forward and backward pass (one step of gradient descent) before updating its weights.

# Test Data Prediction

X_test, y_test = [], []

for j in range(sequence_length, len(test_data)):
    X_test.append(test_data[j-sequence_length:j, 0])
    y_test.append(test_data[j,0])
    
X_test, y_test = numpy.array(X_test), numpy.array(y_test)

X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict the test data

predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform([y_test])

# Create Datetime index for plotting
test_datetime = Historical_Forex_data.index[-len(X_test):]  # Adjust this based on the actual range of your test data

# Plot the results

pyplot.figure(figsize=(20, 5))
pyplot.plot(test_datetime, real_prices[0], color='black', label='Real Prices')
pyplot.plot(test_datetime, predicted_prices, color='blue', label='Predicted Prices')
pyplot.title('Price Prediction')
pyplot.xlabel('Datetime')
pyplot.ylabel('Price')
pyplot.legend()
pyplot.savefig("price_prediction_plot.png", format="png")
pyplot.show()

# Check lengths for debugging

# print("Length of plot_dates:", len(test_datetime))
# print("Length of real_prices[0]:", len(real_prices[0]))
# print("Length of predicted_prices:", len(predicted_prices))