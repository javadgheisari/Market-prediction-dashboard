import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
df = pd.read_csv('live-data/crypto/BTC.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date in ascending order
df = df.sort_values('Date')

# Extract the 'Close' column and convert it to float
prices = df['Close'].values.astype(float)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

# Define window size and split data into train/test sets
window_size = 60
train_size = int(0.8 * len(scaled_prices))
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

# Create input sequences and corresponding labels
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, window_size)
X_test, y_test = create_sequences(test_data, window_size)

# Build the GRU model
model = Sequential()
model.add(GRU(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(GRU(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions to original scale
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

# Calculate evaluation metrics
train_mse = mean_squared_error(y_train[0], train_predictions[:, 0])
train_mae = mean_absolute_error(y_train[0], train_predictions[:, 0])
test_mse = mean_squared_error(y_test[0], test_predictions[:, 0])
test_mae = mean_absolute_error(y_test[0], test_predictions[:, 0])

print(f'Train MSE: {train_mse:.4f}')
print(f'Train MAE: {train_mae:.4f}')
print(f'Test MSE: {test_mse:.4f}')
print(f'Test MAE: {test_mae:.4f}')

# Plot the predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][:train_size], prices[:train_size], label='Training Data')
plt.plot(df['Date'][train_size+window_size:], prices[train_size+window_size:], label='Actual Price')
plt.plot(df['Date'][train_size+window_size:], test_predictions[:, 0], label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Bitcoin Price Prediction')
plt.legend()
plt.show()