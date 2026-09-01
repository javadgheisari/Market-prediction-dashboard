import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.callbacks import EarlyStopping
from pathlib import Path

# Load and preprocess data
address = str(Path.cwd()) + '/code/data/forex/EUR-USD.csv'
df = pd.read_csv(address)
coin_name = address.split("/")[-1].split(".")[0]  # Extract 'EUR-USD'

# Validate required columns
required_columns = ['UTC', 'Close']
if not all(col in df.columns for col in required_columns):
    raise ValueError("CSV missing required columns: UTC, Close")

# Sort by UTC and reset index
df = df.sort_values('UTC').reset_index(drop=True)
df['Close'] = df['Close'].astype(float)

# Parameters
window = 5  # Lookback window for sequences
train_ratio = 0.8  # 80% for training
predict_days = 24  # 2 days (24 * 5-minute intervals)

# Split data sequentially
train_size = int(len(df) * train_ratio)
train = df['Close'].values[:train_size].reshape(-1, 1)
test = df['Close'].values[train_size:].reshape(-1, 1)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)

# Prepare training sequences
X_train = []
y_train = []
for i in range(window, len(train_scaled)):
    X_train.append(train_scaled[i-window:i, 0])
    y_train.append(train_scaled[i, 0])
X_train = np.array(X_train).reshape(-1, window, 1)
y_train = np.array(y_train)

# Build GRU model
model = Sequential([
    GRU(50, return_sequences=True, input_shape=(window, 1)),
    Dropout(0.2),
    GRU(50, return_sequences=True),
    Dropout(0.2),
    GRU(50, return_sequences=True),
    Dropout(0.2),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=15, batch_size=128, callbacks=[early_stopping], verbose=1)

# Prepare test data
inputs = df['Close'].values[len(df) - len(test) - window:].reshape(-1, 1)
inputs_scaled = scaler.transform(inputs)
X_test = []
for i in range(window, len(inputs_scaled)):
    X_test.append(inputs_scaled[i-window:i, 0])
X_test = np.array(X_test).reshape(-1, window, 1)

# Predict on test set
predict = model.predict(X_test, verbose=0)
predict = scaler.inverse_transform(predict)

# Align test set with predictions
test = test[:len(predict)]  # Ensure test matches predict length
real_prices = test.flatten()  # Real prices for plotting

# Calculate metrics
diff = predict - test
mse = np.mean(diff**2)
mae = np.mean(np.abs(diff))
rmse = np.sqrt(mse)
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

# Future predictions (24 steps = 2 days)
prediction_full = []
current_input = df['Close'].values[-window:].reshape(-1, 1)
current_input_scaled = scaler.transform(current_input)

for _ in range(predict_days):
    current_input_reshaped = current_input_scaled[-window:].reshape(1, window, 1)
    pred = model.predict(current_input_reshaped, verbose=0)
    prediction_full.append(scaler.inverse_transform(pred)[0, 0])
    current_input_scaled = np.vstack((current_input_scaled, pred))

# Combine test and future predictions
prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1, 1))).flatten()

# Extend dates for future predictions
df_date = df[['UTC']].copy()
for _ in range(predict_days):
    last_date = pd.to_datetime(df_date['UTC'].iloc[-1]) + pd.DateOffset(minutes=5)
    df_date = pd.concat([df_date, pd.DataFrame([last_date.strftime("%Y-%m-%d %H:%M")], columns=['UTC'])])
df_date = df_date.reset_index(drop=True)

# Align plotting data
plot_dates = df_date['UTC'].values[train_size:train_size + len(real_prices) + predict_days]
real_prices_extended = np.concatenate([real_prices, [np.nan] * predict_days])  # Pad real prices with NaN for future period
predicted_prices = prediction_full_new[:len(plot_dates)]  # Ensure predicted prices match plot_dates

# Verify lengths
print(f"Length of plot_dates: {len(plot_dates)}")
print(f"Length of real_prices_extended: {len(real_prices_extended)}")
print(f"Length of predicted_prices: {len(predicted_prices)}")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(plot_dates, real_prices_extended, color='red', label=f'Real {coin_name} Price')
plt.plot(plot_dates, predicted_prices, color='blue', label=f'Predicted {coin_name} Price')
plt.title(f'{coin_name} Price Prediction (5-Minute Timeframe)')
plt.xlabel('UTC Time')
plt.ylabel('Price ($)')
plt.xticks(plot_dates[::12], rotation=45)  # Show every 12th timestamp (1 hour)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()