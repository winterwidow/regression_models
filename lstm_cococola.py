import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("coco_cola.csv")  
print(df.columns)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Selecting features and target
features = ['open', 'high', 'low', 'close', 'volume']
target = 'adj_close'

# Scale data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[features + [target]])

# Build sequences
def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # all features except target
        y.append(data[i+seq_length, -1])     # target (adj_close)
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(scaled_features, seq_length)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(X_train, y_train, epochs=10, batch_size=1, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test)

# Inverse transform just the target (adj_close)
adj_close_scaler = MinMaxScaler()
adj_close_scaler.min_, adj_close_scaler.scale_ = scaler.min_[-1], scaler.scale_[-1]
y_test_inv = adj_close_scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = adj_close_scaler.inverse_transform(y_pred)

# Metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
print(f"\nMSE: {mse:.6f}, MAE: {mae:.6f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Actual adj_close')
plt.plot(y_pred_inv, label='Predicted adj_close', linestyle='--')
plt.title('LSTM Prediction of Adjusted Closing Price')
plt.xlabel('Samples')
plt.ylabel('Price')
plt.legend()
plt.show()
