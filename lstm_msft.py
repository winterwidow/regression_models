import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

print("Script has started.")

# Load data
df = pd.read_csv("MSFT_1986-03-13_2025-02-04.csv")
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

# Smooth closing prices
df['close_smooth'] = df['close'].rolling(window=3).mean()
df['pct_change'] = df['close'].pct_change()
df = df[df['pct_change'].abs() < 0.05]

# Create lag features using the smoothed close
def create_lag_features(data, lag=12):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)
    return data

df = create_lag_features(df)
df.dropna(inplace=True)

# Features and target
features = [f'lag_{i}' for i in range(1, 13)]
target = 'close'

# Scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
y_scaled = scaler.fit_transform(df[[target]])

# Reshape to 3D for LSTM: [samples, time_steps, features]
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, shuffle=False)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_rescaled = scaler.inverse_transform(y_test)

# Evaluate
mse = np.mean((y_pred - y_test_rescaled)**2)
mae = np.mean(np.abs(y_pred - y_test_rescaled))
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Directional accuracy
actual_diff = y_test_rescaled.flatten()[1:] - y_test_rescaled.flatten()[:-1]
pred_diff = y_pred.flatten()[1:] - y_test_rescaled.flatten()[:-1]
direction_correct = np.sign(actual_diff) == np.sign(pred_diff)
direction_accuracy = np.mean(direction_correct) * 100
print(f"Directional Accuracy: {direction_accuracy:.2f}%")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('LSTM Close Price Prediction')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()
