import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("EURUSD_2024-05-01.csv")  
df.columns = df.columns.str.strip().str.lower()  # lowercase column names just in case

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Smooth the close price using a simple moving average
df['close_smooth'] = df['close'].rolling(window=3).mean()
# Compute % daily change
df['pct_change'] = df['close'].pct_change()

# Drop days with absurdly high % change (example: > 5%) - reduce noise
df = df[df['pct_change'].abs() < 0.05]


# Create lag features based on 'close' price
def create_lag_features(data, lag=12):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)
    
    return data

df = create_lag_features(df)
df.dropna(inplace=True)  # drop rows with NaN from lag

# Features & target
features = [f'lag_{i}' for i in range(1, 4)]
target = 'close'

X = df[features]
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Model
model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.24, max_depth = 4, subsample = 0.9, colsample_bytree = 0.8, objective='reg:squarederror')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Compare direction of actual and predicted price movements
actual_diff = y_test.values - y_test.shift(1).values
pred_diff = y_pred - y_test.shift(1).values

# Boolean arrays where direction matches
direction_correct = np.sign(actual_diff[1:]) == np.sign(pred_diff[1:])  # skip first, no previous value
direction_accuracy = direction_correct.mean() * 100

print(f"Directional Accuracy: {direction_accuracy:.2f}%")


'''# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('Close Price Prediction')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()'''

# Save results
results = pd.DataFrame({
    'Date': df.iloc[-len(y_test):]['date'].values,
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv("predicted_vs_actual_forex.csv", index=False)
print("Results saved to 'predicted_vs_actual_forex.csv'")

