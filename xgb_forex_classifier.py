import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("EURUSD_2024-05-01.csv")  # replace with your filename
df.columns = df.columns.str.strip().str.lower()  # lowercase column names just in case

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Smooth the close price using a simple moving average
df['close_smooth'] = df['close'].rolling(window=3).mean()
# Compute % daily change
df['pct_change'] = df['close'].pct_change()

# Drop days with absurdly high % change (example: > 5%)
df = df[df['pct_change'].abs() < 0.05]


# Create lag features based on 'close' price
def create_lag_features(data, lag=10):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)

    return data

df = create_lag_features(df)
df.dropna(inplace=True)  # drop rows with NaN from lag

# Features & target
features = [f'lag_{i}' for i in range(1, 4)]
#target = 'close'
# Direction label (1 = up, 0 = down)
df['direction'] = (df['close_smooth'].shift(-1) > df['close_smooth']).astype(int)


X = df[features]
y = df['direction']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
'''
# Model
model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, objective='reg:squarederror')
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
'''
# Classifier model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy (Direction): {accuracy * 100:.2f}%")

# plot
plt.figure(figsize=(10, 4))
plt.plot(y_test.reset_index(drop=True), label='Actual Direction', marker='o')
plt.plot(y_pred, label='Predicted Direction', linestyle='--', marker='x')
plt.title('Direction Prediction (1=Up, 0=Down)')
plt.xlabel('Samples')
plt.ylabel('Direction')
plt.legend()
plt.tight_layout()
plt.show()

# Save results
results = pd.DataFrame({
    'Date': df.iloc[-len(y_test):]['date'].values,
    'Actual Direction': y_test.values,
    'Predicted Direction': y_pred
})
results.to_csv("predicted_vs_actual_directions.csv", index=False)
print("Results saved to 'predicted_vs_actual_directions.csv'")
