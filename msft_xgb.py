import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

print("Script has started.")

# Load and check the data
df = pd.read_csv("MSFT_1986-03-13_2025-02-04.csv")

print("Columns:", df.columns)
print(df.head())
df.columns = df.columns.str.strip().str.lower()  # lowercase and strip whitespace
print("Columns after cleaning:", df.columns.tolist())

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')

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


# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('Close Price Prediction')
plt.xlabel('Samples')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.show()

# Save results
results = pd.DataFrame({
    'Date': df.iloc[-len(y_test):]['date'].values,
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv("predicted_vs_actual_nvidia.csv", index=False)
print("Results saved to 'predicted_vs_actual_nvidia.csv'")

'''try:
    # Load and check the data
    df = pd.read_csv("MSFT_1986-03-13_2025-02-04.csv")

    print("Columns:", df.columns)
    print(df.head())

    # Clean up column names to ensure they're consistent
    df.columns = df.columns.str.strip().str.lower()  # lowercase and strip whitespace
    print("Columns after cleaning:", df.columns.tolist())

    # Convert date column to datetime format (DD-MM-YYYY)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    else:
        print("⚠️ No 'date' column found. Skipping datetime conversion.")

    # Rename for consistency if needed
    if 'adj close' in df.columns:
        df.rename(columns={'adj close': 'adj_close'}, inplace=True)

    # Create lag features (lagging 'adj_close' and 'volume')
    def create_lag_features(data, lag=12):
        for i in range(1, lag + 1):
            data[f'lag_{i}'] = data['adj_close'].shift(i)
        return data

    # Apply lag features
    df = create_lag_features(df)

    # Drop rows with NaNs resulting from the lag features
    df.dropna(inplace=True)
    print(f"Rows after cleaning: {len(df)}")

    # Features and target columns
    features = [f'lag_{i}' for i in range(1, 4)] + ['volume']
    target = 'adj_close'

    # Split data into features (X) and target (y)
    X = df[features]
    y = df[target]

    # Train-test split (time series split, don't shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model definition
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1,  # Adjusted learning rate
        max_depth=6,        # Added max_depth for better model complexity
        subsample=0.9,      # Reduced overfitting by using subsample
        objective='reg:squarederror'
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Check the first few predictions to see if they're reasonable
    print(f"Sample predictions on the test set: {y_pred[:5]}")

    # Evaluation metrics: Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Price', color='blue')
    plt.plot(y_pred, label='Predicted Price', linestyle='--', color='red')
    plt.title('XGBoost Prediction of NVIDIA Stock')
    plt.xlabel('Samples')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save the results to a CSV file for comparison
    results = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })
    results.to_csv("predicted_vs_actual_nvidia.csv", index=False)
    print("Results saved to 'predicted_vs_actual_nvidia.csv'")

except Exception as e:
    print(f"An error occurred: {e}")'''
