import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

print("Script has started.")

try:
    # Load and check the data
    df = pd.read_csv("NVIDIA_STOCK.csv")

    print("Columns:", df.columns)
    print(df.head())

    df.columns = df.columns.str.strip().str.lower()  # lowercase and strip whitespace
    print("Columns after cleaning:", df.columns.tolist())

    # Convert date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['date'], inplace=True)

    else:
        print("⚠️ No 'date' column found. Skipping datetime conversion.")

    # Rename for consistency if needed
    if 'adj close' in df.columns:
        df.rename(columns={'adj close': 'adj_close'}, inplace=True)

    # Create lag features
    def create_lag_features(data, lag=12):
        for i in range(1, lag + 1):
            data[f'lag_{i}'] = data['adj_close'].shift(i)
        return data

    df = create_lag_features(df)

    # Drop rows with NaNs from lag
    df.dropna(inplace=True)
    print(f"Rows after cleaning: {len(df)}")

    # Features and target
    features = [f'lag_{i}' for i in range(1, 4)] + ['volume']
    target = 'adj_close'

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.01, objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Check if predictions are made
    print(f"Predictions: {y_pred[:5]}")  # Display first few predictions

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price', linestyle='--')
    plt.title('XGBoost Prediction of NVIDIA Stock')
    plt.xlabel('Samples')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save to CSV
    results = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })
    results.to_csv("predicted_vs_actual_nvidia.csv", index=False)
    print("Results saved to 'predicted_vs_actual_nvidia.csv'")

except Exception as e:
    print(f"An error occurred: {e}")
