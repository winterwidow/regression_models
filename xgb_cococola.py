import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

print("Script has started.")

try:
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Load and preprocess the data
    df = pd.read_csv('KO_1919-09-06_2025-04-17.csv')  
    df['date'] = pd.to_datetime(df['date'])

    # Use relevant features and create lag features (i.e., past prices)
    def create_lag_features(data, lag=3):
        for i in range(1, lag + 1):
            data[f'lag_{i}'] = data['adj_close'].shift(i)
        return data

    df = create_lag_features(df)

    # Drop rows with NaN (from the shift)
    df.dropna(inplace=True)

    # Features and target
    features = ['lag_1', 'lag_2', 'lag_3', 'volume']
    target = 'adj_close'

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Model
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price', linestyle='--')
    plt.title('XGBoost Prediction of Coca-Cola Stock')
    plt.xlabel('Samples')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Create a DataFrame with actual vs predicted prices
    results = pd.DataFrame({
        'Actual Price': y_test,
        'Predicted Price': y_pred
    })

    # Save to CSV
    results.to_csv('predicted_vs_actual_prices.csv', index=False)

    # Optional: Print a message confirming the file has been saved
    print("The predicted vs actual prices have been saved to 'predicted_vs_actual_prices.csv'.")

except Exception as e:
    print(f"An error occurred: {e}")
