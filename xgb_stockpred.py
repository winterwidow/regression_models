import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load and prepare the dataset
df = pd.read_csv("stock.csv") 
df.columns = ['date', 'price']  # Ensure column names are correct

# Create lag features
def create_lag_features(data, lag=3):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['price'].shift(i)
    data.dropna(inplace=True)
    return data

df = create_lag_features(df)

# Set up features and target
features = [f'lag_{i}' for i in range(1, 4)]
X = df[features]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Initialize and train model
model = XGBRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}")

# Plotting actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred, label="Predicted Price", linestyle='--')
plt.title("XGBoost Prediction - Price")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.legend()
plt.show()
