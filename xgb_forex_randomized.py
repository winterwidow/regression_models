#randomized searh cv:
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("EURUSD_2024-05-01.csv")  
df.columns = df.columns.str.strip().str.lower()  # lowercase column names just in case

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Smooth the close price using a simple moving average
df['close_smooth'] = df['close'].rolling(window=3).mean()

# Compute % daily change and reduce noise
df['pct_change'] = df['close'].pct_change()
df = df[df['pct_change'].abs() < 0.05]

# Create lag features based on smoothed close
def create_lag_features(data, lag=10):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)
    return data

df = create_lag_features(df)
df.dropna(inplace=True)

# Features & target
features = [f'lag_{i}' for i in range(1, 4)]
target = 'close'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Set up parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.03, 0.05, 0.056, 0.06, 0.07, 0.1],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Base model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=15,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
print("Best parameters found:", random_search.best_params_)

# Predict
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Directional accuracy
actual_diff = y_test.values - y_test.shift(1).values
pred_diff = y_pred - y_test.shift(1).values
direction_correct = np.sign(actual_diff[1:]) == np.sign(pred_diff[1:])
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
results.to_csv("predicted_vs_actual_forex.csv", index=False)
print("Results saved to 'predicted_vs_actual_forex.csv'")
'''
#gridsearch cv:
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("EURUSD_2024-05-01.csv")  
df.columns = df.columns.str.strip().str.lower()

# Convert date
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Smooth the close price using a simple moving average
df['close_smooth'] = df['close'].rolling(window=3).mean()

# Compute % daily change and reduce noise
df['pct_change'] = df['close'].pct_change()
df = df[df['pct_change'].abs() < 0.05]

# Create lag features based on smoothed close
def create_lag_features(data, lag=7):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)
    return data

df = create_lag_features(df)
df.dropna(inplace=True)

# Features & target
features = [f'lag_{i}' for i in range(1, 4)]  # You can experiment with more lags
target = 'close'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Grid search parameter grid
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'learning_rate': [0.12, 0.15,0.18,0.2,0.25,0.3],
    'max_depth': [3, 4, 5],
    'subsample': [0.5,0.6],
    'colsample_bytree': [0.8, 1.0]
}

# Base model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    verbose=1,
    n_jobs=-1
)

# Fit
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

# Predict
y_pred = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Directional accuracy
actual_diff = y_test.values - y_test.shift(1).values
pred_diff = y_pred - y_test.shift(1).values
direction_correct = np.sign(actual_diff[1:]) == np.sign(pred_diff[1:])
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
results.to_csv("predicted_vs_actual_forex.csv", index=False)
print("Results saved to 'predicted_vs_actual_forex.csv'")
'''

#xgboost's built in CV with early stopping

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("EURUSD_2024-05-01.csv")
df.columns = df.columns.str.strip().str.lower()  # lowercase just in case

# Date formatting
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Smoothing + noise reduction
df['close_smooth'] = df['close'].rolling(window=3).mean()
df['pct_change'] = df['close'].pct_change()
df = df[df['pct_change'].abs() < 0.05]

# Create lag features from smoothed close
def create_lag_features(data, lag=7):
    for i in range(1, lag + 1):
        data[f'lag_{i}'] = data['close_smooth'].shift(i)
    return data

df = create_lag_features(df)
df.dropna(inplace=True)

# Features & Target
features = [f'lag_{i}' for i in range(1, 4)]
target = 'close'

X = df[features]
y = df[target]

# Split (no shuffle, maintain time order)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# XGBoost DMatrix format for internal CV
dtrain = xgb.DMatrix(X_train, label=y_train)

# Parameters
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 4,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'seed': 42
}

# Run XGBoost's built-in CV to find best number of boosting rounds
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=500,
    nfold=3,
    early_stopping_rounds=20,
    metrics='rmse',
    verbose_eval=True
)

# Best number of rounds
best_n_estimators = len(cv_results)
print(f"Best number of boosting rounds: {best_n_estimators}")

# Final model
model = xgb.XGBRegressor(
    **params,
    n_estimators=best_n_estimators
)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.6f}, MAE: {mae:.6f}")

# Directional Accuracy
actual_diff = y_test.values - y_test.shift(1).values
pred_diff = y_pred - y_test.shift(1).values
direction_correct = np.sign(actual_diff[1:]) == np.sign(pred_diff[1:])
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

# Save to CSV
results = pd.DataFrame({
    'Date': df.iloc[-len(y_test):]['date'].values,
    'Actual': y_test.values,
    'Predicted': y_pred
})
results.to_csv("predicted_vs_actual_forex.csv", index=False)
print("Results saved to 'predicted_vs_actual_forex.csv'")
