# regression_models
 
## **1. Forex Price Direction Predictor with XGBoost:**

A time series machine learning project that uses historical EUR/USD data to predict the direction of price movement using XGBoost regression and technical features like lagged close prices, smoothed averages, and noise filtering.

### **Project Objective**


The goal of this project is to:

1. Predict the next day's closing price of EUR/USD based on past prices.

2. Evaluate model performance not just on error metrics (MSE, MAE), but more importantly:

3. Measure directional accuracy — i.e., how often the model correctly predicts if the price will go up or down.

### **Data Preparation:**

1. Raw CSV with Date, Open, High, Low, Close is loaded.

2. Dates are parsed and formatted properly.

3. Close prices are smoothed with a 3-day moving average to reduce noise.

4. Outlier days with extreme % changes are dropped.

### **Feature Engineering:**

Lag features are created from smoothed closing prices (e.g., lag_1, lag_2, lag_3).

These lag features serve as the model input.

### **Model Training:**

The model used is XGBRegressor, tuned manually for best directional performance.

Data is split into train/test sets using chronological order (no shuffling).

### **Evaluation Metrics:**

MSE / MAE: Traditional regression errors.

Directional Accuracy: The percentage of times the model correctly predicts if the next price moves up or down.

### **Results:**

Predictions and actuals are saved to a CSV.

Optional plotting is available to visualize predictions vs actuals.


## **2. Coca-Cola Stock Price Prediction:**

This project uses LSTM (Long Short-Term Memory) and XGBoost (Extreme Gradient Boosting) to predict daily closing prices of Coca-Cola stock using historical market data.

**File descriptions:**

lstm_coca_cola.py | Predicts stock prices using an LSTM neural network

xgb_coca_cola.py | Predicts stock prices using XGBoost regression

coca_cola_stock.csv | Historical Coca-Cola stock data (Date, Open, High, Low, Close, Adj Close, Volume)


### **LSTM Model**

**Key Details:**

Uses last 3 days of prices to predict the next day’s closing price

Normalizes values using MinMaxScaler

Neural network architecture:

Two LSTM layers with dropout

One dense layer for output

Trained using mean_squared_error loss and adam optimizer


### **XGBoost Model**

**Key Details:**

Uses lag features (previous 3 days) for prediction

Trained with XGBRegressor

Simpler and faster than LSTM; often performs surprisingly well

**Output**

Both models print actual vs predicted prices for a few test samples

**Also calculate:**

1. MSE: Mean Squared Error

2. MAE: Mean Absolute Error

3. And generate a plot showing predicted vs actual prices
