import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pdb

def load_data(filename):
    df = pd.read_csv(filename, header=None, names=['Price', 'Time'])
    return df

def preprocess_data(data, seq_length=10):
    #sequence length - trains the model on last 3 prices to predict the next
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Price']])
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_predictions(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_true_inv, label='Actual Price')
    plt.plot(y_pred_inv, label='Predicted Price', linestyle='--')
    plt.legend()
    plt.title('LSTM Prediction of Stock Price')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.show()

def plot_error(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    errors = y_true_inv - y_pred_inv  # Difference between actual and predicted
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, color='red')
    plt.title('Prediction Error (Actual - Predicted)')
    plt.xlabel('Samples')
    plt.ylabel('Error in Price')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.7)
    plt.show()

def main():
    df = load_data('stock.csv')  
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #y_test has the actual prices, y_pred has the preiction values
    #we compare y_test and y_pred
    
    model = build_lstm_model((X.shape[1], X.shape[2])) #(samples, time_steps, features)
    #samples- how many sequences present, time_step- how many past values to use, features- only price
    model.fit(X_train, y_train, epochs=5, batch_size=1, validation_data=(X_test, y_test))
    
    y_pred = model.predict(X_test)
    
    # Print actual vs predicted values
    print("\nSample Predictions (Actual vs Predicted):")
    for actual, predicted in zip(y_test[:10], y_pred[:10]):

        print("actual is", actual)
        #pdb.set_trace()
        actual_price = scaler.inverse_transform([actual])[0][0]


        predicted_price = scaler.inverse_transform([[predicted[0]]])[0][0]
        print(f"Actual: {actual_price:.2f}, Predicted: {predicted_price:.2f}")

    # Calculate and print error metrics
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"\nMSE: {mse:.4f}, MAE: {mae:.4f}")

    plot_predictions(y_test, y_pred, scaler)
    #plot_error(y_test, y_pred, scaler)


if __name__ == '__main__':
    main()
