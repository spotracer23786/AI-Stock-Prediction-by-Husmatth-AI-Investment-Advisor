import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


# ==========================
# DOWNLOAD DATA
# ==========================

ticker = "AAPL"

df = yf.Ticker(ticker).history(period="5y")

# ==========================
# TECHNICAL INDICATORS
# ==========================

df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)

df["SMA50"] = ta.trend.sma_indicator(df["Close"], window=50)

df["EMA20"] = ta.trend.ema_indicator(df["Close"], window=20)

df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

df["MACD"] = ta.trend.macd(df["Close"])

df.dropna(inplace=True)

# ==========================
# FEATURES
# ==========================

features = [
    "Close",
    "Volume",
    "SMA20",
    "SMA50",
    "EMA20",
    "RSI",
    "MACD"
]

data = df[features]

# ==========================
# SCALING
# ==========================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# ==========================
# SEQUENCE CREATION
# ==========================

X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):

    X.append(
        scaled_data[i-sequence_length:i]
    )

    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)

# ==========================
# TRAIN TEST SPLIT
# ==========================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================
# MODEL
# ==========================

model = Sequential()

model.add(
    LSTM(
        units=64,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=64
    )
)

model.add(Dropout(0.2))

model.add(Dense(25))

model.add(Dense(1))

# ==========================
# COMPILE
# ==========================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# ==========================
# TRAIN
# ==========================

print("Training Started...\n")

model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    verbose=1
)

# ==========================
# PREDICT
# ==========================

predictions = model.predict(X_test)

dummy_pred = np.zeros(
    (len(predictions), len(features))
)

dummy_pred[:, 0] = predictions.flatten()

predictions_actual = scaler.inverse_transform(
    dummy_pred
)[:, 0]

dummy_actual = np.zeros(
    (len(y_test), len(features))
)

dummy_actual[:, 0] = y_test

actual_prices = scaler.inverse_transform(
    dummy_actual
)[:, 0]

# ==========================
# METRICS
# ==========================

mae = mean_absolute_error(
    actual_prices,
    predictions_actual
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions_actual
    )
)

mape = np.mean(
    np.abs(
        (actual_prices - predictions_actual)
        / actual_prices
    )
) * 100

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")

# ==========================
# SAVE MODEL
# ==========================

model.save("models/lstm_stock_model.h5")

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("\nModel Saved Successfully")