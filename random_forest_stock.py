import yfinance as yf
import pandas as pd
import numpy as np
import ta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ==========================
# DOWNLOAD DATA
# ==========================

ticker = "AAPL"

df = yf.Ticker(ticker).history(period="5y")

# ==========================
# TECHNICAL INDICATORS
# ==========================

df["SMA20"] = ta.trend.sma_indicator(
    df["Close"],
    window=20
)

df["SMA50"] = ta.trend.sma_indicator(
    df["Close"],
    window=50
)

df["EMA20"] = ta.trend.ema_indicator(
    df["Close"],
    window=20
)

df["RSI"] = ta.momentum.rsi(
    df["Close"],
    window=14
)

df["MACD"] = ta.trend.macd(
    df["Close"]
)

df.dropna(inplace=True)

# ==========================
# FEATURES
# ==========================

X = df[
    [
        "Open",
        "High",
        "Low",
        "Volume",
        "SMA20",
        "SMA50",
        "EMA20",
        "RSI",
        "MACD"
    ]
]

y = df["Close"]

# ==========================
# TRAIN TEST SPLIT
# ==========================

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================
# MODEL
# ==========================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)

predictions = model.predict(X_test)

# ==========================
# METRICS
# ==========================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (y_test - predictions)
        / y_test
    )
) * 100

print("\n====================")
print("RANDOM FOREST")
print("====================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape, 2), "%")