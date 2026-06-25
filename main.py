import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# =====================================
# DOWNLOAD DATA
# =====================================

print("Downloading Data...")

df = yf.download(
    "AAPL",
    start="2020-01-01",
    end="2024-01-01"
)

# =====================================
# USE CLOSE PRICE
# =====================================

df["Close"] = df["Close"]

# =====================================
# FEATURE ENGINEERING
# =====================================

# Lag Features
df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)
df["Lag_7"] = df["Close"].shift(7)

# Moving Averages
df["MA_7"] = df["Close"].rolling(7).mean()
df["MA_30"] = df["Close"].rolling(30).mean()

# Daily Return
df["Return"] = df["Close"].pct_change()

# Volatility
df["Volatility"] = df["Return"].rolling(7).std()

# Momentum
df["Momentum"] = df["Close"] - df["Close"].shift(10)

# Volume Average
df["Volume_MA7"] = df["Volume"].rolling(7).mean()

# Tomorrow Price Target
df["Target"] = df["Close"].shift(-1)

# Remove Missing Values
df.dropna(inplace=True)

# =====================================
# FEATURES & TARGET
# =====================================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_7",
    "MA_7",
    "MA_30",
    "Return",
    "Volatility",
    "Momentum",
    "Volume_MA7"
]

X = df[features]
y = df["Target"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# =====================================
# MODEL
# =====================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

print("Training Model...")

model.fit(X_train, y_train)

# =====================================
# PREDICTION
# =====================================

predictions = model.predict(X_test)

# =====================================
# EVALUATION
# =====================================

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

print("\n======================")
print("MODEL PERFORMANCE")
print("======================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# =====================================
# NEXT DAY PREDICTION
# =====================================

latest_data = X.iloc[-1:]

next_day_price = model.predict(
    latest_data
)

print("\nPredicted Next Day Price:")
print(round(next_day_price[0], 2))

# =====================================
# VISUALIZATION
# =====================================

plt.figure(figsize=(14, 7))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title("Apple Stock Prediction")
plt.xlabel("Days")
plt.ylabel("Price")

plt.legend()

plt.show()