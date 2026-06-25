import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# DOWNLOAD STOCK DATA
# ==========================================

print("Downloading Apple Stock Data...")

ticker = yf.Ticker("AAPL")
df = ticker.history(period="5y")

if df.empty:
    print("ERROR: No data downloaded.")
    exit()

print("Data Downloaded Successfully!")
print(df.head())

# ==========================================
# USE CLOSE PRICE ONLY
# ==========================================

data = df[['Close']]

# ==========================================
# SCALING
# ==========================================

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ==========================================
# CREATE SEQUENCES
# ==========================================

X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# ==========================================
# RESHAPE FOR LSTM
# ==========================================

X_train = np.reshape(
    X_train,
    (X_train.shape[0], X_train.shape[1], 1)
)

X_test = np.reshape(
    X_test,
    (X_test.shape[0], X_test.shape[1], 1)
)

# ==========================================
# BUILD MODEL
# ==========================================

model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(Dropout(0.2))

model.add(
    LSTM(
        units=50,
        return_sequences=False
    )
)

model.add(Dropout(0.2))

model.add(Dense(units=25))
model.add(Dense(units=1))

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\nTraining Model...\n")

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=10,
    verbose=1
)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test_actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        predictions
    )
)

print("\n====================")
print("MODEL PERFORMANCE")
print("====================")

print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# ==========================================
# NEXT DAY PREDICTION
# ==========================================

last_60_days = scaled_data[-60:]

X_future = np.array([last_60_days])

X_future = np.reshape(
    X_future,
    (
        X_future.shape[0],
        X_future.shape[1],
        1
    )
)

future_price = model.predict(X_future)

future_price = scaler.inverse_transform(
    future_price
)

print("\nPredicted Next Day Price:")
print(round(float(future_price[0][0]), 2))

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    y_test_actual,
    label="Actual Price"
)

plt.plot(
    predictions,
    label="Predicted Price"
)

plt.title("LSTM Apple Stock Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()

plt.show()