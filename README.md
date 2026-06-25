# 📈 AI Investment Advisor

### AI-Powered Stock Market Prediction using Deep Learning, News Sentiment Analysis & Portfolio Recommendation

---

## 🚀 Overview

**AI Investment Advisor** is a machine learning-based financial intelligence platform that predicts stock prices using an **LSTM (Long Short-Term Memory)** deep learning model. The system combines historical market data, technical indicators, news sentiment analysis, and portfolio recommendations in a single interactive dashboard built with **Streamlit**.

This project demonstrates how machine learning and financial analytics can be integrated into a practical decision-support application for investors and learners.

---

# 🌍 Real-World Problem

Investors often rely on multiple platforms to analyze the market:

* Historical stock charts
* Technical indicators
* Financial news
* Portfolio planning

Switching between different tools makes investment decisions slower and more difficult.

---

# 💡 Proposed Solution

AI Investment Advisor combines these components into a single application.

The system:

* Downloads historical stock market data
* Calculates technical indicators
* Predicts future stock prices using an LSTM model
* Analyzes recent financial news sentiment
* Generates Buy/Sell signals
* Suggests portfolio allocations based on risk level
* Stores prediction history

---

# ✨ Features

* 📈 LSTM Deep Learning Stock Prediction
* 📰 Real-Time News Sentiment Analysis
* 📊 Interactive Candlestick Charts
* 📉 Trend Analysis using SMA & EMA
* 📌 RSI & MACD Technical Indicators
* 💰 Portfolio Recommendation Engine
* 📜 Prediction History Tracking
* 📊 Model Performance Comparison
* 📈 Confidence Score Display
* ⚡ Interactive Streamlit Dashboard

---

# 🛠 Tech Stack

## Programming Language

* Python

## Machine Learning

* TensorFlow
* Keras
* Scikit-learn

## Data Processing

* Pandas
* NumPy

## Financial Data

* Yahoo Finance (yfinance)

## Technical Analysis

* ta Library

## News Analysis

* NewsAPI
* TextBlob

## Visualization

* Plotly
* Plotly Express

## Frontend

* Streamlit

## Model Storage

* Joblib
* HDF5 (.h5)

---

# 🧠 Machine Learning Workflow

```
Yahoo Finance API
        │
        ▼
Historical Stock Data
        │
        ▼
Feature Engineering
        │
        ▼
Technical Indicators
(SMA, EMA, RSI, MACD)
        │
        ▼
Data Scaling
        │
        ▼
LSTM Deep Learning Model
        │
        ▼
Price Prediction
        │
        ▼
Inverse Scaling
        │
        ▼
Buy / Sell Signal
```

---

# 🏗 System Architecture

```
                   User

                     │

                     ▼

        Streamlit Dashboard

                     │

      ┌──────────────┼──────────────┐

      ▼              ▼              ▼

 Yahoo Finance     News API     Portfolio Engine

      │              │

      ▼              ▼

Technical       TextBlob

Indicators   Sentiment Analysis

      │

      ▼

 Feature Engineering

      │

      ▼

Data Scaling

      │

      ▼

 LSTM Model

      │

      ▼

Prediction Engine

      │

      ▼

 Interactive Dashboard
```

---

# 📊 Technical Indicators Used

The application calculates several technical indicators to enrich the input features:

* Simple Moving Average (SMA20)
* Simple Moving Average (SMA50)
* Exponential Moving Average (EMA20)
* Relative Strength Index (RSI)
* Moving Average Convergence Divergence (MACD)

---

# 📰 News Sentiment Analysis

The application fetches recent news headlines using **NewsAPI**.

Each headline is analyzed using **TextBlob** to compute sentiment polarity.

Sentiment is classified as:

* 🟢 Bullish
* 🟡 Neutral
* 🔴 Bearish

This provides additional market context alongside technical predictions.

---

# 💰 Portfolio Recommendation Engine

The application recommends investment allocations based on the selected risk profile.

### Low Risk

* Apple
* Microsoft
* Google
* NVIDIA

### Medium Risk

* Apple
* Microsoft
* NVIDIA
* Google
* Tesla

### High Risk

* NVIDIA
* Tesla
* Meta
* Apple

The dashboard also estimates:

* Expected Return
* Expected Profit
* Future Portfolio Value

---

# 📈 Dashboard Features

The Streamlit dashboard includes:

* Current Stock Price
* Predicted Price
* Buy/Sell Signal
* Confidence Score
* Price Change
* Percentage Change
* RSI
* MACD
* News Sentiment
* Candlestick Chart
* Trend Analysis
* Portfolio Recommendation
* Prediction History
* Model Comparison

---

# 🤖 Machine Learning Model

## Model

Long Short-Term Memory (LSTM)

## Input Features

* Close Price
* Volume
* SMA20
* SMA50
* EMA20
* RSI
* MACD

## Output

Predicted Next Stock Price

---

# 📊 Model Performance

| Model         |      MAE |      RMSE |      MAPE |
| ------------- | -------: | --------: | --------: |
| Random Forest |    15.84 |     22.22 |     5.64% |
| **LSTM**      | **9.94** | **11.67** | **3.72%** |

The LSTM model achieved lower prediction error than the Random Forest model on the evaluated metrics.

---

# 📂 Project Structure

```
AI-Investment-Advisor/
│
├── app.py
├── train_lstm.py
├── lstm_stock_prediction.py
├── random_forest_stock.py
├── news_sentiment.py
├── requirements.txt
├── README.md
│
├── data/
│   └── prediction_history.csv
│
├── models/
│   ├── lstm_stock_model.h5
│   └── scaler.pkl
│
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/spotracer23786/AI-Stock-Prediction-by-Husmatth-AI-Investment-Advisor.git
```

Navigate to the project

```bash
cd AI-Stock-Prediction-by-Husmatth-AI-Investment-Advisor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---
# 🔮 Future Improvements

* Transformer-based Stock Forecasting
* Multi-stock Prediction
* Real-Time Streaming Data
* Explainable AI (XAI)
* Risk Analytics Dashboard
* Reinforcement Learning Portfolio Optimization
* Cryptocurrency Forecasting
* Cloud Deployment
* User Authentication
* Mobile-Friendly Dashboard

---

# 🎯 Learning Outcomes

This project demonstrates practical experience with:

* Deep Learning
* Time Series Forecasting
* Financial Data Analysis
* Feature Engineering
* Technical Indicators
* Sentiment Analysis
* Interactive Dashboard Development
* Portfolio Analytics
* Python Development
* Machine Learning Deployment

---

# 👨‍💻 Author

**Husmatth**

GitHub: https://github.com/spotracer23786

---

# 📄 License

This project is intended for educational and portfolio purposes.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
