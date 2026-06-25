import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import plotly.express as px
import os
import time

from news_sentiment import get_sentiment
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ===================================
# PAGE CONFIG
# ===================================

st.set_page_config(
    page_title="AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

[data-testid="stMetric"] {
    background-color: #1E1E1E;
    border: 1px solid #2A2A2A;
    border-radius: 15px;
    padding: 15px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)
# ===================================
# LOADING ANIMATION
# ===================================

progress_bar = st.progress(0)
status = st.empty()

status.text("🤖 Loading AI Model...")
progress_bar.progress(20)
time.sleep(0.8)

model = load_model(
    "models/lstm_stock_model.h5",
    compile=False
)

scaler = joblib.load(
    "models/scaler.pkl"
)

status.text("📈 Fetching Live Stock Data...")
progress_bar.progress(45)
time.sleep(0.8)

status.text("📰 Analyzing Market News...")
progress_bar.progress(70)
time.sleep(0.8)

status.text("🧠 Running LSTM Prediction...")
progress_bar.progress(90)
time.sleep(0.8)

status.text("🚀 Preparing Dashboard...")
progress_bar.progress(100)
time.sleep(0.6)

status.success("✅ AI Investment Advisor Ready!")
# ===================================
# TITLE
# ===================================

st.markdown("""
# 📈 AI Investment Advisor

### Real-Time Financial Intelligence Platform

Powered by:
- LSTM Deep Learning
- News Sentiment Analysis
- Portfolio Analytics
- Technical Indicators (RSI, MACD, SMA, EMA)
""")

# ===================================
# STOCK SELECTION
# ===================================
st.sidebar.title("📈 AI Advisor")
st.sidebar.markdown("---")
stock = st.sidebar.selectbox(
    "Select Stock",
    [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "GOOGL",
        "AMZN",
        "META"
    ]
)

#sidebar navigation

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Portfolio",
        "Prediction History",
        "Model Comparison",
        "About"
    ]
)
# ===================================
# DOWNLOAD DATA
# ===================================

df = yf.Ticker(stock).history(
    period="5y"
)

# ===================================
# TECHNICAL INDICATORS
# ===================================

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

# ===================================
# FEATURES
# ===================================

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

scaled_data = scaler.transform(data)

# ===================================
# LAST 60 DAYS
# ===================================

last_60 = scaled_data[-60:]

X = np.array([last_60])

# ===================================
# PREDICTION
# ===================================

prediction = model.predict(
    X,
    verbose=0
)

dummy = np.zeros(
    (1, len(features))
)

dummy[:, 0] = prediction

predicted_price = scaler.inverse_transform(
    dummy
)[0][0]

current_price = df["Close"].iloc[-1]

# ===================================
# SIGNAL
# ===================================

price_change = predicted_price - current_price

change_percent = (
    price_change / current_price
) * 100

if predicted_price > current_price:
    signal = "BUY 🟢"
else:
    signal = "SELL 🔴"

# ===================================
# NEWS SENTIMENT
# ===================================

company_names = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta"
}

try:
    sentiment = get_sentiment(
        company_names[stock]
    )
except:
    sentiment = "Unavailable"

# ===================================
# CONFIDENCE
# ===================================

mape = 3.72
confidence = 100 - mape


# ===================================
# PAGE ROUTING
# ===================================

if page == "Dashboard":

    # ===================================
    # DASHBOARD
    # ===================================

    st.subheader("📊 Prediction Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}"
        )

    with col2:
        st.metric(
            "Predicted Price",
            f"${predicted_price:.2f}"
        )

    with col3:
        st.metric(
            "Signal",
            signal
        )

    with col4:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )
    # ===================================
    # EXTRA METRICS
    # ===================================

    col5, col6, col7, col8, col9 = st.columns(5)

    with col5:
        st.metric(
            "Price Change",
            f"${price_change:.2f}"
        )

    with col6:
        st.metric(
            "Change %",
            f"{change_percent:.2f}%"
        )

    with col7:
        st.metric(
            "RSI",
            round(df["RSI"].iloc[-1], 2)
        )

    with col8:
        st.metric(
            "MACD",
            round(df["MACD"].iloc[-1], 2)
        )

    with col9:
        st.metric(
            "News Sentiment",
            sentiment
        )


    # ===================================
    # CANDLESTICK CHART
    # ===================================

    st.subheader("📈 Candlestick Chart")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=stock
            )
        ]
    )

    fig.update_layout(
        title=f"{stock} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================
    # TREND ANALYSIS
    # ===================================

    st.subheader("📉 Trend Analysis")

    trend_fig = go.Figure()

    trend_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close Price"
        )
    )

    trend_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA20"],
            name="SMA20"
        )
    )

    trend_fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SMA50"],
            name="SMA50"
        )
    )

    trend_fig.update_layout(
        height=600
    )

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    )
    # ===================================
    # DATA TABLE
    # ===================================

    st.subheader("📋 Latest Market Data")
    st.dataframe(
    df.tail(10)
    )


elif page == "Portfolio":

    # ===================================
    # PORTFOLIO RECOMMENDATION ENGINE
    # ===================================

    st.subheader("💰 Portfolio Recommendation Engine")

    investment = st.number_input(
        "Investment Amount ($)",
        min_value=100,
        value=10000
    )

    risk = st.selectbox(
        "Risk Level",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    if risk == "Low":

        allocation = {
            "AAPL": 40,
            "MSFT": 30,
            "GOOGL": 20,
            "NVDA": 10
        }

    elif risk == "Medium":

        allocation = {
            "AAPL": 30,
            "MSFT": 25,
            "NVDA": 20,
            "GOOGL": 15,
            "TSLA": 10
        }

    else:

        allocation = {
            "NVDA": 35,
            "TSLA": 30,
            "META": 20,
            "AAPL": 15
        }

    portfolio_df = pd.DataFrame(
        {
            "Stock": allocation.keys(),
            "Allocation %": allocation.values()
        }
    )

    portfolio_df["Investment"] = (
        portfolio_df["Allocation %"]
        * investment
        / 100
    )

    st.dataframe(portfolio_df)

 # ===================================
    # PORTFOLIO PIE CHART
    # ===================================

    fig = px.pie(
        portfolio_df,
        names="Stock",
        values="Allocation %",
        title="Portfolio Allocation"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ===================================
    # PORTFOLIO PROJECTION
    # ===================================

    expected_return = {
        "Low": 8,
        "Medium": 15,
        "High": 25
    }

    expected_profit = (
        investment
        * expected_return[risk]
        / 100
    )

    future_value = (
        investment
        + expected_profit
    )

    st.subheader("📊 Portfolio Projection")

    col10, col11, col12 = st.columns(3)

    with col10:
        st.metric(
            "Expected Return %",
            f"{expected_return[risk]}%"
        )

    with col11:
        st.metric(
            "Expected Profit",
            f"${expected_profit:,.2f}"
        )

    with col12:
        st.metric(
            "Future Portfolio Value",
            f"${future_value:,.2f}"
        )


elif page == "Prediction History":

    # ===================================
    # PREDICTION HISTORY
    # ===================================

    history = pd.DataFrame(
        {
            "Date": [pd.Timestamp.now()],
            "Stock": [stock],
            "Current Price": [current_price],
            "Predicted Price": [predicted_price],
            "Signal": [signal]
        }
    )

    file_path = "data/prediction_history.csv"

    if os.path.exists(file_path):

        history.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        history.to_csv(
            file_path,
            index=False
        )
    history_df = pd.read_csv(file_path)

    st.dataframe(
        history_df.tail(20)
    )

elif page == "Model Comparison":

    st.subheader("🏆 Model Comparison")

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Random Forest",
                "LSTM"
            ],
            "MAE": [
                15.84,
                9.94
            ],
            "RMSE": [
                22.22,
                11.67
            ],
            "MAPE": [
                5.64,
                3.72
            ]
        }
    )

    st.dataframe(comparison_df)

    st.success(
        "🏆 Best Model: LSTM"
    )

elif page == "About":

    st.title("📈 About AI Investment Advisor")

    st.markdown("""
    ## Features

    - LSTM Stock Prediction
    - News Sentiment Analysis
    - Portfolio Recommendation Engine
    - Technical Indicators
    - Prediction History Tracking

    ## Tech Stack

    - Python
    - TensorFlow
    - Streamlit
    - Plotly
    - Pandas
    - NumPy
    - YFinance
    - Scikit-Learn

    ## Model Performance

    - MAE : 9.94
    - RMSE : 11.67
    - MAPE : 3.72%

    ## Developed For

    Financial Intelligence and Stock Market Forecasting using Deep Learning.
    """)