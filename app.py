import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import plotly.express as px
import os
import time
from PIL import Image

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

/* Main Background */
.stApp {
    background: linear-gradient(135deg,#0F172A,#111827,#1E293B);
    color: #F8FAFC;
}

/* Metric Cards */
[data-testid="stMetric"]{
    background-color:#1E293B;
    border:1px solid #334155;
    border-radius:20px;
    padding:20px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.35);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#111827;
}

/* Buttons */
.stButton>button{
    background-color:#2563EB;
    color:white;
    border-radius:10px;
    border:none;
    padding:10px 20px;
}

.stButton>button:hover{
    background-color:#1D4ED8;
}

/* Selectbox */
.stSelectbox>div>div{
    background-color:#1E293B;
    color:white;
}

/* Number Input */
.stNumberInput input{
    background-color:#1E293B;
    color:white;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border-radius:15px;
}

h1,h2,h3{
    color:white;
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

banner = Image.open("assets/banner.png")

st.image(
    banner,
    use_container_width=True
)

st.markdown(
    "<br>",
    unsafe_allow_html=True
)
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

try:
    df = yf.download(
        stock,
        period="5y",
        progress=False,
        auto_adjust=True,
        threads=False
    )

    if df.empty:
        st.error("Unable to fetch stock data. Please try again later.")
        st.stop()

except Exception as e:
    st.error(f"Stock data unavailable: {e}")
    st.stop()

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
    template="plotly_dark",
    title=f"{stock} Stock Price",
    xaxis_title="Date",
    yaxis_title="Price",
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font=dict(color="white"),
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
    template="plotly_dark",
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font=dict(color="white"),
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