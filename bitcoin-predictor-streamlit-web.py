import streamlit as st
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bitcoin Prediction Maker", layout="centered", page_icon="📈")

st.title("📉 Bitcoin Prediction Maker")
st.caption("🔧 Powered by CoinGecko & Binance APIs — No API key required")

st.subheader("💰 Get Current BTC Price (USD)")

def fetch_current_price():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        response = requests.get(url)
        price = response.json()["bitcoin"]["usd"]
        return price
    except:
        return None

if st.button("Fetch Live Price"):
    price = fetch_current_price()
    if price:
        st.success(f"Current BTC Price: ${price}")
    else:
        st.error("❌ Failed to fetch BTC price. Please check your internet or try later.")

# ==============================
st.markdown("---")
st.subheader("📊 Predict BTC Price After 10 Minutes")

def get_coingecko_data():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=minutely"
        res = requests.get(url, timeout=10)
        prices = res.json().get("prices", [])
        return [p[1] for p in prices]
    except:
        return []

def get_binance_data():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=100"
        res = requests.get(url, timeout=10)
        data = res.json()
        return [float(x[4]) for x in data]  # Close prices
    except:
        return []

def get_historical_prices():
    prices = get_coingecko_data()
    if len(prices) < 2:
        prices = get_binance_data()
    return prices

if st.button("Run Prediction & Show Chart"):
    prices = get_historical_prices()

    if not prices or len(prices) < 2:
        st.error("❌ Not enough data to run prediction. Try again later.")
        st.stop()

    # Prepare data for regression
    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices)

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predict 10 minutes into the future
    future_index = len(prices) + 10
    predicted_price = model.predict(np.array([[future_index]]))[0]

    st.success(f"📈 Predicted BTC Price After 10 Minutes: ${predicted_price:.2f}")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(prices, label="Historical Price")
    ax.axvline(x=len(prices), linestyle="--", color="gray")
    ax.plot(len(prices)+10, predicted_price, 'ro', label="Predicted Price")
    ax.legend()
    ax.set_title("BTC Price Prediction")
    st.pyplot(fig)

# ==============================
st.markdown("---")
st.caption("Made with ❤️ using Python + Streamlit • Open Source • Mobile Friendly")
