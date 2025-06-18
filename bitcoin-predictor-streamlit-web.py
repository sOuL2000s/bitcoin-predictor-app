import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Bitcoin Prediction Maker", page_icon="📉")

st.title("📉 Bitcoin Prediction Maker")
st.caption("🔧 Powered by CoinGecko & Binance APIs — No API key required")

# --- BTC Live Price ---
st.subheader("💰 Get Current BTC Price (USD)")
if st.button("Fetch Live Price"):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        res.raise_for_status()
        btc_price = res.json()["bitcoin"]["usd"]
        st.success(f"Current BTC Price: ${btc_price}")
    except Exception as e:
        st.error("⚠️ Failed to fetch BTC price from CoinGecko.")

# --- Historical Data Fetch ---
@st.cache_data(ttl=600)
def get_btc_data(minutes=20):
    def get_btc_data(minutes=20):
        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {"vs_currency": "usd", "days": "1", "interval": "minute"}
            res = requests.get(url, params=params)
            prices = res.json()["prices"]  # List of [timestamp, price]
    
            df = pd.DataFrame(prices, columns=["time", "price"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df = df.tail(minutes)
            df.reset_index(drop=True, inplace=True)
            df["minute"] = df.index
            return df
        except:
            return pd.DataFrame()

# --- Prediction ---
st.subheader("📊 Predict BTC Price After 10 Minutes")
if st.button("Run Prediction & Show Chart"):
    df = get_btc_data()
    if df.empty or len(df) < 10:
        st.error("❌ Not enough data to run prediction. Try again later.")
    else:
        model = LinearRegression().fit(df[["minute"]], df["price"])
        prediction = model.predict([[len(df) + 10]])[0]
        st.success(f"🔮 Predicted BTC Price in 10 Minutes: ${prediction:.2f}")


        # Chart
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["close"], label="Historical Price")
        ax.scatter(df["time"].iloc[-1] + timedelta(minutes=10), predicted_price, color="red", label="Prediction")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USD)")
        ax.set_title("BTC Price Trend + 10-min Prediction")
        ax.legend()
        st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + Python • Mobile-friendly • Open-source")
