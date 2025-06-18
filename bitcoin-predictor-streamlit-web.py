import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Page Config ---
st.set_page_config(
    page_title="Bitcoin Prediction Maker",
    layout="centered",
    page_icon="₿"
)

# --- UI Header ---
st.title("📈 Bitcoin Prediction Maker")
st.caption("🔌 Powered by CoinGecko — No API key required")

# --- Live BTC Price ---
st.subheader("💰 Get Current BTC Price (USD)")

if st.button("Fetch Live Price"):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
        price = res.json()["bitcoin"]["usd"]
        st.success(f"₿ 1 BTC = ${price:,.2f}")
    except Exception as e:
        st.error(f"Error fetching price: {e}")

# --- Prediction Section ---
st.divider()
st.subheader("📊 Predict BTC Price After 10 Minutes")

if st.button("Run Prediction & Show Chart"):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1&interval=minutely"
        data = requests.get(url).json()["prices"]

        if not data or len(data) < 30:
            st.error("❌ Not enough data to run prediction. Try again later.")
        else:
            df = pd.DataFrame(data, columns=["timestamp", "price"])
            df["minute"] = range(len(df))

            model = LinearRegression()
            model.fit(df[["minute"]], df["price"])
            predicted_price = model.predict([[df["minute"].max() + 10]])[0]

            st.success(f"📈 Predicted BTC Price (in 10 mins): ${predicted_price:,.2f}")
            st.line_chart(df["price"], use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.divider()
st.caption("Made with ❤️ using Python + Streamlit • Open Source • Mobile Friendly")
