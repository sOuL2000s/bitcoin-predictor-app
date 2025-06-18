import streamlit as st
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Bitcoin Prediction Maker", layout="centered", page_icon="₿")
st.title("📈 Bitcoin Prediction Maker")
st.caption("🔌 Powered by CoinGecko — No API key required")

st.subheader("💰 Get Current BTC Price (USD)")

if st.button("Fetch Live Price"):
    try:
        res = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", timeout=10)
        price = res.json()["bitcoin"]["usd"]
        st.success(f"₿ 1 BTC = ${price:,.2f}")
    except Exception as e:
        st.error(f"Error fetching price: {e}")

st.divider()
st.subheader("📊 Predict BTC Price After 10 Minutes")

if st.button("Run Prediction & Show Chart"):
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "1",
            "interval": "minutely"
        }
        headers = {
            "accept": "application/json",
            "user-agent": "Mozilla/5.0"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        if "prices" not in data:
            raise ValueError("API response does not contain 'prices' data.")

        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["minute"] = range(len(df))

        model = LinearRegression()
        model.fit(df[["minute"]], df["price"])
        predicted_price = model.predict([[df["minute"].max() + 10]])[0]

        st.success(f"📈 Predicted BTC Price (in 10 mins): ${predicted_price:,.2f}")
        st.line_chart(df["price"], use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Made with ❤️ using Python + Streamlit • Open Source • Mobile Friendly")
