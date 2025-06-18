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
st.caption("🔌 Powered by Binance API — No API key required")

# --- Live BTC Price ---
st.subheader("💰 Get Current BTC Price (USD)")

if st.button("Fetch Live Price"):
    try:
        res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT")
        res.raise_for_status()
        price = float(res.json()["price"])
        st.success(f"₿ 1 BTC = ${price:,.2f}")
    except Exception as e:
        st.error(f"Error fetching price: {e}")

# --- Prediction Section ---
st.divider()
st.subheader("📊 Predict BTC Price")

# Prediction time options
prediction_options = {
    "10 Minutes": 10,
    "15 Minutes": 15,
    "30 Minutes": 30,
    "1 Hour": 60
}

selected_option = st.selectbox("⏳ Select Prediction Time", list(prediction_options.keys()))
minutes_to_predict = prediction_options[selected_option]

if st.button(f"Predict Price After {selected_option}"):
    try:
        # Fetch last 60 minutes data from Binance
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=60"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Check for valid response
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Binance API returned no data or is blocked.")

        # Prepare DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "_", "_", "_", "_", "_", "_"
        ])
        df["close"] = df["close"].astype(float)
        df["minute"] = range(len(df))

        # Train model
        model = LinearRegression()
        model.fit(df[["minute"]], df["close"])

        # Predict future price
        future_minute = df["minute"].max() + minutes_to_predict
        predicted_price = model.predict([[future_minute]])[0]

        st.success(f"📈 Predicted BTC Price (in {selected_option}): ${predicted_price:,.2f}")
        st.line_chart(df["close"], use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Footer ---
st.divider()
st.caption("Made with ❤️ using Python + Streamlit • Open Source • Mobile Friendly")
