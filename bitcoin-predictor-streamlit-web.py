import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import base64
import numpy as np

# --- Set Timezone ---
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
now_local = datetime.now(LOCAL_TZ)

# --- Page Config ---
st.set_page_config(page_title="Crypto Prediction Maker", layout="centered", page_icon="₿")

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.toggle("🌌 Dark Mode", value=True)

if dark_mode:
    st.markdown("""
        <style>
        body {background-color: #111; color: #eee;}
        .stApp {background-color: #111; color: #eee;}
        </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.title("📈 Crypto Prediction Maker")
st.caption("🔌 Powered by Binance API + AI")
st.markdown(f"🕒 **Current IST Time:** {now_local.strftime('%H:%M:%S')}")

# --- Crypto Selection ---
symbols = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT"
}
selected_crypto = st.selectbox("🔑 Select Cryptocurrency", list(symbols.keys()))
symbol = symbols[selected_crypto]

# --- Live Price ---
st.subheader(f"💰 Live {selected_crypto} Price (USD)")
if st.button("Fetch Live Price"):
    try:
        res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
        res.raise_for_status()
        price = float(res.json()["price"])
        st.success(f"₿ 1 {symbol[:-4]} = ${price:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Prediction Mode ---
st.divider()
st.subheader(f"📊 Predict {selected_crypto} Price")
mode = st.radio("📌 Prediction Mode", ["Custom Time (HH:MM)", "Quick Option (10m, 15m, etc.)"])

minutes_to_predict, user_time = None, None

# --- Custom Time ---
if mode == "Custom Time (HH:MM)":
    custom_time = st.text_input("⏳ Enter Time in IST (e.g., 21:45)")
    if st.button("Predict"):
        try:
            user_time = datetime.strptime(custom_time, "%H:%M").replace(
                year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=LOCAL_TZ
            )
            if user_time < now_local:
                user_time += timedelta(days=1)
            minutes_to_predict = int((user_time - now_local).total_seconds() // 60)
            if not 1 <= minutes_to_predict <= 240:
                st.warning("Prediction allowed only for 1 to 240 minutes ahead.")
                minutes_to_predict = None
            else:
                st.info(f"Predicting {minutes_to_predict} min ahead (at {user_time.strftime('%H:%M')} IST)...")
        except:
            st.error("Invalid format. Use HH:MM (24h)")

# --- Quick Option ---
else:
    quick_option = st.selectbox("⏰ Quick Time", ["10 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "2 Hours", "4 Hours"])
    label_to_minutes = {
        "10 Minutes": 10,
        "15 Minutes": 15,
        "30 Minutes": 30,
        "1 Hour": 60,
        "2 Hours": 120,
        "4 Hours": 240
    }
    if st.button("Predict"):
        minutes_to_predict = label_to_minutes[quick_option]
        user_time = now_local + timedelta(minutes=minutes_to_predict)
        st.info(f"Predicting {minutes_to_predict} min ahead (at {user_time.strftime('%H:%M')} IST)...")

# --- Prediction Logic ---
if minutes_to_predict:
    try:
        # Use 240 minutes of data (4 hours) for better training
        candle_limit = 240
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={candle_limit}"
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)
        df["minute"] = range(len(df))

        # --- Linear Regression ---
        model_lr = LinearRegression().fit(df[["minute"]], df["close"])
        lin_pred = model_lr.predict([[df["minute"].max() + minutes_to_predict]])[0]

        # --- XGBoost Regressor ---
        X = df["minute"].values.reshape(-1, 1)
        y = df["close"].values
        model_xgb = XGBRegressor(n_estimators=100, max_depth=4)
        model_xgb.fit(X, y)
        xgb_pred = model_xgb.predict([[df["minute"].max() + minutes_to_predict]])[0]

        # --- Output ---
        st.success(f"🔹 Linear Regression: ${lin_pred:,.2f}")
        st.success(f"⚡ XGBoost Prediction: ${xgb_pred:,.2f}")

        # --- Candlestick Chart ---
        st.subheader(f"📊 {selected_crypto} - Last {candle_limit} Minutes")
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"].astype(float),
            high=df["high"].astype(float),
            low=df["low"].astype(float),
            close=df["close"].astype(float),
            increasing_line_color='lime',
            decreasing_line_color='red'
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Export CSV ---
        st.subheader("📄 Export Prediction")
        export_df = pd.DataFrame({
            "Crypto": [symbol],
            "Mode": ["XGBoost & Linear"],
            "Minutes Ahead": [minutes_to_predict],
            "Predicted Price (LR)": [round(lin_pred, 2)],
            "Predicted Price (XGBoost)": [round(xgb_pred, 2)],
            "Time (IST)": [user_time.strftime("%H:%M")]
        })

        csv = export_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">📥 Download Prediction CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")

# --- Footer ---
st.divider()
st.caption("Made with ❤️ in Python + Streamlit • Mobile Ready • Multi-Crypto • XGBoost + ML Powered")
