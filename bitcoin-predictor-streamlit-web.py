
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import base64
import numpy as np
import ta

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
st.title("📈 Crypto Prediction Maker (Enhanced)")
st.caption("🔌 Powered by Binance API + AI + Technical Indicators")
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
        candle_limit = 1440
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={candle_limit}"
        data = requests.get(url).json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "_", "_", "_", "_", "_", "_"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df["minute"] = range(len(df))

        # --- Technical Indicators ---
        df["ma_10"] = df["close"].rolling(window=10).mean()
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"]).rsi()
        df["macd"] = ta.trend.MACD(close=df["close"]).macd()
        df.dropna(inplace=True)

        features = ["minute", "open", "high", "low", "volume", "ma_10", "rsi", "macd"]
        X = df[features]
        y = df["close"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1)
        model.fit(X_train, y_train)

        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.write(f"📉 Model MAE (test): {mae:.4f}")

        future_minute = df["minute"].max() + minutes_to_predict
        latest = df.iloc[-1:]
        future_features = pd.DataFrame({
            "minute": [future_minute],
            "open": latest["open"].values,
            "high": latest["high"].values,
            "low": latest["low"].values,
            "volume": latest["volume"].values,
            "ma_10": latest["ma_10"].values,
            "rsi": latest["rsi"].values,
            "macd": latest["macd"].values
        })
        future_scaled = scaler.transform(future_features)
        prediction = model.predict(future_scaled)[0]
        st.success(f"🚀 Predicted Price at {user_time.strftime('%H:%M')} IST: ${prediction:,.2f}")

        # --- Chart ---
        st.subheader(f"📊 {selected_crypto} - Last 24 Hours")
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color='lime',
            decreasing_line_color='red'
        )])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # --- Export ---
        export_df = pd.DataFrame({
            "Crypto": [symbol],
            "Mode": ["XGBoost + Technical"],
            "Minutes Ahead": [minutes_to_predict],
            "Predicted Price": [round(prediction, 2)],
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
st.caption("Made with ❤️ using Python, Streamlit, Binance API, and AI-powered Technical Indicators.")
