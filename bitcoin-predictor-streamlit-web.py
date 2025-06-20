# Crypto Prediction Maker Pro (Dashboard Version)
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import base64
import numpy as np
import ta

# --- Setup ---
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
now_local = datetime.now(LOCAL_TZ)
st.set_page_config(page_title="Crypto Dashboard", layout="wide", page_icon="₿")

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌌 Dark Mode", value=True)
model_choice = st.sidebar.selectbox("📊 Model", ["XGBoost", "Linear Regression"])
live_refresh = st.sidebar.toggle("🔄 Auto Refresh Live Price", value=False)

# --- Apply Theme ---
if dark_mode:
    st.markdown("""
        <style>
        body {background-color: #111; color: #eee;}
        .stApp {background-color: #111; color: #eee;}
        </style>
    """, unsafe_allow_html=True)

# --- Crypto Selection ---
symbols = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT"
}
selected_crypto = st.sidebar.selectbox("🔑 Select Cryptocurrency", list(symbols.keys()))
symbol = symbols[selected_crypto]

# --- Tabs Layout ---
tabs = st.tabs(["Live Price", "Prediction", "Accuracy", "Chart", "Export"])

# ========== Tab 1: Live Price ==========
with tabs[0]:
    st.header(f"💰 Live {selected_crypto} Price (USD)")
    if st.button("Fetch Live Price") or live_refresh:
        try:
            res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            price = float(res.json()["price"])
            st.success(f"₿ 1 {symbol[:-4]} = ${price:,.2f}")
        except:
            st.error("Failed to fetch live price.")

# ========== Tab 2: Prediction ==========
with tabs[1]:
    st.header(f"📊 Predict {selected_crypto} Price")
    mode = st.radio("📌 Time Selection", ["Custom Time (HH:MM)", "Quick Option"])
    minutes_to_predict, user_time = None, None

    if mode == "Custom Time (HH:MM)":
        custom_time = st.text_input("⏳ Enter Time in IST (e.g., 21:45)")
        if st.button("Predict"):
            try:
                user_time = datetime.strptime(custom_time, "%H:%M").replace(
                    year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=LOCAL_TZ)
                if user_time < now_local:
                    user_time += timedelta(days=1)
                minutes_to_predict = int((user_time - now_local).total_seconds() // 60)
                if 1 <= minutes_to_predict <= 240:
                    st.success(f"Predicting {minutes_to_predict} min ahead (at {user_time.strftime('%H:%M')} IST)...")
                else:
                    st.warning("Only 1-240 minutes allowed.")
                    minutes_to_predict = None
            except:
                st.error("Invalid format. Use HH:MM (24h)")

    else:
        quick_option = st.selectbox("⏰ Quick Time", ["10m", "15m", "30m", "1h", "2h", "4h"])
        label_to_minutes = {"10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
        if st.button("Predict"):
            minutes_to_predict = label_to_minutes[quick_option]
            user_time = now_local + timedelta(minutes=minutes_to_predict)
            st.success(f"Predicting {minutes_to_predict} min ahead (at {user_time.strftime('%H:%M')} IST)...")

    if minutes_to_predict:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1440"
        df = pd.DataFrame(requests.get(url).json(), columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["minute"] = range(len(df))
        df["ma_10"] = df["close"].rolling(window=10).mean()
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df.dropna(inplace=True)

        features = ["minute", "open", "high", "low", "volume", "ma_10", "rsi", "macd"]
        X, y = df[features], df["close"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=150) if model_choice == "XGBoost" else LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        prediction = model.predict(scaler.transform(pd.DataFrame({
            "minute": [df["minute"].max() + minutes_to_predict],
            "open": df.iloc[-1:]["open"].values,
            "high": df.iloc[-1:]["high"].values,
            "low": df.iloc[-1:]["low"].values,
            "volume": df.iloc[-1:]["volume"].values,
            "ma_10": df.iloc[-1:]["ma_10"].values,
            "rsi": df.iloc[-1:]["rsi"].values,
            "macd": df.iloc[-1:]["macd"].values
        })))[0]

        latest_price = df["close"].iloc[-1]
        delta = prediction - latest_price
        st.success(f"🚀 Predicted Price at {user_time.strftime('%H:%M')} IST: ${prediction:,.2f}")
        st.info(f"📊 Change: ${delta:.2f} ({'Increase 📈' if delta > 0 else 'Decrease 📉'})")

# ========== Tab 3: Accuracy ==========
with tabs[2]:
    st.header("🔢 Model Evaluation")
    if minutes_to_predict:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        naive_mae = mean_absolute_error(y_test, y_test.shift(1).fillna(method='bfill'))
        cv_score = -np.mean(cross_val_score(model, X_scaled, y, scoring='neg_mean_absolute_error', cv=5))
        st.write(f"**Model MAE:** {mae:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.write(f"**Cross-Val MAE:** {cv_score:.4f}")
        st.write(f"**Naive Baseline MAE:** {naive_mae:.4f}")
        st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}))

# ========== Tab 4: Chart ==========
with tabs[3]:
    st.header(f"📊 {selected_crypto} - Candlestick")
    if minutes_to_predict:
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color='lime', decreasing_line_color='red')])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ========== Tab 5: Export ==========
with tabs[4]:
    st.header("📄 Export Report")
    if minutes_to_predict:
        result = pd.DataFrame({
            "Crypto": [symbol], "Model": [model_choice],
            "Minutes Ahead": [minutes_to_predict],
            "Predicted Price": [round(prediction, 2)],
            "Time (IST)": [user_time.strftime("%H:%M")],
            "Delta": [round(delta, 2)]
        })
        csv = result.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Prediction CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, Binance API & Machine Learning.")
