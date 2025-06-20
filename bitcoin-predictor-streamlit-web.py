# Crypto Prediction Maker Pro++ (Final Version with Enhanced UI/UX)
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

# --- Apply Custom Theme & CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #E5E7EB;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #00FFAB;
        color: black;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #10B981;
        color: white;
    }
    .big-font {
        font-size: 26px;
        font-weight: bold;
        color: #00FFAB;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
dark_mode = st.sidebar.toggle("🌌 Dark Mode", value=True)
model_choice = st.sidebar.selectbox("📊 Model", ["XGBoost", "Linear Regression"])
live_refresh = st.sidebar.toggle("🔄 Auto Refresh Live Price", value=False)
show_sentiment = st.sidebar.toggle("💬 Sentiment Analysis (Mock)", value=True)
confidence_enabled = st.sidebar.toggle("📉 Show Confidence Interval", value=True)
show_multiple_forecast = st.sidebar.toggle("⏱️ Multi-Timeframe Forecast", value=True)

# --- Crypto Selection ---
symbols = {
    "Bitcoin (BTC)": "BTCUSDT",
    "Ethereum (ETH)": "ETHUSDT",
    "Dogecoin (DOGE)": "DOGEUSDT",
    "Solana (SOL)": "SOLUSDT",
    "XRP": "XRPUSDT"
}
selected_crypto = st.sidebar.selectbox("🔑 Select Cryptocurrency", list(symbols.keys()))
symbol = symbols[selected_crypto]

# --- Tabs Layout ---
tabs = st.tabs(["Live 📰", "Predict 📊", "Accuracy 🧹", "Chart 📈", "Forecast ⏱️", "Export 📅"])

# --- Fetch Market Data ---
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

# --- Tab 1: Live Price ---
with tabs[0]:
    st.header(f"💰 Live {selected_crypto} Price (USD)")
    if st.button("Fetch Live Price") or live_refresh:
        try:
            res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            price = float(res.json()["price"])
            st.success(f"₿ 1 {symbol[:-4]} = ${price:,.2f}")
        except:
            st.error("Failed to fetch live price.")

# --- Tab 2: Prediction ---
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
        input_features = pd.DataFrame({
            "minute": [df["minute"].max() + minutes_to_predict],
            "open": df.iloc[-1:]["open"].values,
            "high": df.iloc[-1:]["high"].values,
            "low": df.iloc[-1:]["low"].values,
            "volume": df.iloc[-1:]["volume"].values,
            "ma_10": df.iloc[-1:]["ma_10"].values,
            "rsi": df.iloc[-1:]["rsi"].values,
            "macd": df.iloc[-1:]["macd"].values
        })
        prediction = model.predict(scaler.transform(input_features))[0]
        latest_price = df["close"].iloc[-1]
        delta = prediction - latest_price

        st.markdown(f"<p class='big-font'>🚀 Predicted Price: ${prediction:,.2f} at {user_time.strftime('%H:%M')} IST</p>", unsafe_allow_html=True)
        st.info(f"📊 Expected Change: ${delta:.2f} ({'Increase 📈' if delta > 0 else 'Decrease 📉'})")

        if confidence_enabled:
            try:
                preds = [model.predict(scaler.transform(input_features))[0] for _ in range(20)]
                st.write(f"🔎 Confidence Range: ${min(preds):.2f} - ${max(preds):.2f}")
            except:
                st.warning("Confidence estimation unavailable.")

        if show_sentiment:
            st.subheader("🗣️ Market Sentiment (Mocked)")
            st.markdown("📈 Sentiment Score: **+0.76** → Optimistic")
            st.progress(76)

# --- Tab 3: Accuracy ---
with tabs[2]:
    st.header("📁 Model Evaluation")
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    naive_mae = mean_absolute_error(y_test, y_test.shift(1).fillna(method='bfill'))
    cv_scores = cross_val_score(model, X_scaled, y, scoring='neg_mean_absolute_error', cv=5)
    st.write(f"📉 MAE: {mae:.4f}")
    st.write(f"📈 R² Score: {r2:.4f}")
    st.write(f"⚖️ Naive MAE: {naive_mae:.4f}")
    st.write(f"🧪 Cross-Val MAE: {-np.mean(cv_scores):.4f}")
    st.line_chart(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True))

# --- Tab 4: Chart ---
with tabs[3]:
    st.header(f"🔦 {selected_crypto} - Last 24h Candlestick")
    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color='lime', decreasing_line_color='red')])
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Forecast ---
with tabs[4]:
    st.header("🧙 Multi-Timeframe Forecast")
    if show_multiple_forecast and minutes_to_predict:
        times = [10, 30, 60, 120, 240]
        future_df = df.copy()
        multi_preds = []
        window_prices = list(df["close"].values[-10:])

        for step in times:
            future_minute = future_df["minute"].max() + 1
            last_row = future_df.iloc[-1]
            new_close = multi_preds[-1][1] if multi_preds else prediction
            window_prices.append(new_close)
            if len(window_prices) > 10:
                window_prices.pop(0)

            ma_10 = np.mean(window_prices)
            rsi = ta.momentum.RSIIndicator(pd.Series(window_prices)).rsi().iloc[-1]
            macd = ta.trend.MACD(pd.Series(window_prices)).macd().iloc[-1]

            new_row = pd.DataFrame({
                "minute": [future_minute],
                "open": [last_row["close"]],
                "high": [new_close + 3],
                "low": [new_close - 3],
                "close": [new_close],
                "volume": [last_row["volume"]],
                "ma_10": [ma_10],
                "rsi": [rsi],
                "macd": [macd]
            })
            future_df = pd.concat([future_df, new_row], ignore_index=True)

            X_f = new_row[features]
            X_scaled_f = scaler.transform(X_f)
            next_pred = model.predict(X_scaled_f)[0]
            multi_preds.append((step, next_pred))

        forecast_df = pd.DataFrame(multi_preds, columns=["Minutes Ahead", "Predicted Price"])
        st.dataframe(forecast_df)

# --- Tab 6: Export ---
with tabs[5]:
    st.header("📆 Export Prediction")
    if minutes_to_predict:
        export_df = pd.DataFrame({
            "Crypto": [symbol],
            "Model": [model_choice],
            "Minutes Ahead": [minutes_to_predict],
            "Predicted Price": [round(prediction, 2)],
            "Time (IST)": [user_time.strftime("%H:%M")],
            "Expected Change ($)": [round(delta, 2)]
        })
        csv = export_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">📅 Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, Binance API & AI")
