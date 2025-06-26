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
import time # For time.sleep during multiple API calls

# --- Setup ---
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
now_local = datetime.now(LOCAL_TZ)
st.set_page_config(page_title="Crypto Dashboard", layout="centered", page_icon="₿")

# --- Apply Custom Theme & Responsive CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #E5E7EB;
        font-family: 'Segoe UI', sans-serif;
        padding: 10px;
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
    @media only screen and (max-width: 768px) {
        .big-font {
            font-size: 20px !important;
        }
        .stButton>button {
            padding: 6px 12px !important;
            font-size: 14px !important;
        }
        .stSidebar {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar.expander("⚙️ Settings", expanded=True):
    dark_mode = st.toggle("🌌 Dark Mode", value=True)
    model_choice = st.selectbox("📊 Model", ["XGBoost", "Linear Regression"])
    live_refresh = st.toggle("🔄 Auto Refresh Live Price", value=False)
    show_sentiment = st.toggle("💬 Sentiment Analysis (Mock)", value=True)
    confidence_enabled = st.toggle("📉 Show Confidence Interval", value=True)
    show_multiple_forecast = st.toggle("⏱️ Multi-Timeframe Forecast", value=True)
    # New setting for data fetching
    data_days_limit = st.slider("Historical Data (days)", 1, 7, 3) # Fetch up to 7 days of 1-minute data

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

@st.cache_data(ttl=60) # Cache data for 60 seconds to avoid excessive API calls
def fetch_and_process_data(symbol, days_limit):
    """Fetches historical kline data from Binance and processes it."""
    all_data = []
    end_time_ms = int(datetime.now().timestamp() * 1000) # Current time in ms

    # Fetch data in chunks as Binance limit is 1000 klines per request
    # To get `days_limit` days of 1-minute data:
    # 1 day = 1440 minutes (klines)
    # Num requests needed = (days_limit * 1440) / 1000
    num_requests = int(np.ceil((days_limit * 1440) / 1000))

    st.info(f"Fetching ~{days_limit} days of 1-minute data for {selected_crypto}...")
    progress_bar = st.progress(0)

    for i in range(num_requests):
        # Calculate start_time for the next chunk
        # Each request gets 1000 minutes of data, so go back 1000 * (i+1) minutes
        start_time_ms = end_time_ms - ((i + 1) * 1000 * 60 * 1000)
        
        # Ensure start_time doesn't go too far back
        if start_time_ms < 0:
            start_time_ms = 0

        # Corrected Binance API URL
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1000&endTime={end_time_ms}"
        try:
            res = requests.get(url)
            res.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            klines = res.json()
            all_data.extend(klines)
            end_time_ms = klines[0][0] - 1 if klines and klines[0] else end_time_ms # Set end_time for next loop
            progress_bar.progress((i + 1) / num_requests)
            time.sleep(0.1) # Be polite to the API
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data from Binance API: {e}")
            break
    
    progress_bar.empty()

    if not all_data:
        st.warning("No data fetched. Please try again or check symbol/API access.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df = df.iloc[::-1].reset_index(drop=True) # Reverse order to be chronological
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # --- Feature Engineering ---
    df["minute_idx"] = range(len(df)) # Renamed from 'minute' to avoid confusion

    # Moving Averages
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

    # Momentum Indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()

    # Volatility Indicators
    df["bollinger_hband"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
    df["bollinger_lband"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

    # Trend Indicators
    adx_indicator = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx_indicator.adx()
    df["plus_di"] = adx_indicator.adx_pos()
    df["minus_di"] = adx_indicator.adx_neg()

    # Volume Indicators
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    # Lagged features (capturing recent price memory)
    for lag in [1, 2, 3, 5, 10, 15]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
        df[f"rsi_lag_{lag}"] = df["rsi"].shift(lag) # Lagged RSI can be powerful

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek # Monday=0, Sunday=6

    # Drop rows with NaN values introduced by rolling windows and lags
    df.dropna(inplace=True)

    return df

df = fetch_and_process_data(symbol, data_days_limit)

if df.empty:
    st.stop() # Stop execution if no data is available

# Define features after they've been created in the dataframe
features = [
    "minute_idx", "open", "high", "low", "volume",
    "ma_10", "ma_20", "ema_12", "ema_26", "rsi", "macd",
    "bollinger_hband", "bollinger_lband", "atr", "adx", "plus_di", "minus_di", "obv",
    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10", "close_lag_15",
    "volume_lag_1", "volume_lag_2", "volume_lag_3", "volume_lag_5", "volume_lag_10", "volume_lag_15",
    "rsi_lag_1", "rsi_lag_2", "rsi_lag_3", "rsi_lag_5", "rsi_lag_10", "rsi_lag_15",
    "hour", "day_of_week"
]

# Ensure all features exist in df before selecting
for f in features:
    if f not in df.columns:
        st.error(f"Missing feature in DataFrame: {f}. This indicates an issue with feature engineering or data availability.")
        st.stop()

X, y = df[features], df["close"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use a time-based split for more realistic evaluation (e.g., last 20% for testing)
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Model Training
model = XGBRegressor(n_estimators=200, random_state=42) if model_choice == "XGBoost" else LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate prediction_errors globally after model training
prediction_errors = (y_test - y_pred).abs()

# --- Define tabs here ---
tabs = st.tabs(["💰 Live Price", "📊 Prediction", "📁 Model Accuracy", "🔦 Chart", "🧙 Forecast", "📆 Export"])

# --- Tab 1: Live Price ---
with tabs[0]:
    st.header(f"💰 Live {selected_crypto} Price (USD)")
    if st.button("Fetch Live Price") or live_refresh:
        try:
            # Corrected Binance API URL
            res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            res.raise_for_status()
            price = float(res.json()["price"])
            st.success(f"₿ 1 {symbol[:-4]} = ${price:,.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch live price: {e}")

# --- Tab 2: Prediction ---
with tabs[1]:
    st.header(f"📊 Predict {selected_crypto} Price")
    mode = st.radio("📌 Time Selection", ["Custom Time (HH:MM)", "Quick Option"])
    minutes_to_predict, user_time = None, None
    prediction = None # Initialize prediction

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
                    st.warning("Only 1-240 minutes allowed for custom prediction.")
                    minutes_to_predict = None
            except ValueError:
                st.error("Invalid format. Use HH:MM (24h)")
    else:
        quick_option = st.selectbox("⏰ Quick Time", ["10m", "15m", "30m", "1h", "2h", "4h"])
        label_to_minutes = {"10m": 10, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
        if st.button("Predict"):
            minutes_to_predict = label_to_minutes[quick_option]
            user_time = now_local + timedelta(minutes=minutes_to_predict)
            st.success(f"Predicting {minutes_to_predict} min ahead (at {user_time.strftime('%H:%M')} IST)...")

    if minutes_to_predict:
        # Create input features for prediction
        last_row = df.iloc[-1]
        
        # Initialize input_features with base values, ensuring they are scalar floats
        input_features_dict = {
            "minute_idx": float(df["minute_idx"].max() + minutes_to_predict),
            "open": float(last_row["close"]), # Assuming open for next candle is last close
            "high": float(last_row["close"] * 1.005), # Simple estimation for future high/low/volume
            "low": float(last_row["close"] * 0.995),
            "volume": float(last_row["volume"]),
            # Existing TA features - ensuring they are scalar floats
            "ma_10": float(last_row["ma_10"]),
            "ma_20": float(last_row["ma_20"]),
            "ema_12": float(last_row["ema_12"]),
            "ema_26": float(last_row["ema_26"]),
            "rsi": float(last_row["rsi"]),
            "macd": float(last_row["macd"]),
            "bollinger_hband": float(last_row["bollinger_hband"]),
            "bollinger_lband": float(last_row["bollinger_lband"]),
            "atr": float(last_row["atr"]),
            "adx": float(last_row["adx"]),
            "plus_di": float(last_row["plus_di"]),
            "minus_di": float(last_row["minus_di"]),
            "obv": float(last_row["obv"]),
            # Lagged features (use existing lagged values as a proxy for future lags) - ensuring scalar floats
            "close_lag_1": float(last_row["close"]),
            "close_lag_2": float(last_row["close_lag_1"]) if 'close_lag_1' in last_row else float(last_row["close"]),
            "close_lag_3": float(last_row["close_lag_2"]) if 'close_lag_2' in last_row else float(last_row["close"]),
            "close_lag_5": float(last_row["close_lag_4"]) if 'close_lag_4' in last_row else float(last_row["close"]),
            "close_lag_10": float(last_row["close_lag_9"]) if 'close_lag_9' in last_row else float(last_row["close"]),
            "close_lag_15": float(last_row["close_lag_14"]) if 'close_lag_14' in last_row else float(last_row["close"]),
            "volume_lag_1": float(last_row["volume"]),
            "volume_lag_2": float(last_row["volume_lag_1"]) if 'volume_lag_1' in last_row else float(last_row["volume"]),
            "volume_lag_3": float(last_row["volume_lag_2"]) if 'volume_lag_2' in last_row else float(last_row["volume"]),
            "volume_lag_5": float(last_row["volume_lag_4"]) if 'volume_lag_4' in last_row else float(last_row["volume"]),
            "volume_lag_10": float(last_row["volume_lag_9"]) if 'volume_lag_9' in last_row else float(last_row["volume"]),
            "volume_lag_15": float(last_row["volume_lag_14"]) if 'volume_lag_14' in last_row else float(last_row["volume"]),
            "rsi_lag_1": float(last_row["rsi"]),
            "rsi_lag_2": float(last_row["rsi_lag_1"]) if 'rsi_lag_1' in last_row else float(last_row["rsi"]),
            "rsi_lag_3": float(last_row["rsi_lag_2"]) if 'rsi_lag_2' in last_row else float(last_row["rsi"]),
            "rsi_lag_5": float(last_row["rsi_lag_4"]) if 'rsi_lag_4' in last_row else float(last_row["rsi"]),
            "rsi_lag_10": float(last_row["rsi_lag_9"]) if 'rsi_lag_9' in last_row else float(last_row["rsi"]),
            "rsi_lag_15": float(last_row["rsi_lag_14"]) if 'rsi_lag_14' in last_row else float(last_row["rsi"]),
            # Time-based features for the prediction time
            "hour": float(user_time.hour),
            "day_of_week": float(user_time.weekday())
        }
        
        # Create DataFrame from the dictionary. pd.DataFrame.from_dict() with orient='index' might be an alternative
        # but for a single row, passing a list of dicts or a dict of lists is common.
        # The key is to ensure the values within the dict are scalars, not lists.
        input_features = pd.DataFrame([input_features_dict])[features] # Now this will create a DataFrame with scalar cells.

        # Make prediction
        prediction = model.predict(scaler.transform(input_features))[0]
        latest_price = df["close"].iloc[-1]
        delta = prediction - latest_price

        st.markdown(f"<p class='big-font'>🚀 Predicted Price: ${prediction:,.2f} at {user_time.strftime('%H:%M')} IST</p>", unsafe_allow_html=True)
        st.info(f"📊 Expected Change: ${delta:.2f} ({'Increase 📈' if delta > 0 else 'Decrease 📉'})")

        # --- Confidence Interval (Simplified Volatility-Based) ---
        if confidence_enabled and not prediction_errors.empty:
            st.subheader("🔎 Prediction Range (Confidence)")
            # Calculate standard deviation of historical prediction errors (residuals)
            mae_std = prediction_errors.std()
            # A simple multiplier for the confidence interval, can be tuned
            confidence_multiplier = 2.0 # Roughly 2 standard deviations for ~95% if errors are normal
            lower_bound = prediction - (mae_std * confidence_multiplier)
            upper_bound = prediction + (mae_std * confidence_multiplier)
            st.write(f"**Predicted Range:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")
            st.warning("Note: This is a simplified range based on historical error volatility, not a rigorous statistical confidence interval.")
        else:
            st.warning("Cannot calculate confidence interval: Not enough test data for error analysis.")

        if show_sentiment:
            st.subheader("🗣️ Market Sentiment (Mocked)")
            st.markdown("📈 Sentiment Score: **+0.76** → Optimistic")
            st.progress(76)

    # --- Probo-Specific Input & Output ---
    st.markdown("---")
    st.subheader("🎯 Probo Question Alignment")
    probo_target_price_str = st.text_input("Probo Target Price (e.g., 20000.00)", value="", key="probo_price")
    probo_target_time_str = st.text_input("Probo Target Time (HH:MM IST, e.g., 22:30)", value="", key="probo_time")

    if st.button("Get Probo Prediction"):
        # Strip whitespace from inputs
        stripped_price_str = probo_target_price_str.strip()
        stripped_time_str = probo_target_time_str.strip()

        if stripped_price_str and stripped_time_str:
            try:
                probo_target_price = float(stripped_price_str)
                probo_target_time = datetime.strptime(stripped_time_str, "%H:%M").replace(
                    year=now_local.year, month=now_local.month, day=now_local.day, tzinfo=LOCAL_TZ
                )
                if probo_target_time < now_local:
                    probo_target_time += timedelta(days=1) # Assume next day if time already passed

                # Calculate minutes_to_predict for Probo's time
                probo_minutes_to_predict = int((probo_target_time - now_local).total_seconds() // 60)

                if 1 <= probo_minutes_to_predict <= 240: # Still limit to your model's range
                    # Recalculate prediction for Probo's specific time using the same logic as above
                    last_row = df.iloc[-1]
                    # Explicitly cast all values to float, ensuring they are scalar
                    probo_input_features_dict = {
                        "minute_idx": float(df["minute_idx"].max() + probo_minutes_to_predict),
                        "open": float(last_row["close"]),
                        "high": float(last_row["close"] * 1.005),
                        "low": float(last_row["close"] * 0.995),
                        "volume": float(last_row["volume"]),
                        "ma_10": float(last_row["ma_10"]),
                        "ma_20": float(last_row["ma_20"]),
                        "ema_12": float(last_row["ema_12"]),
                        "ema_26": float(last_row["ema_26"]),
                        "rsi": float(last_row["rsi"]),
                        "macd": float(last_row["macd"]),
                        "bollinger_hband": float(last_row["bollinger_hband"]),
                        "bollinger_lband": float(last_row["bollinger_lband"]),
                        "atr": float(last_row["atr"]),
                        "adx": float(last_row["adx"]),
                        "plus_di": float(last_row["plus_di"]),
                        "minus_di": float(last_row["minus_di"]),
                        "obv": float(last_row["obv"]),
                        "close_lag_1": float(last_row["close"]),
                        "close_lag_2": float(last_row["close_lag_1"]) if 'close_lag_1' in last_row else float(last_row["close"]),
                        "close_lag_3": float(last_row["close_lag_2"]) if 'close_lag_2' in last_row else float(last_row["close"]),
                        "close_lag_5": float(last_row["close_lag_4"]) if 'close_lag_4' in last_row else float(last_row["close"]),
                        "close_lag_10": float(last_row["close_lag_9"]) if 'close_lag_9' in last_row else float(last_row["close"]),
                        "close_lag_15": float(last_row["close_lag_14"]) if 'close_lag_14' in last_row else float(last_row["close"]),
                        "volume_lag_1": float(last_row["volume"]),
                        "volume_lag_2": float(last_row["volume_lag_1"]) if 'volume_lag_1' in last_row else float(last_row["volume"]),
                        "volume_lag_3": float(last_row["volume_lag_2"]) if 'volume_lag_2' in last_row else float(last_row["volume"]),
                        "volume_lag_5": float(last_row["volume_lag_4"]) if 'volume_lag_4' in last_row else float(last_row["volume"]),
                        "volume_lag_10": float(last_row["volume_lag_9"]) if 'volume_lag_9' in last_row else float(last_row["volume"]),
                        "volume_lag_15": float(last_row["volume_lag_14"]) if 'volume_lag_14' in last_row else float(last_row["volume"]),
                        "rsi_lag_1": float(last_row["rsi"]),
                        "rsi_lag_2": float(last_row["rsi_lag_1"]) if 'rsi_lag_1' in last_row else float(last_row["rsi"]),
                        "rsi_lag_3": float(last_row["rsi_lag_2"]) if 'rsi_lag_2' in last_row else float(last_row["rsi"]),
                        "rsi_lag_5": float(last_row["rsi_lag_4"]) if 'rsi_lag_4' in last_row else float(last_row["rsi"]),
                        "rsi_lag_10": float(last_row["rsi_lag_9"]) if 'rsi_lag_9' in last_row else float(last_row["rsi"]),
                        "rsi_lag_15": float(last_row["rsi_lag_14"]) if 'rsi_lag_14' in last_row else float(last_row["rsi"]),
                        "hour": float(probo_target_time.hour),
                        "day_of_week": float(probo_target_time.weekday())
                    }
                    probo_input_features = pd.DataFrame([probo_input_features_dict])[features]
                    probo_prediction = model.predict(scaler.transform(probo_input_features))[0]

                    st.markdown(f"<p class='big-font'>Probo Prediction for {probo_target_time.strftime('%H:%M')} IST:</p>", unsafe_allow_html=True)
                    st.write(f"**Your Predicted Price:** ${probo_prediction:,.2f}")
                    st.write(f"**Probo Target Price:** ${probo_target_price:,.2f}")

                    probo_predicted_outcome = ""
                    if probo_prediction > probo_target_price:
                        probo_predicted_outcome = "YES"
                        st.success(f"**Conclusion:** Probo Says: **{probo_predicted_outcome}** (Predicted price is HIGHER than target)")
                    else:
                        probo_predicted_outcome = "NO"
                        st.error(f"**Conclusion:** Probo Says: **{probo_predicted_outcome}** (Predicted price is LOWER than or equal to target)")

                    # Use confidence for nuanced advice if enabled
                    if confidence_enabled and not prediction_errors.empty:
                        mae_std = prediction_errors.std()
                        confidence_multiplier = 2.0
                        lower_bound = probo_prediction - (mae_std * confidence_multiplier)
                        upper_bound = probo_prediction + (mae_std * confidence_multiplier)

                        if probo_target_price < lower_bound:
                            st.info("High Confidence 'YES' (Target is below your entire predicted range)")
                        elif probo_target_price > upper_bound:
                            st.info("High Confidence 'NO' (Target is above your entire predicted range)")
                        else:
                            st.warning("Moderate Confidence / Neutral (Target falls within your predicted range)")

                else:
                    st.warning("Probo prediction limited to 1-240 minutes for now. Adjust time.")
            except ValueError as e:
                st.error(f"Invalid Probo Target Price or Time format: {e}. Please ensure price is a valid number (e.g., 20000.00) and time is HH:MM (e.g., 22:30).")
        else:
            st.warning("Please enter both Probo Target Price and Target Time.")


# --- Tab 3: Accuracy ---
with tabs[2]:
    st.header("📁 Model Evaluation")
    
    # Recalculate MAE and R2 for clarity, using the time-split test data
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Naive MAE (predicting next value is current value)
    naive_mae = mean_absolute_error(y_test.iloc[1:], y_test.shift(1).iloc[1:])
    
    # Cross-validation MAE
    # Note: Cross-validation on time series requires TimeSeriesSplit for proper evaluation
    # For simplicity, using standard cross_val_score here, but be aware of its limitations
    cv_scores = cross_val_score(model, X_scaled, y, scoring='neg_mean_absolute_error', cv=5)
    
    # Directional Accuracy
    y_test_direction = (y_test.diff().dropna() > 0).astype(int)
    y_pred_direction = (pd.Series(y_pred, index=y_test.index).diff().dropna() > 0).astype(int)
    directional_accuracy = (y_test_direction == y_pred_direction).mean()

    st.write(f"📉 MAE: {mae:.4f}")
    st.write(f"📈 R² Score: {r2:.4f}")
    st.write(f"⚖️ Naive MAE (Predicting previous price): {naive_mae:.4f}")
    st.write(f"🧪 Cross-Val MAE (Approximation): {-np.mean(cv_scores):.4f}")
    st.write(f"🎯 Directional Accuracy: {directional_accuracy:.2%}")
    
    # Chart for Actual vs Predicted
    plot_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.line_chart(plot_df)

# --- Tab 4: Chart ---
with tabs[3]:
    st.header(f"🔦 {selected_crypto} - Candlestick Chart (Last {data_days_limit} Day(s))")
    if not df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=df["timestamp"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color='lime', decreasing_line_color='red')])
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to display chart.")

# --- Tab 5: Forecast ---
with tabs[4]:
    st.header("🧙 Multi-Timeframe Forecast (Simplified)")
    if show_multiple_forecast and prediction is not None: # Only show if a prediction has been made
        times = [10, 30, 60, 120, 240] # Forecast for these minutes ahead
        multi_preds = []
        
        # We need the last N closing prices to calculate rolling indicators for future steps
        # Use a list to simulate rolling window for MA, RSI, MACD etc.
        # This is a simplification; a full simulation would re-calculate all features based on predicted prices
        
        # Start with the current price and recent history for indicator calculation
        recent_closes = list(df["close"].tail(max(15, 26)).values) # Needs enough for EMA26, RSI window etc.

        for step in times:
            # Simulate the next minute's features for prediction
            # This is a very simplified simulation of future candle features
            current_time_for_step = now_local + timedelta(minutes=step)
            
            # Update recent_closes with the last predicted value for the rolling window effect
            if multi_preds: # If not the first prediction
                recent_closes.append(multi_preds[-1][1])
                if len(recent_closes) > max(15, 26): # Keep window size constant
                    recent_closes.pop(0)

            # Recalculate indicators based on the simulated recent_closes, ensuring scalar floats
            temp_series = pd.Series(recent_closes)
            # Ensure .iloc[-1] is safe and handle potential NaNs from rolling/ema/rsi functions returning empty series
            ma_10_val = float(temp_series.rolling(window=10).mean().iloc[-1]) if len(temp_series) >= 10 and not temp_series.rolling(window=10).mean().isnull().all() else float(recent_closes[-1])
            ma_20_val = float(temp_series.rolling(window=20).mean().iloc[-1]) if len(temp_series) >= 20 and not temp_series.rolling(window=20).mean().isnull().all() else float(recent_closes[-1])
            ema_12_val = float(ta.trend.ema_indicator(temp_series, window=12).iloc[-1]) if len(temp_series) >= 12 and not ta.trend.ema_indicator(temp_series, window=12).isnull().all() else float(recent_closes[-1])
            ema_26_val = float(ta.trend.ema_indicator(temp_series, window=26).iloc[-1]) if len(temp_series) >= 26 and not ta.trend.ema_indicator(temp_series, window=26).isnull().all() else float(recent_closes[-1])
            rsi_val = float(ta.momentum.RSIIndicator(temp_series).rsi().iloc[-1]) if len(temp_series) >= 14 and not ta.momentum.RSIIndicator(temp_series).rsi().isnull().all() else 50.0 # RSI needs 14 periods, default to 50 if not enough data
            macd_val = float(ta.trend.MACD(temp_series).macd().iloc[-1]) if len(temp_series) >= 26 and not ta.trend.MACD(temp_series).macd().isnull().all() else 0.0 # MACD needs 26 periods, default to 0.0
            
            # Simple placeholder for other complex indicators for future steps - ensuring scalar floats
            bollinger_hband_val = float(recent_closes[-1] * 1.01) # Placeholder
            bollinger_lband_val = float(recent_closes[-1] * 0.99) # Placeholder
            atr_val = float(last_row["atr"]) # Assume constant for simplicity
            adx_val = float(last_row["adx"]) # Assume constant
            plus_di_val = float(last_row["plus_di"]) # Assume constant
            minus_di_val = float(last_row["minus_di"]) # Assume constant
            obv_val = float(last_row["obv"]) # Assume constant

            # For lagged features, ensuring scalar floats
            simulated_input_features_dict = {
                "minute_idx": float(df["minute_idx"].max() + step),
                "open": float(recent_closes[-1]), # Use last known/predicted close as next open
                "high": float(recent_closes[-1] * 1.005),
                "low": float(recent_closes[-1] * 0.995),
                "volume": float(last_row["volume"]), # Simplified
                "ma_10": ma_10_val,
                "ma_20": ma_20_val,
                "ema_12": ema_12_val,
                "ema_26": ema_26_val,
                "rsi": rsi_val,
                "macd": macd_val,
                "bollinger_hband": bollinger_hband_val,
                "bollinger_lband": bollinger_lband_val,
                "atr": atr_val,
                "adx": adx_val,
                "plus_di": plus_di_val,
                "minus_di": minus_di_val,
                "obv": obv_val,
                # Lagged features - ensuring scalar floats
                "close_lag_1": float(recent_closes[-1]),
                "close_lag_2": float(recent_closes[-2]) if len(recent_closes) >= 2 else float(recent_closes[-1]),
                "close_lag_3": float(recent_closes[-3]) if len(recent_closes) >= 3 else float(recent_closes[-1]),
                "close_lag_5": float(recent_closes[-5]) if len(recent_closes) >= 5 else float(recent_closes[-1]),
                "close_lag_10": float(recent_closes[-10]) if len(recent_closes) >= 10 else float(recent_closes[-1]),
                "close_lag_15": float(recent_closes[-15]) if len(recent_closes) >= 15 else float(recent_closes[-1]),
                "volume_lag_1": float(last_row["volume"]), # Placeholder for lagged volume
                "volume_lag_2": float(last_row["volume"]),
                "volume_lag_3": float(last_row["volume"]),
                "volume_lag_5": float(last_row["volume"]),
                "volume_lag_10": float(last_row["volume"]),
                "volume_lag_15": float(last_row["volume"]),
                "rsi_lag_1": float(recent_closes[-1]), # Simplification
                "rsi_lag_2": float(recent_closes[-2]) if len(recent_closes) >= 2 else float(recent_closes[-1]),
                "rsi_lag_3": float(recent_closes[-3]) if len(recent_closes) >= 3 else float(recent_closes[-1]),
                "rsi_lag_5": float(recent_closes[-5]) if len(recent_closes) >= 5 else float(recent_closes[-1]),
                "rsi_lag_10": float(recent_closes[-10]) if len(recent_closes) >= 10 else float(recent_closes[-1]),
                "rsi_lag_15": float(recent_closes[-15]) if len(recent_closes) >= 15 else float(recent_closes[-1]),
                "hour": float(current_time_for_step.hour),
                "day_of_week": float(current_time_for_step.weekday())
            }

            X_f = pd.DataFrame([simulated_input_features_dict])[features] # Ensure correct order
            X_scaled_f = scaler.transform(X_f)
            next_pred = model.predict(X_scaled_f)[0]
            multi_preds.append((f"{step}m", next_pred))

        forecast_df = pd.DataFrame(multi_preds, columns=["Minutes Ahead", "Predicted Price"])
        st.dataframe(forecast_df)
    elif show_multiple_forecast:
        st.info("Please make a single prediction first in the 'Predict' tab to enable multi-timeframe forecast.")


# --- Tab 6: Export ---
with tabs[5]:
    st.header("📆 Export Prediction")
    if prediction is not None and minutes_to_predict is not None:
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
    else:
        st.info("Make a prediction first to enable export.")

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using Python, Streamlit, Binance API & AI")
