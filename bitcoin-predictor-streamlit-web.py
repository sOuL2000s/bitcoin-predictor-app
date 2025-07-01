import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

# --- Configuration ---
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
TIMEZONES = pytz.all_timezones
DEFAULT_TIMEZONE = "Asia/Kolkata"

# --- Helper Functions ---

@st.cache_data(ttl=60)
def fetch_binance_data(symbol, interval, limit=500):
    """
    Fetches cryptocurrency historical kline data from Binance API.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(BINANCE_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ])
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
        df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from Binance: {e}")
        return pd.DataFrame()

def create_features(df):
    """
    Creates new features from the raw cryptocurrency data for improved prediction accuracy.
    """
    df_copy = df.copy()

    df_copy['SMA_7'] = df_copy['Close'].rolling(window=7).mean()
    df_copy['SMA_14'] = df_copy['Close'].rolling(window=14).mean()
    df_copy['SMA_21'] = df_copy['Close'].rolling(window=21).mean()

    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = np.where(loss == 0, np.inf, gain / loss)
    df_copy['RSI'] = 100 - (100 / (1 + rs))

    df_copy['Prev_Close'] = df_copy['Close'].shift(1)
    df_copy['Prev_High'] = df_copy['High'].shift(1)
    df_copy['Prev_Low'] = df_copy['Low'].shift(1)
    df_copy['Prev_Volume'] = df_copy['Volume'].shift(1)

    df_copy['Price_Change'] = df_copy['Close'].diff()

    df_copy.dropna(inplace=True)
    return df_copy

def train_and_predict(df):
    """
    Trains XGBoost and Linear Regression models, evaluates them, and selects the best one.

    Returns:
        tuple: A tuple containing:
            - best_future_pred (float): The predicted future price from the best model.
            - best_model_name (str): Name of the best model ('XGBoost' or 'Linear Regression').
            - xgb_predictions (np.array): XGBoost predictions on the test set.
            - lr_predictions (np.array): Linear Regression predictions on the test set.
            - y_test_values (np.array): Actual 'Close' prices from the test set.
            - xgb_metrics (dict): Dictionary of XGBoost metrics.
            - lr_metrics (dict): Dictionary of Linear Regression metrics.
    """
    df_features = create_features(df)

    if df_features.empty:
        st.warning("Not enough data after feature engineering to train models.")
        return 0.0, "N/A", np.array([]), np.array([]), np.array([]), {}, {}

    features = [col for col in df_features.columns if col not in [
        "Open time", "Close time", "Close", "Quote asset volume",
        "Number of trades", "Taker buy base asset volume",
        "Taker buy quote asset volume", "Ignore"
    ]]
    X = df_features[features]
    y = df_features['Close']

    tscv = TimeSeriesSplit(n_splits=5)
    train_index, test_index = list(tscv.split(X))[-1]

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    xgb_metrics = {}
    lr_metrics = {}
    xgb_predictions = np.array([])
    lr_predictions = np.array([])
    y_test_values = y_test.values if not y_test.empty else np.array([])

    # Initialize models
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    lr_model = LinearRegression()

    best_future_pred = 0.0
    best_model_name = "N/A"

    # Handle case where test set is empty (e.g., very small data_limit)
    if X_test.empty or y_test.empty:
        st.info("Not enough data to create a test set for evaluation. Training on all available data for prediction.")
        # Train on all available data if no test set for robust comparison
        xgb_model.fit(X, y)
        lr_model.fit(X, y)
        last_data_point = df_features.iloc[[-1]]
        future_X = last_data_point[features]
        xgb_future_pred = xgb_model.predict(future_X)[0]
        lr_future_pred = lr_model.predict(future_X)[0]

        # In this scenario, we can't definitively say which is "best" based on test metrics.
        # We'll just return both for now, or you could set a default.
        # For simplicity, we'll default to XGBoost as it's generally more robust.
        best_future_pred = xgb_future_pred
        best_model_name = "XGBoost (Default - No Test Set)"
        return best_future_pred, best_model_name, np.array([]), np.array([]), np.array([]), {}, {}


    # --- XGBoost Model ---
    st.subheader("XGBoost Model Performance")
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    mae_xgb = mean_absolute_error(y_test, xgb_predictions)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_predictions))
    r2_xgb = r2_score(y_test, xgb_predictions)
    xgb_metrics = {"MAE": mae_xgb, "RMSE": rmse_xgb, "R2": r2_xgb}

    st.write(f"XGBoost MAE: {mae_xgb:.4f}")
    st.write(f"XGBoost RMSE: {rmse_xgb:.4f}")
    st.write(f"XGBoost R2 Score: {r2_xgb:.4f}")

    # --- Linear Regression Model ---
    st.subheader("Linear Regression Model Performance")
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    mae_lr = mean_absolute_error(y_test, lr_predictions)
    rmse_lr = np.sqrt(mean_squared_error(y_test, lr_predictions))
    r2_lr = r2_score(y_test, lr_predictions)
    lr_metrics = {"MAE": mae_lr, "RMSE": rmse_lr, "R2": r2_lr}

    st.write(f"Linear Regression MAE: {mae_lr:.4f}")
    st.write(f"Linear Regression RMSE: {rmse_lr:.4f}")
    st.write(f"Linear Regression R2 Score: {r2_lr:.4f}")

    # --- Select Best Model for Future Prediction ---
    last_data_point = df_features.iloc[[-1]]
    future_X = last_data_point[features]

    if r2_xgb >= r2_lr:
        best_model_name = "XGBoost"
        best_future_pred = xgb_model.predict(future_X)[0]
        st.success(f"**Selected Best Model for Future Prediction: {best_model_name} (Higher R2 Score)**")
    else:
        best_model_name = "Linear Regression"
        best_future_pred = lr_model.predict(future_X)[0]
        st.success(f"**Selected Best Model for Future Prediction: {best_model_name} (Higher R2 Score)**")

    return best_future_pred, best_model_name, xgb_predictions, lr_predictions, y_test_values, xgb_metrics, lr_metrics

# --- Streamlit Application Main Function ---
def main():
    st.set_page_config(layout="wide", page_title="Crypto Price Predictor")
    st.title("Cryptocurrency Price Prediction Dashboard")

    # --- Sidebar for User Inputs ---
    st.sidebar.header("Configuration")
    crypto_symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"], index=0)
    interval = st.sidebar.selectbox("Select Interval", ["1h", "4h", "1d"], index=0)
    data_limit = st.sidebar.slider("Data Limit (Candles)", 100, 1000, 500)
    dark_mode = st.sidebar.checkbox("Dark Mode", True)
    auto_refresh = st.sidebar.checkbox("Auto Refresh Data", False)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60) if auto_refresh else None

    # --- Probo-specific Section (for target price and time) ---
    st.sidebar.header("Probo Prediction")
    target_price = st.sidebar.number_input("Target Price (Probo)", value=0.0, format="%.2f", help="Enter the target price for your prediction.")
    target_time_str = st.sidebar.text_input("Target Time (HH:MM) (Probo)", value="17:00", help="Enter time in HH:MM format (e.g., 17:00).")
    target_date_str = st.sidebar.date_input("Target Date (Probo)", value=datetime.now().date(), help="Select the target date.")
    selected_timezone = st.sidebar.selectbox("Select Timezone", TIMEZONES, index=TIMEZONES.index(DEFAULT_TIMEZONE), help="Select your local timezone.")

    # Apply dark mode styling if selected
    if dark_mode:
        st.markdown("""
            <style>
            .stApp {
                background-color: #1a1a1a;
                color: white;
            }
            .stTextInput>div>div>input {
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

    if auto_refresh:
        st.info(f"Auto-refresh enabled. Data will refresh every {refresh_interval} seconds.")
        st.rerun()

    st.subheader(f"Live Price for {crypto_symbol}")

    df = fetch_binance_data(crypto_symbol, interval, data_limit)

    if not df.empty:
        current_price = df["Close"].iloc[-1]
        st.metric(label=f"Current {crypto_symbol} Price", value=f"${current_price:,.2f}")

        st.subheader("Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=df['Open time'],
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'])])
        fig.update_layout(xaxis_rangeslider_visible=False,
                          template="plotly_dark" if dark_mode else "plotly_white",
                          title=f"{crypto_symbol} Candlestick Chart ({interval})")
        st.plotly_chart(fig, use_container_width=True)

        # Train models, evaluate, and get the best prediction
        best_future_pred, best_model_name, xgb_predictions, lr_predictions, y_test_values, xgb_metrics, lr_metrics = train_and_predict(df)

        st.subheader("Future Price Prediction (from Best Model)")
        st.write(f"Predicted Future Price by {best_model_name}: ${best_future_pred:,.2f}")

        # Plot actual vs predicted for the test set (showing both for comparison)
        if len(y_test_values) > 0:
            st.subheader("Model Performance: Actual vs Predicted (Test Set)")
            performance_df = pd.DataFrame({
                'Actual': y_test_values,
                'XGBoost Predicted': xgb_predictions,
                'Linear Regression Predicted': lr_predictions
            }, index=df['Open time'].iloc[-len(y_test_values):])

            st.line_chart(performance_df)
        else:
            st.info("Not enough data to display test set performance chart.")

        # Probo Prediction Logic
        if target_price > 0 and target_time_str and target_date_str:
            try:
                target_datetime_naive = datetime.combine(target_date_str, datetime.strptime(target_time_str, "%H:%M").time())
                local_tz = pytz.timezone(selected_timezone)
                target_datetime_local = local_tz.localize(target_datetime_naive)

                st.subheader("Probo Target Prediction Analysis")
                st.write(f"Your Target Price: ${target_price:,.2f}")
                st.write(f"Your Target Time (in {selected_timezone}): {target_datetime_local.strftime('%Y-%m-%d %H:%M %Z%z')}")

                st.markdown("---")
                st.markdown(f"**{best_model_name} Prediction vs. Target:**")
                if best_future_pred >= target_price:
                    st.success(f"Based on {best_model_name}, the price is predicted to reach or exceed your target of ${target_price:,.2f} (predicted: ${best_future_pred:,.2f}).")
                else:
                    st.warning(f"Based on {best_model_name}, the price is predicted to be below your target of ${target_price:,.2f} (predicted: ${best_future_pred:,.2f}).")

                st.info("Note: These are model predictions and do not guarantee future outcomes. Cryptocurrency markets are highly volatile.")

            except ValueError:
                st.error("Invalid time format. Please use HH:MM (e.g., 17:00).")
            except Exception as e:
                st.error(f"An error occurred during Probo prediction analysis: {e}")
        else:
            st.info("Enter a Target Price and Target Time in the sidebar to get Probo Prediction analysis.")

        st.subheader("Data Export")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name=f"{crypto_symbol}_data.csv",
            mime="text/csv",
        )
    else:
        st.error("Could not fetch data for the selected cryptocurrency. Please check your internet connection or try a different symbol/interval.")

if __name__ == "__main__":
    main()
