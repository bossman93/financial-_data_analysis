import requests
import pandas as pd
import numpy as np
import time
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import schedule
import os
import logging

# Configure logging to save forecast logs to a file
logging.basicConfig(filename="forecast_log.txt", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Top 10 cryptos (excluding stablecoins & staked ETH)
TOP_10_CRYPTOS = ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "DOGE", "TRX", "LINK", "HBAR"]

# Binance US API URLs
BINANCE_KLINES_URL = "https://api.binance.us/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.us/api/v3/ticker/price"

# File paths
MODEL_PATH = "gru_crypto_model.keras"
DB_PATH = "crypto_forecasts.db"


def fetch_crypto_data_binance(symbol, interval="1d", limit=180):
    """Fetch historical daily price data from Binance US API with retry logic."""
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    response = requests.get(BINANCE_KLINES_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                         "quote_asset_vol", "num_trades", "taker_buy_base", "taker_buy_quote",
                                         "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df["close"] = df["close"].astype(float)
        return df[["timestamp", "close"]]
    logging.error(f"Error fetching data for {symbol}: {response.status_code}")
    return None


def fetch_latest_price_binance(symbol):
    """Fetch the latest price for a cryptocurrency from Binance US."""
    params = {"symbol": f"{symbol}USDT"}
    response = requests.get(BINANCE_TICKER_URL, params=params)
    if response.status_code == 200:
        return float(response.json()["price"])
    logging.error(f"Error fetching latest price for {symbol}: {response.status_code}")
    return None


def preprocess_data(df):
    """Scale data and prepare sequences for GRU model."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    df["close_scaled"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))
    X, y = [], []
    sequence_length = 30
    for i in range(len(df) - sequence_length):
        X.append(df["close_scaled"].values[i:i + sequence_length])
        y.append(df["close_scaled"].values[i + sequence_length])
    return np.array(X), np.array(y).reshape(-1, 1), scaler


def create_gru_model():
    """Create and compile a GRU model."""
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(30, 1)),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def save_prediction_to_db(crypto, predicted_price):
    """Save prediction results to SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        crypto TEXT NOT NULL,
        predicted_price REAL NOT NULL,
        actual_price REAL,
        error REAL
    )""")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO predictions (timestamp, crypto, predicted_price) VALUES (?, ?, ?)",
                   (timestamp, crypto, predicted_price))
    conn.commit()
    conn.close()


def generate_summary_report():
    """Generate a summary of the last 10 predictions with actual prices."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()

    actual_prices = {crypto: fetch_latest_price_binance(crypto) for crypto in TOP_10_CRYPTOS}
    df["actual_price"] = df["crypto"].map(actual_prices)
    df["error"] = df["actual_price"] - df["predicted_price"]

    # Calculate performance metrics (MAE, RMSE)
    valid_rows = df.dropna()
    if not valid_rows.empty:
        mae = mean_absolute_error(valid_rows["actual_price"], valid_rows["predicted_price"])
        rmse = np.sqrt(mean_squared_error(valid_rows["actual_price"], valid_rows["predicted_price"]))
        logging.info(f"Model Performance - MAE: {mae}, RMSE: {rmse}")

    logging.info("--- Crypto Prediction Summary ---")
    logging.info(df.to_string(index=False))
    print("\n--- Crypto Prediction Summary ---")
    print(df.to_string(index=False))


def plot_predictions(actual, predicted, crypto):
    """Plot actual vs. predicted prices for visualization."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price", color="blue")
    plt.plot(predicted, label="Predicted Price", color="red", linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title(f"{crypto.upper()} Price Prediction (GRU)")
    plt.legend()

    # Print the current working directory to confirm where the file is being saved
    current_dir = os.getcwd()
    print(f"Current Working Directory: {current_dir}")

    # Hardcoded path for saving chart
    chart_path = os.path.join(current_dir, f"{crypto}_forecast_chart.png")
    print(f"Saving chart to: {chart_path}")  # Print the full path to ensure it's correct

    # Save chart as PNG
    try:
        plt.savefig(chart_path)  # Ensure this line is here
        print(f"Chart saved successfully at: {chart_path}")
    except Exception as e:
        print(f"Error saving the chart: {str(e)}")
    plt.show()  # This will display the chart


def run_crypto_prediction():
    """Run cryptocurrency price prediction for the top cryptos."""
    model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else create_gru_model()
    for crypto in TOP_10_CRYPTOS:
        df = fetch_crypto_data_binance(crypto)
        if df is None or len(df) < 31:
            continue
        X, y, scaler = preprocess_data(df)
        last_30_days = X[-1].reshape(1, 30, 1)
        predicted_price_scaled = model.predict(last_30_days)[0][0]
        predicted_price = scaler.inverse_transform([[predicted_price_scaled]])[0][0]
        save_prediction_to_db(crypto, predicted_price)
        logging.info(f"Predicted {crypto}: ${predicted_price:.2f}")

        # Call the plot_predictions function for all cryptocurrencies
        plot_predictions(df["close"].values[-30:], [predicted_price] * 30, crypto)

    generate_summary_report()


# Schedule predictions and summary reports
schedule.every().day.at("00:00").do(run_crypto_prediction)
schedule.every().day.at("23:59").do(generate_summary_report)

# Run prediction immediately for initial execution
run_crypto_prediction()

print("Crypto prediction script scheduled to run daily at midnight.")

while True:
    schedule.run_pending()  # Keep checking for pending scheduled tasks
    time.sleep(60)  # Sleep for 60 seconds before checking again
