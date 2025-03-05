import os
import json
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output folder exists
output_folder = "crypto_charts"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Function to fetch cryptocurrency data
def fetch_crypto_data(ticker, period="3mo", interval="1d"):
    crypto = yf.Ticker(ticker)
    df = crypto.history(period=period, interval=interval)
    df["ticker"] = ticker
    df.index = df.index.tz_localize(None)  # Remove timezone for database storage
    return df


# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data["Close"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    data["rsi"] = rsi
    return data


# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data["Close"].ewm(span=short_window, adjust=False).mean()
    long_ema = data["Close"].ewm(span=long_window, adjust=False).mean()
    data["macd"] = short_ema - long_ema
    data["signal_line"] = data["macd"].ewm(span=signal_window, adjust=False).mean()
    return data


# Function to generate and save charts
def generate_charts(data, ticker):
    plt.figure(figsize=(10, 5))

    # MACD Chart
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data["macd"], label="MACD", color="blue")
    plt.plot(data.index, data["signal_line"], label="Signal Line", color="red")
    plt.title(f"{ticker} - MACD & Signal Line")
    plt.legend()

    # RSI Chart
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data["rsi"], label="RSI", color="purple")
    plt.axhline(70, linestyle="--", color="red")  # Overbought
    plt.axhline(30, linestyle="--", color="green")  # Oversold
    plt.title(f"{ticker} - RSI")
    plt.legend()

    # Candlestick Chart (Closing Prices)
    plt.subplot(3, 1, 3)
    plt.plot(data.index, data["Close"], label="Closing Price", color="black")
    plt.title(f"{ticker} - Closing Price Candlestick Pattern")
    plt.legend()

    plt.tight_layout()

    # Save the chart
    chart_path = os.path.join(output_folder, f"{ticker}_charts.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"✅ Saved chart: {chart_path}")
    return chart_path


# Main function to analyze crypto and save charts
def analyze_crypto(tickers):
    chart_paths = []  # Store file paths
    for ticker in tickers:
        print(f"\n📈 Analyzing {ticker}...\n")

        df = fetch_crypto_data(ticker)
        df = calculate_rsi(df)
        df = calculate_macd(df)

        # Generate and save charts
        chart_path = generate_charts(df, ticker)
        chart_paths.append(chart_path)

    return chart_paths


# Define cryptocurrencies to analyze
crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Run analysis and save charts
chart_files = analyze_crypto(crypto_tickers)

# ✅ Save chart file paths
with open("chart_files.json", "w") as f:
    json.dump(chart_files, f)

print("✅ Chart file paths saved to chart_files.json")

