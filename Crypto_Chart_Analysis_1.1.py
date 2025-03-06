import pandas as pd
import numpy as np
import requests
import sqlite3
import talib
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from scipy.stats import linregress

# Binance.US API endpoint
BINANCE_US_BASE_URL = "https://api.binance.us/api/v3/"


# Fetch top 10 cryptocurrencies dynamically by market cap
def get_top_cryptos(limit=10):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching top cryptos: {response.text}")
        return []

    data = response.json()
    top_symbols = [entry["symbol"].upper() + "USDT" for entry in data if "symbol" in entry]
    print(f"Top {limit} Cryptos by Market Cap: {top_symbols}")
    return top_symbols


# Fetch historical data from Binance.US API
def get_binanceus_historical_data(symbol, interval="1d", limit=90):
    url = f"{BINANCE_US_BASE_URL}klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching data for {symbol}: {response.text}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                     "quote_asset_volume", "trades", "taker_base_volume", "taker_quote_volume",
                                     "ignore"])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df


# Compute indicators
def add_indicators(df):
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['close'], timeperiod=20)
    slowk, slowd = talib.STOCHRSI(df['close'], timeperiod=14)
    df['Stoch_RSI_K'] = slowk
    df['Stoch_RSI_D'] = slowd

    # Detect Moving Average Crossovers
    df['MA_Crossover'] = np.where((df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1)), 1,
                                  np.where(
                                      (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1)),
                                      -1, 0))
    return df


# Store data in SQLite database
def store_data(df, symbol):
    conn = sqlite3.connect("crypto_analysis.db")
    df.to_sql(symbol, conn, if_exists="replace", index=True)
    conn.close()


# Generate charts with trendlines and volume
def plot_crypto_analysis(df, symbol):
    if df.empty:
        print(f"Skipping {symbol}: No data available for plotting.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='dashed')
    plt.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='dashed')
    plt.fill_between(df.index, df['Lower_BB'], df['Upper_BB'], color='gray', alpha=0.2)

    # Volume Bar Chart
    plt.twinx()
    plt.bar(df.index, df['volume'], alpha=0.3, color='gray', label='Volume')

    # Trendline Overlay
    x_vals = np.arange(len(df))
    slope, intercept, _, _, _ = linregress(x_vals, df['close'])
    trendline = slope * x_vals + intercept
    plt.plot(df.index, trendline, linestyle='dotted', color='red', label='Trendline')

    plt.title(f"{symbol} Price Chart with Indicators")
    plt.legend()

    # Save the plot
    output_dir = "crypto_charts"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{symbol}_chart.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"Saved chart for {symbol} at: {file_path}")


# Main function to analyze cryptocurrencies
def analyze_cryptos():
    symbols = get_top_cryptos()
    if not symbols:
        print("No cryptocurrencies fetched. Exiting analysis.")
        return

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")
        df = get_binanceus_historical_data(symbol)
        if df.empty:
            print(f"Skipping {symbol}: No data available.")
            continue
        df = add_indicators(df)
        store_data(df, symbol)
        plot_crypto_analysis(df, symbol)
        print(f"Completed analysis for {symbol}. Waiting 5 seconds before next request...")
        time.sleep(5)


# Run analysis
analyze_cryptos()