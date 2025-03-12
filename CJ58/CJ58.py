import pandas as pd
import numpy as np
import sqlite3
import requests
import xgboost as xgb
import matplotlib.pyplot as plt
import mplfinance as mpf
import logging
import hashlib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# Configure logging
logging.basicConfig(filename="trade_logs.txt", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Database path
db_path = os.path.abspath("crypto_analysis.db")


def get_user_assets():
    """Prompt user for asset selection and validate input."""
    default_assets = ['BTC', 'ETH', 'XRP']
    user_input = input(
        f"Enter cryptocurrencies to analyze (comma-separated, or press Enter for default {default_assets}): ").strip()

    if not user_input:
        print(f"✅ Using default assets: {default_assets}")
        return default_assets

    assets = [asset.strip().upper() for asset in user_input.split(",")]
    print(f"✅ Selected assets: {assets}")
    return assets


def get_user_scenario():
    """Prompt user to select a market scenario."""
    scenarios = ['bullish', 'bearish', 'high_vol', 'low_liq', 'all']

    print("\nMarket Scenarios:")
    print("1 - Bullish (RSI > 60 & MACD Positive)")
    print("2 - Bearish (RSI < 40 & MACD Negative)")
    print("3 - High Volatility (High Std Dev in Close Prices)")
    print("4 - Low Liquidity (Low Trading Volume)")
    print("5 - All Market Scenarios")

    user_choice = input("Select a market scenario (1-5, default = 5): ").strip()

    if user_choice in ["1", "2", "3", "4"]:
        selected_scenario = scenarios[int(user_choice) - 1]
        print(f"✅ Selected market scenario: {selected_scenario}")
        return selected_scenario

    print("✅ Defaulting to all market scenarios.")
    return "all"


def fetch_data(crypto):
    """Fetch historical crypto data from Binance."""
    url = f"https://api.binance.us/api/v3/klines?symbol={crypto}USDT&interval=1d&limit=365"
    response = requests.get(url)

    if response.status_code != 200:
        logging.warning(f"⚠️ No data for {crypto}")
        print(f"⚠️ No data for {crypto}")
        return pd.DataFrame()  # Return empty DataFrame if API call fails

    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])

    # Ensure 'df' is created before any further processing
    if df.empty:
        print(f"⚠️ No data available for {crypto}")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    # Inspect the first few Close prices
    print(df[['Close']].head(10))  # Step 1: Check Close prices before calculating MACD

    return df


def calculate_indicators(df):
    """Calculate MACD, RSI, Bollinger Bands, and Stochastic RSI."""

    # Scale down Close prices for MACD calculation if necessary (optional)
    # df['Close'] = df['Close'] / 1000  # Uncomment this line to scale down Close prices

    # Calculate MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    # Calculate RSI
    gain = df['Close'].diff().where(df['Close'].diff() > 0, 0)
    loss = -df['Close'].diff().where(df['Close'].diff() < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['Bollinger_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Bollinger_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()

    # Calculate Stochastic RSI
    df['Stochastic_RSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (
            df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())

    # Calculate percentage change in Close prices for debugging (Step 1)
    df['pct_change'] = df['Close'].pct_change() * 100
    print(df[['Close', 'pct_change']].head(10))  # Print first 10 rows to check changes

    df['price_change'] = df['Close'].pct_change().shift(-1) * 100
    df.dropna(inplace=True)
    return df


def store_results(crypto, df):
    """Store computed indicators in SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crypto TEXT, date TEXT, macd REAL, rsi REAL, 
        bollinger_upper REAL, bollinger_lower REAL, stochastic_rsi REAL, 
        market_condition INTEGER
    )""")
    for index, row in df.iterrows():
        market_condition = 2 if row['price_change'] > 1 else 0 if row['price_change'] < -1 else 1
        cursor.execute("""
        INSERT INTO indicators (crypto, date, macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, market_condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                       (crypto, index.strftime('%Y-%m-%d'), row['MACD'], row['RSI'], row['Bollinger_Upper'],
                        row['Bollinger_Lower'], row['Stochastic_RSI'], market_condition))
    conn.commit()
    conn.close()


def generate_commit_hash():
    """Generate a commit hash for reproducibility."""
    try:
        with open(__file__, "rb") as f:
            content = f.read()
        commit_hash = hashlib.sha256(content).hexdigest()
        print(f"\n✅ Commit Hash: {commit_hash}")
        logging.info(f"Commit Hash: {commit_hash}")
    except Exception as e:
        print(f"⚠️ Error generating commit hash: {e}")
        logging.warning(f"Error generating commit hash: {e}")

def generate_market_summary(selected_cryptos, market_scenario):
    """Generate a filtered market summary report with actionable insights."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    df = df[df['crypto'].isin(selected_cryptos)]

    if df.empty:
        return "⚠️ No selected assets match the chosen market scenario."

    summary = []
    for crypto in df['crypto'].unique():
        latest = df[df['crypto'] == crypto].iloc[-1]
        trend = "Bullish" if latest['macd'] > 0 and latest['rsi'] > 60 else \
                "Bearish" if latest['macd'] < 0 and latest['rsi'] < 40 else "Neutral"

        recommendation = "📈 Consider Buying on Dips." if trend == "Bullish" else "📉 Potential Short Signal." if trend == "Bearish" else "⚖️ Wait for confirmation."
        summary.append(f"{crypto}: {trend} - RSI: {latest['rsi']:.2f}, MACD: {latest['macd']:.2f}. {recommendation}")

    return "\n".join(summary)

def backtest_trading_strategy():
    """Backtest and compare ML-assisted trading vs. indicator-only trading."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Filter data to include only selected assets
    selected_assets = ['BTC', 'ETH']  # Modify this list based on user input
    df = df[df['crypto'].isin(selected_assets)]  # Filter for selected assets

    results = []

    for crypto in df['crypto'].unique():
        crypto_df = df[df['crypto'] == crypto].copy()
        initial_balance = 10000  # Starting balance in USD
        balance_indicator = initial_balance
        balance_ml = initial_balance
        position = 0  # Number of assets held
        position_ml = 0

        false_signals = 0
        false_signals_ml = 0
        max_drawdown = 0
        max_drawdown_ml = 0
        peak_balance = initial_balance
        peak_balance_ml = initial_balance

        for i in range(len(crypto_df) - 1):
            current_price = crypto_df.iloc[i]['bollinger_upper']
            next_price = crypto_df.iloc[i + 1]['bollinger_upper']

            # Indicator-Only Strategy (MACD & RSI)
            if crypto_df.iloc[i]['rsi'] < 30 and crypto_df.iloc[i]['macd'] > 0 and position == 0:
                position = balance_indicator / current_price  # Buy
                balance_indicator = 0
            elif crypto_df.iloc[i]['rsi'] > 70 and crypto_df.iloc[i]['macd'] < 0 and position > 0:
                balance_indicator = position * current_price  # Sell
                position = 0

            # ML-Assisted Strategy
            if crypto_df.iloc[i]['market_condition'] == 2 and position_ml == 0:  # ML predicts uptrend
                position_ml = balance_ml / current_price  # Buy
                balance_ml = 0
            elif crypto_df.iloc[i]['market_condition'] == 0 and position_ml > 0:  # ML predicts downtrend
                balance_ml = position_ml * current_price  # Sell
                position_ml = 0

            # Track Drawdown
            if balance_indicator > peak_balance:
                peak_balance = balance_indicator
            drawdown = (peak_balance - balance_indicator) / max(peak_balance, 1)
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            if balance_ml > peak_balance_ml:
                peak_balance_ml = balance_ml
            drawdown_ml = (peak_balance_ml - balance_ml) / max(peak_balance_ml, 1)
            if drawdown_ml > max_drawdown_ml:
                max_drawdown_ml = drawdown_ml

            # False signals count
            if (crypto_df.iloc[i]['rsi'] < 30 and crypto_df.iloc[i]['macd'] > 0) and next_price < current_price:
                false_signals += 1
            if (crypto_df.iloc[i]['market_condition'] == 2) and next_price < current_price:
                false_signals_ml += 1

        # Final Balance Calculation
        if position > 0:
            balance_indicator = position * crypto_df.iloc[-1]['bollinger_upper']
        if position_ml > 0:
            balance_ml = position_ml * crypto_df.iloc[-1]['bollinger_upper']

        pnl_indicator = ((balance_indicator - initial_balance) / initial_balance) * 100
        pnl_ml = ((balance_ml - initial_balance) / initial_balance) * 100

        # Initialize false signal reduction
        false_signals_reduction = 0

        # Case 1: If both indicator and ML have false signals
        if false_signals > 0 and false_signals_ml > 0:
            false_signals_reduction = max(0, ((false_signals - false_signals_ml) / max(false_signals, 1)) * 100)

        # Case 2: If indicator has no false signals but ML has false signals
        elif false_signals == 0 and false_signals_ml > 0:
            false_signals_reduction = 0  # No reduction if ML performs worse

        # Case 3: If indicator has false signals but ML does not
        elif false_signals > 0 and false_signals_ml == 0:
            false_signals_reduction = 100  # Maximum reduction if only the indicator has false signals

        drawdown_reduction = 0
        if max_drawdown > 0:
            drawdown_reduction = ((max_drawdown - max_drawdown_ml) / max(max_drawdown, 1)) * 100

        # Print false signals for debugging
        print(f"False Signals (Indicator): {false_signals}")
        print(f"False Signals (ML): {false_signals_ml}")

        results.append({
            "Asset": crypto,
            "Indicator P&L (%)": round(pnl_indicator, 2),
            "ML P&L (%)": round(pnl_ml, 2),
            "False Signals Reduced (%)": round(false_signals_reduction, 2),
            "Drawdown Reduction (%)": round(drawdown_reduction, 2)
        })

    return results





if __name__ == "__main__":
    cryptos = get_user_assets()
    market_scenario = get_user_scenario()

    for crypto in cryptos:
        df = fetch_data(crypto)
        if not df.empty:
            df = calculate_indicators(df)
            store_results(crypto, df)

    print("\n📊 Market Summary Report:\n")
    print(generate_market_summary(cryptos, market_scenario))

    print("\n🔄 Running Backtest...")
    backtest_results = backtest_trading_strategy()

    print("\n📈 Backtest Results:\n")
    for result in backtest_results:
        print(f"{result['Asset']} | Indicator P&L: {result['Indicator P&L (%)']}% | "
              f"ML P&L: {result['ML P&L (%)']}% | "
              f"False Signal Reduction: {result['False Signals Reduced (%)']}% | "
              f"Drawdown Reduction: {result['Drawdown Reduction (%)']}%")

    generate_commit_hash()
