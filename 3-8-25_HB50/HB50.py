import pandas as pd
import numpy as np
import sqlite3
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import hashlib
import os
from datetime import datetime

# Configure logging
logging.basicConfig(filename="trade_logs.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

db_path = os.path.abspath("crypto_analysis.db")


def fetch_data(crypto):
    url = f"https://api.binance.us/api/v3/klines?symbol={crypto}USDT&interval=1d&limit=365"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"⚠️ No data for {crypto}")
        print(f"⚠️ No data for {crypto}")
        return pd.DataFrame()

    # Ensure correct column mapping
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"
    ])

    # Convert timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)

    # Select only required columns
    df = df[["Close", "High", "Low", "Open", "Volume"]].astype(float)

    return df


def calculate_indicators(df):
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() /
                              -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean()))
    df['Bollinger_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Bollinger_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()
    df['Stochastic_RSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (
                df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())
    df['price_change'] = df['Close'].pct_change().shift(-1) * 100
    return df.dropna()


def store_results(crypto, df):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for index, row in df.iterrows():
        market_condition = 2 if row['price_change'] > 1 else 0 if row['price_change'] < -1 else 1
        cursor.execute("""
        INSERT INTO indicators (crypto, date, macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, market_condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                       (crypto, index.strftime('%Y-%m-%d'), row['MACD'], row['RSI'], row['Bollinger_Upper'],
                        row['Bollinger_Lower'], row['Stochastic_RSI'], market_condition))
    conn.commit()
    conn.close()


def train_xgboost():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()
    X = df[['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi']]
    y = df['market_condition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    with open("trade_logs.txt", "a") as log_file:
        log_file.write("\n--- Cryptocurrency Performance Analysis ---\n")
        for crypto in sorted(df['crypto'].unique()):
            crypto_data = df[df['crypto'] == crypto]
            crypto_accuracy = accuracy_score(crypto_data['market_condition'], model.predict(
                crypto_data[['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi']]))
            summary = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {crypto}: Accuracy = {crypto_accuracy:.4f}"
            print(summary)
            log_file.write(summary + "\n")
        log_file.write("\nOverall Model Accuracy: {:.4f}\n".format(accuracy))

    print("\nOverall Model Accuracy: {:.4f}".format(accuracy))
    return accuracy


def generate_commit_hash():
    with open(db_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"✅ Commit Hash: {file_hash}")


if __name__ == "__main__":
    cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE', 'TRX', 'LINK', 'HBAR']
    for crypto in cryptos:
        df = fetch_data(crypto)
        if not df.empty:
            df = calculate_indicators(df)
            store_results(crypto, df)
    accuracy = train_xgboost()
    generate_commit_hash()
