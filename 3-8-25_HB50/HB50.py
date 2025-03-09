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

# Configure logging
logging.basicConfig(filename="trade_logs.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

db_path = os.path.abspath("crypto_analysis.db")

def fetch_data(crypto):
    url = f"https://api.binance.us/api/v3/klines?symbol={crypto}USDT&interval=1d&limit=365"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"⚠️ No data retrieved for {crypto}")
        return pd.DataFrame()
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["Close", "High", "Low", "Open", "Volume"]].astype(float)
    return df

def calculate_indicators(df):
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA20'] + (rolling_std * 2)
    df['Bollinger_Lower'] = df['SMA20'] - (rolling_std * 2)
    df['Stochastic_RSI'] = (df['RSI'] - df['RSI'].rolling(14).min()) / (df['RSI'].rolling(14).max() - df['RSI'].rolling(14).min())
    df['price_change'] = df['Close'].pct_change().shift(-1) * 100
    df.dropna(inplace=True)
    return df

def store_results(crypto, df):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for index, row in df.iterrows():
        market_condition = 2 if row['price_change'] > 1 else 0 if row['price_change'] < -1 else 1
        cursor.execute("""
        INSERT INTO indicators (crypto, date, macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, market_condition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (crypto, index.strftime('%Y-%m-%d'), row['MACD'], row['RSI'], row['Bollinger_Upper'], row['Bollinger_Lower'], row['Stochastic_RSI'], market_condition))
    conn.commit()
    conn.close()

def train_xgboost():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM indicators", conn)
    conn.close()
    X = df[['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi']]
    y = df['market_condition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}\n")
    print(report)
    return accuracy, report

def generate_commit_hash():
    with open(db_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    with open("trade_logs.txt", "a") as log_file:
        log_file.write(f"\nCommit Hash: {file_hash}\n")
    print(f"✅ Commit Hash Generated: {file_hash}")

if __name__ == "__main__":
    cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE', 'TRX', 'LINK', 'HBAR']
    for crypto in cryptos:
        df = fetch_data(crypto)
        if not df.empty:
            df = calculate_indicators(df)
            store_results(crypto, df)
    accuracy, report = train_xgboost()
    generate_commit_hash()
