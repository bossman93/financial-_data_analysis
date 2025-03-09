import pandas as pd
import numpy as np
import sqlite3
import requests
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import subprocess

# Configure logging
logging.basicConfig(filename="trade_logs.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Database setup
def init_db():
    conn = sqlite3.connect("crypto_analysis.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crypto TEXT,
        date TEXT,
        macd REAL,
        rsi REAL,
        bollinger_upper REAL,
        bollinger_lower REAL,
        stochastic_rsi REAL,
        market_condition INTEGER
    )""")
    conn.commit()
    conn.close()

# Fetch historical data from Binance US
def fetch_data(crypto):
    url = f"https://api.binance.us/api/v3/klines?symbol={crypto}USDT&interval=1d&limit=365"
    response = requests.get(url)
    if response.status_code != 200:
        logging.warning(f"⚠️ Warning: No data retrieved for {crypto}")
        return pd.DataFrame()

    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "Open", "High", "Low", "Close", "Volume", "CloseTime", "QuoteVolume",
                                     "Trades", "TakerBase", "TakerQuote", "Ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["Close", "High", "Low", "Open", "Volume"]].astype(float)
    return df

# Calculate technical indicators
def calculate_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = df['SMA20'] + (rolling_std * 2)
    df['Bollinger_Lower'] = df['SMA20'] - (rolling_std * 2)

    df['Lowest_RSI'] = df['RSI'].rolling(window=14).min()
    df['Highest_RSI'] = df['RSI'].rolling(window=14).max()
    df['Stochastic_RSI'] = np.where(
        df['Highest_RSI'] - df['Lowest_RSI'] == 0,
        0,
        (df['RSI'] - df['Lowest_RSI']) / (df['Highest_RSI'] - df['Lowest_RSI'])
    )

    df = df.bfill()

    return df

# Store results with clear type handling and logging
def store_results(crypto, df):
    conn = sqlite3.connect("crypto_analysis.db")
    cursor = conn.cursor()

    df['price_change'] = df['Close'].pct_change().shift(-1) * 100

    def get_market_condition(change):
        return 2 if change > 1 else 0 if change < -1 else 1

    try:
        for index, row in df.iterrows():
            cursor.execute("""
                INSERT INTO indicators 
                (crypto, date, macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, market_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    crypto,
                    index.strftime('%Y-%m-%d'),
                    float(row['MACD']),
                    float(row['RSI']),
                    float(row['Bollinger_Upper']),
                    float(row['Bollinger_Lower']),
                    float(row['Stochastic_RSI']),
                    get_market_condition(row['price_change'])
                ))
        conn.commit()
    except Exception as e:
        logging.error(f"Database insertion error for {crypto}: {e}")
    finally:
        conn.close()


# Train and evaluate XGBoost model clearly with indicator importance
def train_xgboost():
    conn = sqlite3.connect("crypto_analysis.db")
    df = pd.read_sql("""
        SELECT macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, market_condition
        FROM indicators
        WHERE market_condition IS NOT NULL
    """, conn)
    conn.close()

    if df.empty:
        logging.error("No data available for training.")
        return

    # Ensure data types are numeric and remove problematic rows
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    X = df[['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi']]
    y = df['market_condition'].astype(int)

    if X.empty or y.empty:
        logging.error("Dataset is empty after cleaning. Check data insertion.")
        print("Error: Dataset is empty after cleanup.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    logging.info(f"Model accuracy: {accuracy:.2%}")
    print(f"Accuracy: {accuracy:.2%}")

    report = classification_report(y_test, predictions, target_names=['Bearish', 'Neutral', 'Bullish'])
    print(report)
    logging.info(report)

    feature_importance = model.feature_importances_
    for feature, importance in zip(X.columns, feature_importance):
        logging.info(f"Indicator: {feature}, Importance: {importance:.4f}")

    conn.close()

# Commit hash for reproducibility
def get_commit_hash():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception as e:
        commit_hash = "Unavailable"
    logging.info(f"Commit Hash: {commit_hash}")
    return commit_hash

# Main Execution
if __name__ == "__main__":
    init_db()
    cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE', 'TRX', 'LINK', 'HBAR']

    for crypto in cryptos:
        df = fetch_data(crypto)
        if not df.empty:
            df = calculate_indicators(df)
            store_results(crypto, df)
        else:
            logging.warning(f"Skipping {crypto} due to missing data.")

    train_xgboost()
    print(f"Commit hash: {get_commit_hash()}")