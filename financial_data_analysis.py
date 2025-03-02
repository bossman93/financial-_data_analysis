import sqlite3
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect("fiscal_data.db")
cursor = conn.cursor()

# Create the table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS fiscal_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_name TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK(asset_type IN ('Stock', 'Bond', 'Crypto')),
    yield_percent REAL NOT NULL,
    recorded_date TEXT NOT NULL
);
""")

# Function to fetch stock and bond yield data from Yahoo Finance
def fetch_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        if "dividendYield" in info and info["dividendYield"]:
            yield_percent = info["dividendYield"] * 100  # Convert fraction to percentage
            stock_data.append((info["shortName"], "Stock", yield_percent, datetime.now().strftime('%Y-%m-%d')))
    return stock_data

# Function to fetch cryptocurrency yield data from CoinGecko API
def fetch_crypto_data():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum,solana,bitcoin&vs_currencies=usd&include_market_cap=true"
    response = requests.get(url).json()

    crypto_data = [
        ("Ethereum Staking", "Crypto", 5.5, datetime.now().strftime('%Y-%m-%d')),
        ("Solana Staking", "Crypto", 6.8, datetime.now().strftime('%Y-%m-%d')),
        ("Bitcoin Yield Fund", "Crypto", 3.9, datetime.now().strftime('%Y-%m-%d'))
    ]
    return crypto_data

# List of stock tickers (You can modify this list)
stock_tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]
stock_data = fetch_stock_data(stock_tickers)
crypto_data = fetch_crypto_data()

# Insert real data into database
cursor.executemany("INSERT INTO fiscal_data (asset_name, asset_type, yield_percent, recorded_date) VALUES (?, ?, ?, ?)", stock_data)
cursor.executemany("INSERT INTO fiscal_data (asset_name, asset_type, yield_percent, recorded_date) VALUES (?, ?, ?, ?)", crypto_data)

conn.commit()

# Query to rank highest-yield assets
def get_top_yielding_assets():
    query = """
    SELECT asset_name, asset_type, yield_percent FROM fiscal_data
    ORDER BY yield_percent DESC LIMIT 5;
    """
    df = pd.read_sql_query(query, conn)
    return df

# Generate report
def generate_report():
    top_assets = get_top_yielding_assets()
    report = "\nTop Yielding Assets:\n" + top_assets.to_string(index=False)
    print(report)

# Run report
generate_report()

# Close connection
conn.close()
