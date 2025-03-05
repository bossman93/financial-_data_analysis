import sqlite3
import requests
import schedule
import time
import pandas as pd
import yfinance as yf
import json
from datetime import datetime
from bs4 import BeautifulSoup
import pytz


# Database connection
def connect_db():
    return sqlite3.connect("top100_sp500_fiscal_data.db", timeout=10)  # ✅ Added timeout
def initialize_db():
    """Ensure required tables exist and enable WAL mode."""
    with connect_db() as conn:
        cursor = conn.cursor()
        cursor.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT
        );
        CREATE TABLE IF NOT EXISTS summary_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT,
            fetched_stocks INTEGER,
            avg_crypto_price REAL
        );
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            price REAL,
            market_cap REAL,
            volume REAL,
            recorded_date TEXT,
            UNIQUE(asset_name, recorded_date)
        );
        """)
        conn.commit()
        log_message("INFO", "✅ WAL Mode enabled and database initialized successfully.")


# ✅ Use a persistent connection for logging
def log_message(level, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect("top100_sp500_fiscal_data.db", timeout=30)  # ✅ Increased timeout

    try:
        conn.execute("INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)", (timestamp, level, message))
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"⚠️ Log Error: {e}")  # Log the error but don't crash the program
    finally:
        conn.close()  # ✅ Ensure connection is properly closed

    print(f"[{timestamp}] {level}: {message}")



# Function to retry failed API requests
def fetch_with_retries(url, retries=3, wait=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            log_message("ERROR", f"Attempt {attempt + 1} failed: {e}")
            time.sleep(wait * (2 ** attempt))
    return None


# Function to store data in SQLite
def store_data(crypto_data, sp500_data):
    try:
        log_message("INFO", "Storing data in database...")

        # ✅ Use `with` to ensure connection is closed properly
        with sqlite3.connect("top100_sp500_fiscal_data.db") as conn:
            cursor = conn.cursor()
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # ✅ Ensure cryptos are stored correctly per pull
            if crypto_data:
                crypto_insert_query = """
                    INSERT INTO financial_data 
                    (asset_name, price, market_cap, volume, recorded_date)
                    VALUES (?, ?, ?, ?, ?)
                """
                records = [
                    (crypto["name"], crypto["current_price"], crypto["market_cap"], crypto["total_volume"], current_timestamp)
                    for crypto in crypto_data[:100]  # ✅ Store only top 100 cryptos
                ]
                cursor.executemany(crypto_insert_query, records)
                log_message("INFO", f"✅ {len(records)} cryptos stored successfully at {current_timestamp}.")

            # ✅ Ensure stocks are stored with a timestamp
            if sp500_data:
                stock_insert_query = """
                    INSERT INTO financial_data 
                    (asset_name, price, market_cap, volume, recorded_date)
                    VALUES (?, ?, ?, ?, ?)
                """
                stock_records = [(symbol, price, None, volume, current_timestamp) for symbol, price, _, volume, _ in sp500_data]
                cursor.executemany(stock_insert_query, stock_records)

            conn.commit()

        log_message("INFO", "✅ Data storage complete.")

    except sqlite3.OperationalError as e:
        log_message("ERROR", f"Database error: {e}")


def format_yf_ticker(ticker):
    return ticker.replace(".", "-")  # Convert "BF.B" -> "BF-B"

# Fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    log_message("INFO", "Fetching S&P 500 ticker list...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find("table", {"id": "constituents"})
        if not table:
            log_message("ERROR", "Failed to find the S&P 500 table on Wikipedia.")
            return []

        tickers = []
        for row in table.find_all("tr")[1:]:  # Skip header row
            columns = row.find_all("td")
            if columns:
                ticker = columns[0].text.strip()
                tickers.append(format_yf_ticker(ticker))  # Convert dot notation

                if len(tickers) == 500:  # Ensure exactly 500 tickers
                    break

        log_message("INFO", f"Retrieved {len(tickers)} S&P 500 tickers.")
        return tickers

    except Exception as e:
        log_message("ERROR", f"Failed to fetch S&P 500 tickers: {e}")
        return []


# Fetch S&P 500 data with rate limits
def fetch_sp500_data():
    log_message("INFO", "Fetching S&P 500 data from Yahoo Finance...")
    tickers = get_sp500_tickers()
    chunk_size = 20
    sp500_data = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            log_message("INFO", f"Fetching batch {i+1}-{i+len(chunk)} of {len(tickers)} tickers...")
            stocks = yf.download(chunk, period="1d", auto_adjust=True, progress=False, threads=False)

            for symbol in chunk:
                if symbol in stocks["Close"] and not stocks["Close"][symbol].isna().all():
                    price = stocks["Close"][symbol].iloc[-1]
                    volume = stocks["Volume"][symbol].iloc[-1]
                    sp500_data.append((symbol, price, None, volume, datetime.now().strftime('%Y-%m-%d')))
                else:
                    failed_tickers.append(symbol)

        except yf.YFRateLimitError as e:
            log_message("ERROR", f"Rate limit hit for batch {chunk}. Retrying in 90 seconds...")
            time.sleep(90)  # Extended delay
            continue

        except Exception as e:
            log_message("ERROR", f"Failed fetching batch {chunk}: {e}")
            time.sleep(5)

    log_message("INFO", f"Finished S&P 500 data fetch. {len(sp500_data)} tickers successfully retrieved.")
    return sp500_data

def fetch_missing_tickers(missing_tickers):
    """Retry failed tickers one by one with a delay."""
    recovered_data = []
    for ticker in missing_tickers:
        try:
            log_message("INFO", f"Retrying missing ticker: {ticker}...")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                price = hist["Close"].iloc[-1]
                volume = hist["Volume"].iloc[-1]
                recovered_data.append((ticker, price, None, volume, datetime.now().strftime('%Y-%m-%d')))
            time.sleep(2)  # Delay between retries
        except Exception as e:
            log_message("ERROR", f"Failed retrying {ticker}: {e}")
    return recovered_data


def log_failed_tickers(failed_tickers):
    with open('failed_tickers.log', 'a') as f:
        for ticker in failed_tickers:
            f.write(f"{ticker} failed\n")

# Fetch cryptocurrency data
def fetch_crypto_data():
    log_message("INFO", "Fetching cryptocurrency data...")
    crypto_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
    return fetch_with_retries(crypto_url)


def generate_summary_report():
    """Generates an optimized summary report."""
    log_message("INFO", "Generating optimized summary report...")

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM financial_data WHERE recorded_date = DATE('now')
    """)
    fetched_stocks = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT AVG(price) FROM financial_data 
        WHERE asset_name IN (SELECT asset_name FROM financial_data WHERE recorded_date = DATE('now') LIMIT 100)
    """)
    avg_crypto_price = cursor.fetchone()[0] or 0

    report_date = datetime.now().strftime('%Y-%m-%d')

    cursor.execute("""
        INSERT INTO summary_reports (report_date, fetched_stocks, avg_crypto_price)
        VALUES (?, ?, ?)
    """, (report_date, fetched_stocks, avg_crypto_price))

    conn.commit()
    conn.close()

    log_message("INFO",
                f"Summary Report Generated: {report_date} | Stocks: {fetched_stocks} | Avg Crypto Price: {avg_crypto_price:.2f}")


# Initialize database
initialize_db()

def is_market_open():
    """Check if the market is open in Eastern Time."""
    est = pytz.timezone('US/Eastern')
    now = datetime.now(est)  # ✅ Set timezone to Eastern Time
    return 9 <= now.hour < 16

log_message("INFO", "Performing initial data pull for cryptos and S&P 500.")

# ✅ Fetch both datasets first, then store them together
crypto_data = fetch_crypto_data()
sp500_data = fetch_sp500_data()
store_data(crypto_data, sp500_data)  # ✅ Store both together

# ✅ Generate the first report after data storage completes
log_message("INFO", "✅ Initial data storage complete. Generating first report...")
generate_summary_report()
log_message("INFO", "✅ First report generated after initial data storage.")


# ✅ Schedule the **final daily stock pull** at 16:05 (after market close)
schedule.every().day.at("16:05").do(lambda: store_data(None, fetch_sp500_data()))

# ✅ Schedule the **final daily report** at 23:59
schedule.every().day.at("23:59").do(generate_summary_report)

log_message("INFO", "✅ Report generation scheduled only for end of day.")
log_message("INFO", "✅ Automated financial data tracking is active...")

# ✅ Keep the script running
while True:
    schedule.run_pending()
    time.sleep(60)
