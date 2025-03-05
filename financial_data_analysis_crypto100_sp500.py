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

        -- Separate table for cryptocurrency data
        CREATE TABLE IF NOT EXISTS crypto_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            price REAL,
            market_cap REAL,
            volume REAL,
            recorded_date TEXT,
            UNIQUE(asset_name, recorded_date)
        );

        -- Separate table for stock market data
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            price REAL,
            volume REAL,
            recorded_date TEXT,
            UNIQUE(asset_name, recorded_date)
        );
        """)
        conn.commit()
        log_message("INFO", "✅ WAL Mode enabled and database initialized successfully.")


def store_data(crypto_data, sp500_data):
    try:
        log_message("INFO", "Storing data in database...")

        with sqlite3.connect("top100_sp500_fiscal_data.db") as conn:
            cursor = conn.cursor()
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # ✅ Store cryptocurrency data separately
            if crypto_data:
                crypto_insert_query = """
                    INSERT INTO crypto_data 
                    (asset_name, price, market_cap, volume, recorded_date)
                    VALUES (?, ?, ?, ?, ?)
                """
                crypto_records = [
                    (crypto["name"], crypto["current_price"], crypto["market_cap"], crypto["total_volume"],
                     current_timestamp)
                    for crypto in crypto_data[:100]  # ✅ Store only top 100 cryptos
                ]
                cursor.executemany(crypto_insert_query, crypto_records)
                log_message("INFO", f"✅ {len(crypto_records)} cryptos stored successfully at {current_timestamp}.")

            # ✅ Store S&P 500 stock data separately
            if sp500_data:
                stock_insert_query = """
                    INSERT INTO stock_data 
                    (asset_name, price, volume, recorded_date)
                    VALUES (?, ?, ?, ?)
                """
                stock_records = [(symbol, price, volume, current_timestamp) for symbol, price, _, volume, _ in
                                 sp500_data]
                cursor.executemany(stock_insert_query, stock_records)
                log_message("INFO", f"✅ {len(stock_records)} stocks stored successfully at {current_timestamp}.")

            conn.commit()

        log_message("INFO", "✅ Data storage complete.")

    except sqlite3.OperationalError as e:
        log_message("ERROR", f"Database error: {e}")


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
    """Generates an optimized summary report including stock and crypto performance."""
    log_message("INFO", "Generating optimized summary report...")

    conn = connect_db()
    cursor = conn.cursor()

    today = datetime.now().strftime('%Y-%m-%d')

    # ✅ Fix date query by matching full timestamp format
    cursor.execute("SELECT COUNT(*) FROM stock_data WHERE recorded_date LIKE ? || '%'", (today,))
    stored_stocks = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM crypto_data WHERE recorded_date LIKE ? || '%'", (today,))
    stored_cryptos = cursor.fetchone()[0] or 0

    # ✅ Fix performance queries to ensure proper percentage change calculations
    cursor.execute("""
        SELECT asset_name, 
               COALESCE(((price - LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date)) / 
               NULLIF(LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date), 0)) * 100, 0) AS percent_change
        FROM stock_data 
        WHERE recorded_date LIKE ? || '%'
        ORDER BY percent_change DESC
        LIMIT 3
    """, (today,))
    top_stocks = cursor.fetchall()

    cursor.execute("""
        SELECT asset_name, 
               COALESCE(((price - LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date)) / 
               NULLIF(LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date), 0)) * 100, 0) AS percent_change
        FROM stock_data 
        WHERE recorded_date LIKE ? || '%'
        ORDER BY percent_change ASC
        LIMIT 3
    """, (today,))
    worst_stocks = cursor.fetchall()

    cursor.execute("""
        SELECT asset_name, 
               COALESCE(((price - LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date)) / 
               NULLIF(LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date), 0)) * 100, 0) AS percent_change
        FROM crypto_data 
        WHERE recorded_date LIKE ? || '%'
        ORDER BY percent_change DESC
        LIMIT 3
    """, (today,))
    top_cryptos = cursor.fetchall()

    cursor.execute("""
        SELECT asset_name, 
               COALESCE(((price - LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date)) / 
               NULLIF(LAG(price) OVER (PARTITION BY asset_name ORDER BY recorded_date), 0)) * 100, 0) AS percent_change
        FROM crypto_data 
        WHERE recorded_date LIKE ? || '%'
        ORDER BY percent_change ASC
        LIMIT 3
    """, (today,))
    worst_cryptos = cursor.fetchall()

    conn.close()

    # ✅ Market Sentiment Analysis
    total_gainers = len([s for s in top_stocks if s[1] is not None]) + len([c for c in top_cryptos if c[1] is not None])
    total_losers = len([s for s in worst_stocks if s[1] is not None]) + len([c for c in worst_cryptos if c[1] is not None])
    market_sentiment = "Bullish" if total_gainers > total_losers else "Bearish"

    # ✅ Store the report in `summary_reports` table
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO summary_reports (report_date, fetched_stocks, avg_crypto_price)
        VALUES (?, ?, ?)
    """, (today, stored_stocks, stored_cryptos))
    conn.commit()
    conn.close()

    # ✅ Print the fixed report
    print("\n📊 Daily Market Summary ({})".format(today))
    print("-----------------------------------")
    print("Stocks Retrieved: {}".format(stored_stocks))
    print("Cryptocurrencies Retrieved: {}\n".format(stored_cryptos))

    print("📈 Top 3 Performing Stocks:")
    for stock in top_stocks:
        percent_change = stock[1] if stock[1] is not None else "N/A"
        print("{}: {:.2f}%".format(stock[0], percent_change) if isinstance(percent_change, (int, float)) else "{}: {}".format(stock[0], percent_change))

    print("\n📉 Worst 3 Performing Stocks:")
    for stock in worst_stocks:
        percent_change = stock[1] if stock[1] is not None else "N/A"
        print("{}: {:.2f}%".format(stock[0], percent_change) if isinstance(percent_change, (int, float)) else "{}: {}".format(stock[0], percent_change))

    print("\n🔥 Top 3 Cryptos:")
    for crypto in top_cryptos:
        percent_change = crypto[1] if crypto[1] is not None else "N/A"
        print("{}: {:.2f}%".format(crypto[0], percent_change) if isinstance(percent_change, (int, float)) else "{}: {}".format(crypto[0], percent_change))

    print("\n❄️ Worst 3 Cryptos:")
    for crypto in worst_cryptos:
        percent_change = crypto[1] if crypto[1] is not None else "N/A"
        print("{}: {:.2f}%".format(crypto[0], percent_change) if isinstance(percent_change, (int, float)) else "{}: {}".format(crypto[0], percent_change))

    print("\n📊 Market Sentiment: {}".format(market_sentiment))
    print("-----------------------------------")

    log_message("INFO", "✅ Summary Report Generated Successfully.")

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
