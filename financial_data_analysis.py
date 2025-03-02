import sqlite3
import pandas as pd
from datetime import datetime

# Create and connect to SQLite database
conn = sqlite3.connect("fiscal_data.db")
cursor = conn.cursor()

# Create table for financial assets
tables = """
CREATE TABLE IF NOT EXISTS fiscal_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_name TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK(asset_type IN ('Stock', 'Bond', 'Crypto')),
    yield_percent REAL NOT NULL,
    recorded_date TEXT NOT NULL
);
"""
cursor.execute(tables)

# Insert sample data
data = [
    ("Apple Stock", "Stock", 2.5, datetime.now().strftime('%Y-%m-%d')),
    ("Tesla Stock", "Stock", 3.1, datetime.now().strftime('%Y-%m-%d')),
    ("US Treasury Bond", "Bond", 4.0, datetime.now().strftime('%Y-%m-%d')),
    ("Ethereum Staking", "Crypto", 5.5, datetime.now().strftime('%Y-%m-%d')),
    ("Solana Staking", "Crypto", 6.8, datetime.now().strftime('%Y-%m-%d')),
    ("Bitcoin Yield Fund", "Crypto", 3.9, datetime.now().strftime('%Y-%m-%d'))
]

cursor.executemany("INSERT INTO fiscal_data (asset_name, asset_type, yield_percent, recorded_date) VALUES (?, ?, ?, ?)", data)
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
