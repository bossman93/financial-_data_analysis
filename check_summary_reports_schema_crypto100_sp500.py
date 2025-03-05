import sqlite3

conn = sqlite3.connect("top100_sp500_fiscal_data.db")
cursor = conn.cursor()

# ✅ Check how many unique stock tickers were stored today
cursor.execute("""
    SELECT COUNT(DISTINCT asset_name) 
    FROM financial_data 
    WHERE recorded_date = DATE('now')
""")
unique_stocks = cursor.fetchone()[0]
print(f"✅ Unique Stocks Stored Today: {unique_stocks}")

# ✅ Check if some tickers were inserted multiple times
cursor.execute("""
    SELECT asset_name, COUNT(*) 
    FROM financial_data 
    WHERE recorded_date = DATE('now')
    GROUP BY asset_name
    HAVING COUNT(*) > 1
""")
duplicates = cursor.fetchall()

conn.close()

if duplicates:
    print("⚠️ Duplicate Stocks Found:")
    for stock in duplicates:
        print(f"{stock[0]} appears {stock[1]} times")
else:
    print("✅ No duplicate stocks found.")
