import sqlite3

conn = sqlite3.connect("fiscal_data.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM fiscal_data ORDER BY yield_percent DESC;")
for row in cursor.fetchall():
    print(row)
