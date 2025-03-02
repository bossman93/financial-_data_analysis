import requests
import csv

# Free API URL (replace 'your_api_key' with a valid key from exchangerate-api.com)
API_URL = "https://v6.exchangerate-api.com/v6/df4277356b08a49bcddc5540/latest/USD"

# File to store exchange rates
CSV_FILE = "exchange_rates.csv"

def fetch_exchange_rates():
    """Fetches exchange rates from the API and writes them to a CSV file."""
    response = requests.get(API_URL)
    if response.status_code == 200:
        data = response.json()
        rates = data["conversion_rates"]

        # Write to CSV
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Currency", "Exchange Rate (vs USD)"])  # Header row
            for currency, rate in rates.items():
                writer.writerow([currency, rate])

        print(f"Exchange rates saved to {CSV_FILE}")
    else:
        print("Failed to fetch exchange rates.")

def find_strongest_weakest_currency():
    """Reads the CSV file and identifies the strongest and weakest currency compared to USD."""
    with open(CSV_FILE, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row

        rates = list(reader)
        rates.sort(key=lambda x: float(x[1]))  # Sort by exchange rate

        weakest = rates[-1]  # Highest rate (weakest currency)
        strongest = rates[0]  # Lowest rate (strongest currency)

        print(f"Strongest Currency: {strongest[0]} (Rate: {strongest[1]})")
        print(f"Weakest Currency: {weakest[0]} (Rate: {weakest[1]})")

# Run functions
fetch_exchange_rates()
find_strongest_weakest_currency()