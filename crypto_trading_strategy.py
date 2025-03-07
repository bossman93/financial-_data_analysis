import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# Define function to calculate MACD and RSI
def calculate_macd_rsi(df):
    """
    Calculate MACD and RSI indicators for a given dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing historical price data with a 'Close' column.

    Returns:
    pd.DataFrame: DataFrame with added MACD and RSI columns.
    """
    # MACD calculation
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


# Define function to determine entry and exit points
def identify_trading_signals(df):
    """
    Identify potential buy and sell signals based on MACD and RSI indicators.

    Parameters:
    df (pd.DataFrame): DataFrame containing historical price data with MACD and RSI columns.

    Returns:
    dict: A dictionary containing buy and sell signals.
    """
    buy_signals = df[(df['MACD'] > df['Signal']) & (df['RSI'] < 50)].index
    sell_signals = df[(df['MACD'] < df['Signal']) & (df['RSI'] > 50)].index

    return {'buy': buy_signals, 'sell': sell_signals}


# Define function to plot MACD, RSI, and signals
def plot_macd_rsi(df, crypto, signals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot MACD
    ax1.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax1.plot(df.index, df['Signal'], label='Signal Line', color='red')
    ax1.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray')
    ax1.set_title(f'{crypto} MACD')
    ax1.legend()

    # Plot RSI
    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
    ax2.set_title(f'{crypto} RSI')
    ax2.legend()

    # Mark buy and sell signals
    for buy_signal in signals['buy']:
        ax1.axvline(buy_signal, color='green', linestyle='--', alpha=0.7)
        ax2.axvline(buy_signal, color='green', linestyle='--', alpha=0.7)
    for sell_signal in signals['sell']:
        ax1.axvline(sell_signal, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(sell_signal, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{crypto}_macd_rsi_signals.png')  # Save the plot for verification
    plt.show()


# List of cryptocurrencies
cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE', 'TRX', 'LINK', 'HBAR']

# Fetch data and perform analysis
summary_data = []
for crypto in cryptos:
    df = yf.download(f'{crypto}-USD', start='2025-02-01', end='2025-03-01')
    df = calculate_macd_rsi(df)
    signals = identify_trading_signals(df)
    plot_macd_rsi(df, crypto, signals)

    # Append summary data
    summary_data.append({
        'Crypto': crypto,
        'Buy Signals': len(signals['buy']),
        'Sell Signals': len(signals['sell'])
    })

# Convert summary to DataFrame and save
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("crypto_trading_summary.csv", index=False)  # Save for third-party verification
print(summary_df)
