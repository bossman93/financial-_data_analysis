import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf

# Function to calculate MACD, RSI, Bollinger Bands, and Fibonacci retracements
def calculate_indicators(df):
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

    # Bollinger Bands calculation
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std().squeeze()  # Ensure it's a Series
    df['Bollinger_Upper'] = df['SMA20'] + (rolling_std * 2)
    df['Bollinger_Lower'] = df['SMA20'] - (rolling_std * 2)

    # Fibonacci Retracements calculation
    highest_price = df['Close'].max()
    lowest_price = df['Close'].min()
    df['Fibonacci_23_6'] = lowest_price + 0.236 * (highest_price - lowest_price)
    df['Fibonacci_38_2'] = lowest_price + 0.382 * (highest_price - lowest_price)
    df['Fibonacci_50_0'] = lowest_price + 0.5 * (highest_price - lowest_price)
    df['Fibonacci_61_8'] = lowest_price + 0.618 * (highest_price - lowest_price)

    return df

# Function to identify buy and sell signals based on the given strategy
def identify_trading_signals(df):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = df.columns.str.rstrip('_')

    signals = {'old_buy': [], 'old_sell': [], 'new_buy': [], 'new_sell': []}

    for i in range(1, len(df)):
        ticker = df.columns[0].split('_')[1]  # Extract ticker, e.g., 'BTC-USD'
        close_column = f'Close_{ticker}'

        # Use .iloc to avoid FutureWarning
        if df.iloc[i]['MACD'] > df.iloc[i]['Signal'] and df.iloc[i]['RSI'] < 50:
            signals['old_buy'].append(i)
        elif df.iloc[i]['MACD'] < df.iloc[i]['Signal'] and df.iloc[i]['RSI'] > 50:
            signals['old_sell'].append(i)

        if df.iloc[i]['MACD'] > df.iloc[i]['Signal'] and df.iloc[i]['RSI'] < 50 and df.iloc[i][close_column] < df.iloc[i]['Bollinger_Lower'] and df.iloc[i][close_column] > df.iloc[i]['Fibonacci_23_6']:
            signals['new_buy'].append(i)
        elif df.iloc[i]['MACD'] < df.iloc[i]['Signal'] and df.iloc[i]['RSI'] > 50 and df.iloc[i][close_column] > df.iloc[i]['Bollinger_Upper'] and df.iloc[i][close_column] < df.iloc[i]['Fibonacci_61_8']:
            signals['new_sell'].append(i)

    return signals

# Function to backtest and evaluate a strategy's performance
def backtest_strategy(df, signals, initial_balance=1000, strategy='new'):
    balance = initial_balance
    success_count = 0
    total_trades = 0
    max_drawdown = 0
    peak_balance = initial_balance

    ticker = df.columns[0].split('_')[1]  # Extract ticker
    close_column = f'Close_{ticker}'

    buy_signals = signals[f'{strategy}_buy']
    sell_signals = signals[f'{strategy}_sell']

    print(f"Buy Signals for {strategy}: {buy_signals}")
    print(f"Sell Signals for {strategy}: {sell_signals}")

    for i in range(1, len(df)):
        if i in buy_signals:  # Buy signal
            entry_price = df.iloc[i][close_column]  # Use dynamically generated column
            total_trades += 1
            balance -= entry_price  # Buy the asset
        elif i in sell_signals:  # Sell signal
            exit_price = df.iloc[i][close_column]  # Use dynamically generated column
            balance += exit_price  # Sell the asset

            if exit_price > entry_price:
                success_count += 1

        peak_balance = max(peak_balance, balance)
        drawdown = (peak_balance - balance) / peak_balance
        max_drawdown = max(max_drawdown, drawdown)

    success_rate = success_count / total_trades if total_trades > 0 else 0
    return {
        'final_balance': balance,
        'success_rate': success_rate,
        'max_drawdown': max_drawdown
    }

# Function to plot indicators and signals
def plot_indicators(df, crypto, signals):
    # Create a 'graphs' folder if it doesn't exist
    save_folder = "graphs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # Create the folder

    # Create the plot
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

    # Plot buy and sell signals
    for buy_signal in signals['new_buy']:
        ax1.axvline(buy_signal, color='green', linestyle='--', alpha=0.7)
        ax2.axvline(buy_signal, color='green', linestyle='--', alpha=0.7)
    for sell_signal in signals['new_sell']:
        ax1.axvline(sell_signal, color='red', linestyle='--', alpha=0.7)
        ax2.axvline(sell_signal, color='red', linestyle='--', alpha=0.7)

    # Save the plot in the 'graphs' folder
    plt.tight_layout()
    plt.savefig(f'{save_folder}/{crypto}_indicators_signals.png')  # Save in the 'graphs' folder
    plt.show()

# Main script to fetch data, process it, and generate performance comparisons
cryptos = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'ADA', 'DOGE', 'TRX', 'LINK', 'HBAR']

# Initialize an empty list to store the performance results for both strategies
performance_results = []

for crypto in cryptos:
    # Fetch historical data for each cryptocurrency
    df = yf.download(f'{crypto}-USD', start='2025-02-01', end='2025-03-01')

    # Calculate indicators
    df = calculate_indicators(df)

    # Identify trading signals for both strategies
    signals = identify_trading_signals(df)

    # Backtest and evaluate the old strategy (using MACD and RSI only)
    old_strategy_results = backtest_strategy(df, signals, strategy='old')

    # Backtest and evaluate the new strategy (using MACD, RSI, Bollinger Bands, Fibonacci)
    new_strategy_results = backtest_strategy(df, signals, strategy='new')

    # Store the results for comparison
    performance_results.append({
        'Crypto': crypto,
        'Old_Strategy_Final_Balance': old_strategy_results['final_balance'],
        'Old_Strategy_Success_Rate': old_strategy_results['success_rate'],
        'Old_Strategy_Max_Drawdown': old_strategy_results['max_drawdown'],
        'New_Strategy_Final_Balance': new_strategy_results['final_balance'],
        'New_Strategy_Success_Rate': new_strategy_results['success_rate'],
        'New_Strategy_Max_Drawdown': new_strategy_results['max_drawdown']
    })

    # Plot the indicators and signals for each cryptocurrency
    plot_indicators(df, crypto, signals)

# Convert results to DataFrame and save to CSV
performance_df = pd.DataFrame(performance_results)
performance_df.to_csv('strategy_comparison_report.csv', index=False)

# Print the comparison report
print(performance_df)
