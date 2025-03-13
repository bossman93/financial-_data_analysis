print("Starting script...")

import pandas as pd
print("Pandas imported")

import numpy as np
print("Numpy imported")

import sqlite3
print("SQLite3 imported")

import requests
print("Requests imported")

import matplotlib.pyplot as plt
print("Matplotlib imported")

import mplfinance as mpf
print("Mplfinance imported")

import logging
print("Logging imported")

import hashlib
print("Hashlib imported")

import os
print("OS imported")

import json
print("JSON imported")

from sklearn.model_selection import train_test_split, cross_val_score
print("Sklearn model selection imported")

from sklearn.ensemble import RandomForestClassifier
print("RandomForestClassifier imported")

import xgboost as xgb
print("XGBoost imported")

from sklearn.preprocessing import StandardScaler
print("StandardScaler imported")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy score imported")

from datetime import datetime
print("Datetime imported")

import shap
print("SHAP imported")

import tweepy
print("Tweepy imported")

from textblob import TextBlob
print("TextBlob imported")

import time
print("Time imported")

print("All imports successful!")

# Configure logging
logging.basicConfig(filename="trade_logs.txt", level=logging.INFO, encoding="utf-8",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Database path
db_path = os.path.abspath("crypto_analysis.db")

# File path for Twitter API keys
KEYS_FILE = "twitter_api_keys.json"


def get_user_assets():
    """Prompt user for asset selection and validate input."""
    default_assets = ['BTC', 'ETH', 'XRP']
    user_input = input(
        f"Enter cryptocurrencies to analyze (comma-separated, or press Enter for default {default_assets}): ").strip()

    if not user_input:
        print(f"✅ Using default assets: {default_assets}")
        return default_assets

    assets = [asset.strip().upper() for asset in user_input.split(",")]
    print(f"✅ Selected assets: {assets}")
    return assets


def get_user_scenario():
    """Prompt user to select a market scenario."""
    scenarios = ['bullish', 'bearish', 'high_vol', 'low_liq', 'all']

    print("\nMarket Scenarios:")
    print("1 - Bullish (RSI > 60 & MACD Positive)")
    print("2 - Bearish (RSI < 40 & MACD Negative)")
    print("3 - High Volatility (High Std Dev in Close Prices)")
    print("4 - Low Liquidity (Low Trading Volume)")
    print("5 - All Market Scenarios")

    user_choice = input("Select a market scenario (1-5, default = 5): ").strip()

    if user_choice in ["1", "2", "3", "4"]:
        selected_scenario = scenarios[int(user_choice) - 1]
        print(f"✅ Selected market scenario: {selected_scenario}")
        return selected_scenario

    print("✅ Defaulting to all market scenarios.")
    return "all"


def fetch_data(crypto):
    """Fetch historical crypto data from Binance."""
    url = f"https://api.binance.us/api/v3/klines?symbol={crypto}USDT&interval=1d&limit=365"
    try:
        response = requests.get(url, timeout=10)  # Added timeout for better error handling

        if response.status_code != 200:
            logging.warning(f"⚠️ No data for {crypto}. Status code: {response.status_code}")
            print(f"⚠️ No data for {crypto}. Status code: {response.status_code}")
            return pd.DataFrame()  # Return empty DataFrame if API call fails

        data = response.json()

        # Check if data is empty
        if not data:
            logging.warning(f"⚠️ Empty data received for {crypto}")
            print(f"⚠️ Empty data received for {crypto}")
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteVolume", "Trades", "TakerBase", "TakerQuote", "Ignore"
        ])

        # Ensure 'df' is created before any further processing
        if df.empty:
            print(f"⚠️ No data available for {crypto}")
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

        # Inspect the first few Close prices
        print(f"\n{crypto} close prices (first 5 rows):")
        print(df[['Close']].head())

        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"⚠️ Error fetching data for {crypto}: {e}")
        print(f"⚠️ Error fetching data for {crypto}: {e}")
        return pd.DataFrame()


def calculate_indicators(df):
    """Calculate MACD, RSI, Bollinger Bands, and Stochastic RSI."""
    if df.empty:
        return df

    try:
        # Calculate MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']

        # Calculate RSI with error handling for division by zero
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        # Handle division by zero in RSI calculation
        rs = pd.Series(index=avg_loss.index, dtype=float)
        for i, (g, l) in enumerate(zip(avg_gain, avg_loss)):
            rs.iloc[i] = g / l if l != 0 else 100

        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['STDDEV_20'] = df['Close'].rolling(20).std()
        df['Bollinger_Upper'] = df['SMA_20'] + 2 * df['STDDEV_20']
        df['Bollinger_Lower'] = df['SMA_20'] - 2 * df['STDDEV_20']

        # Calculate Stochastic RSI with error handling
        rsi_min = df['RSI'].rolling(14).min()
        rsi_max = df['RSI'].rolling(14).max()
        denominator = rsi_max - rsi_min

        # Replace zero denominators with small values to avoid division by zero
        denominator = denominator.replace(0, 1e-10)

        df['Stochastic_RSI'] = (df['RSI'] - rsi_min) / denominator
        df['Stochastic_RSI'] = df['Stochastic_RSI'].clip(0, 1)  # Ensure values between 0 and 1

        # Calculate percentage change in Close prices for future price movement
        df['pct_change'] = df['Close'].pct_change() * 100
        df['price_change'] = df['Close'].pct_change().shift(-1) * 100

        # Drop NaN values but don't modify the original dataframe (inplace=False)
        cleaned_df = df.dropna()

        # Print percentage changes for debugging
        print("\nPercentage changes (first 5 rows):")
        print(df[['Close', 'pct_change']].head())

        return cleaned_df

    except Exception as e:
        logging.error(f"⚠️ Error calculating indicators: {e}")
        print(f"⚠️ Error calculating indicators: {e}")
        return df


def store_results(crypto, df):
    """Store computed indicators in SQLite database."""
    if df.empty:
        print(f"⚠️ No data to store for {crypto}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table with enhanced schema for market conditions and sentiment
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
            fear_and_greed_index REAL,
            social_media_sentiment REAL,
            market_condition REAL,
            market_scenario TEXT,
            volatility_score REAL,
            liquidity_score REAL,
            model_prediction REAL,
            model_confidence REAL,
            feature_importance TEXT
        )""")

        # Check for columns to ensure they exist for backward compatibility
        cursor.execute(f"PRAGMA table_info(indicators)")
        columns = [column[1] for column in cursor.fetchall()]

        # Add missing columns if needed
        new_columns = {
            'fear_and_greed_index': 'REAL',
            'social_media_sentiment': 'REAL',
            'market_scenario': 'TEXT',
            'volatility_score': 'REAL',
            'liquidity_score': 'REAL',
            'model_prediction': 'REAL',
            'model_confidence': 'REAL',
            'feature_importance': 'TEXT'
        }

        for col, col_type in new_columns.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE indicators ADD COLUMN {col} {col_type}")

        # Calculate additional scores
        df['volatility_score'] = df['Close'].rolling(14).std() / df['Close'].rolling(14).mean()
        df['liquidity_score'] = df['Volume'].rolling(14).mean() / df['Volume'].rolling(14).std()

        # Get the Fear & Greed Index value from the DataFrame
        fng_value = float(df['fear_and_greed_index'].iloc[0]) if 'fear_and_greed_index' in df.columns else 50.0
        print(f"Storing Fear & Greed Index value: {fng_value}")

        # Store the data
        for index, row in df.iterrows():
            # Determine market condition: 2 (bullish), 0 (bearish), or 1 (neutral)
            market_condition = float(2.0 if row.get('price_change', 0) > 1 else 0.0 if row.get('price_change', 0) < -1 else 1.0)

            # Determine market scenario based on indicators
            if row.get('MACD', 0) > 0 and row.get('RSI', 50) > 60:
                market_scenario = 'bullish'
            elif row.get('MACD', 0) < 0 and row.get('RSI', 50) < 40:
                market_scenario = 'bearish'
            elif row.get('volatility_score', 0) > 1.5:
                market_scenario = 'high_vol'
            elif row.get('liquidity_score', 0) < 0.5:
                market_scenario = 'low_liq'
            else:
                market_scenario = 'neutral'

            # Store feature importance as JSON if available
            feature_importance = row.get('feature_importance', {})
            feature_importance_json = json.dumps(feature_importance) if feature_importance else None

            cursor.execute("""
            INSERT INTO indicators 
            (crypto, date, macd, rsi, bollinger_upper, bollinger_lower, stochastic_rsi, 
            fear_and_greed_index, social_media_sentiment, market_condition, market_scenario,
            volatility_score, liquidity_score, model_prediction, model_confidence, feature_importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                           (crypto, index.strftime('%Y-%m-%d'),
                            float(row.get('MACD', 0)),
                            float(row.get('RSI', 50)),
                            float(row.get('Bollinger_Upper', 0)),
                            float(row.get('Bollinger_Lower', 0)),
                            float(row.get('Stochastic_RSI', 0)),
                            float(fng_value),  # Use the Fear & Greed Index value
                            float(row.get('social_media_sentiment', 0)),
                            float(market_condition),
                            market_scenario,
                            float(row.get('volatility_score', 0)),
                            float(row.get('liquidity_score', 0)),
                            None,  # model_prediction will be updated later
                            None,  # model_confidence will be updated later
                            feature_importance_json))

        conn.commit()
        conn.close()
        print(f"✅ Stored enhanced indicator data for {crypto}")

    except sqlite3.Error as e:
        logging.error(f"⚠️ Database error: {e}")
        print(f"⚠️ Database error: {e}")
    except Exception as e:
        logging.error(f"⚠️ Error storing results: {e}")
        print(f"⚠️ Error storing results: {e}")


def generate_commit_hash():
    """Generate a commit hash for reproducibility."""
    try:
        with open(__file__, "rb") as f:
            content = f.read()
        commit_hash = hashlib.sha256(content).hexdigest()
        print(f"\n✅ Commit Hash: {commit_hash}")
        logging.info(f"Commit Hash: {commit_hash}")
        return commit_hash
    except Exception as e:
        print(f"⚠️ Error generating commit hash: {e}")
        logging.warning(f"Error generating commit hash: {e}")
        return None


def apply_market_scenario(df, market_scenario):
    """Apply user-defined market scenario to filter data."""
    if df.empty:
        return df

    original_row_count = len(df)

    try:
        if market_scenario == "bullish":
            filtered_df = df[(df['MACD'] > 0) & (df['RSI'] > 60)]
        elif market_scenario == "bearish":
            filtered_df = df[(df['MACD'] < 0) & (df['RSI'] < 40)]
        elif market_scenario == "high_vol":
            std_dev = df['Close'].rolling(14).std()
            avg_std_dev = std_dev.mean()
            filtered_df = df[std_dev > avg_std_dev]
        elif market_scenario == "low_liq":
            avg_volume = df['Volume'].rolling(14).mean()
            filtered_df = df[df['Volume'] < avg_volume]
        else:  # "all" scenario
            filtered_df = df

        filtered_row_count = len(filtered_df)
        print(f"Applied {market_scenario} filter: {original_row_count} rows → {filtered_row_count} rows")

        # If filtering removed all rows, return original data with a warning
        if filtered_row_count == 0 and original_row_count > 0:
            print(f"⚠️ Filter for {market_scenario} removed all data. Using original data instead.")
            return df

        return filtered_df

    except Exception as e:
        logging.error(f"⚠️ Error applying market scenario: {e}")
        print(f"⚠️ Error applying market scenario: {e}")
        return df  # Return original dataframe if error


def save_twitter_api_keys(api_key, api_secret_key, access_token, access_token_secret, bearer_token):
    """Save the Twitter API keys to a JSON file."""
    keys = {
        "api_key": api_key,
        "api_secret_key": api_secret_key,
        "access_token": access_token,
        "access_token_secret": access_token_secret,
        "bearer_token": bearer_token
    }

    try:
        with open(KEYS_FILE, 'w') as file:
            json.dump(keys, file, indent=4)
        print("✅ Twitter API keys saved successfully.")
    except Exception as e:
        print(f"⚠️ Error saving Twitter API keys: {e}")


def load_twitter_api_keys():
    """Load the Twitter API keys from a JSON file, handling empty or corrupt files."""
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, 'r') as file:
                keys = json.load(file)

            # Ensure the required keys exist
            required_keys = ["api_key", "api_secret_key", "access_token", "access_token_secret", "bearer_token"]
            if not all(key in keys for key in required_keys):
                print("⚠️ Missing required Twitter API keys. Please re-enter them.")
                return None

            print("✅ Twitter API keys loaded successfully.")
            return keys
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Error loading Twitter API keys: {e}. Please re-enter your keys.")
            return None
    else:
        print("⚠️ No saved Twitter API keys found. Please enter your keys.")
        return None


def get_twitter_api_keys():
    """Prompt user for API keys if they don't exist or are invalid."""
    # First try to load existing keys
    existing_keys = load_twitter_api_keys()
    if existing_keys:
        return existing_keys

    # If no valid keys exist, prompt for new ones
    print("\n--- Twitter API Configuration ---")
    api_key = input("Enter your Twitter API Key: ").strip()
    api_secret_key = input("Enter your Twitter API Secret Key: ").strip()
    access_token = input("Enter your Twitter Access Token: ").strip()
    access_token_secret = input("Enter your Twitter Access Token Secret: ").strip()
    bearer_token = input("Enter your Twitter Bearer Token (Leave blank to use API keys): ").strip()

    # Save the keys to a file for later use
    save_twitter_api_keys(api_key, api_secret_key, access_token, access_token_secret, bearer_token)

    keys = {
        "api_key": api_key,
        "api_secret_key": api_secret_key,
        "access_token": access_token,
        "access_token_secret": access_token_secret,
        "bearer_token": bearer_token
    }

    return keys


def authenticate_twitter_api(api_key, api_secret_key, access_token, access_token_secret, bearer_token):
    """Authenticate to Twitter API using either OAuth 1.0a (API keys) or OAuth 2.0 (Bearer Token)."""
    try:
        if bearer_token:
            print("✅ Using Bearer Token for OAuth 2.0 authentication.")
            headers = {"Authorization": f"Bearer {bearer_token}"}
            test_url = "https://api.twitter.com/2/tweets/search/recent?query=bitcoin&max_results=10"
            test_response = requests.get(test_url, headers=headers)

            # Check rate limit headers
            remaining_requests = test_response.headers.get('x-rate-limit-remaining')
            reset_time = test_response.headers.get('x-rate-limit-reset')
            
            if test_response.status_code == 429:  # Too Many Requests
                if reset_time:
                    reset_datetime = datetime.fromtimestamp(int(reset_time))
                    wait_time = (reset_datetime - datetime.now()).total_seconds()
                    if wait_time > 0:
                        print(f"⚠️ Rate limit exceeded. Waiting {wait_time:.0f} seconds until reset...")
                        time.sleep(wait_time)
                        # Retry after waiting
                        test_response = requests.get(test_url, headers=headers)
                    else:
                        print("⚠️ Rate limit exceeded. Please try again later.")
                        return None, None
                else:
                    print("⚠️ Rate limit exceeded. Please try again later.")
                    return None, None
            elif test_response.status_code == 401:
                print("⚠️ Bearer Token is invalid or expired.")
                return None, None
            elif test_response.status_code == 403:
                print("⚠️ Bearer Token doesn't have the required permissions.")
                return None, None

            if test_response.status_code == 200:
                print("✅ Bearer Token authentication successful.")
                print(f"Remaining requests: {remaining_requests}")
                return headers, None
            else:
                print(f"⚠️ Bearer Token authentication failed. Status Code: {test_response.status_code}")
                print(f"⚠️ Response: {test_response.text}")
                return None, None

        else:
            print("✅ Using OAuth 1.0a (API Key and Access Token) for authentication.")
            try:
                # Create OAuth 1.0a User Context authentication
                auth = tweepy.OAuth1UserHandler(
                    api_key,
                    api_secret_key,
                    access_token,
                    access_token_secret
                )
                
                # Create API object with wait_on_rate_limit=True
                api = tweepy.API(auth, wait_on_rate_limit=True)

                # Test the API connection with a simple search
                test_tweets = api.search_tweets(q="bitcoin", count=1)
                print("✅ OAuth 1.0a authentication successful.")
                return None, api
            except tweepy.Unauthorized:
                print("⚠️ API Key or Access Token is invalid.")
                return None, None
            except tweepy.Forbidden:
                print("⚠️ API Key or Access Token doesn't have the required permissions.")
                return None, None
            except tweepy.TweepError as e:
                print(f"⚠️ Tweepy API error: {str(e)}")
                return None, None

    except Exception as e:
        print(f"⚠️ Error authenticating Twitter API: {str(e)}")
        return None, None


# Add caching for sentiment analysis
sentiment_cache = {}
sentiment_cache_timeout = 3600  # 1 hour in seconds

def get_cached_sentiment(crypto):
    """Get cached sentiment if available and not expired."""
    if crypto in sentiment_cache:
        cached_data = sentiment_cache[crypto]
        if time.time() - cached_data['timestamp'] < sentiment_cache_timeout:
            print(f"✅ Using cached sentiment for {crypto}")
            return cached_data['sentiment']
    return None

def cache_sentiment(crypto, sentiment):
    """Cache sentiment data with timestamp."""
    sentiment_cache[crypto] = {
        'sentiment': sentiment,
        'timestamp': time.time()
    }

def get_coingecko_id(symbol):
    """Map trading symbol to CoinGecko ID."""
    # Common mappings
    coingecko_ids = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'LINK': 'chainlink',
        'XRP': 'ripple',
        'PEPE': 'pepe',
        'SOL': 'solana',
        'ADA': 'cardano',
        'DOGE': 'dogecoin',
        'DOT': 'polkadot',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'UNI': 'uniswap',
        'AAVE': 'aave',
        'COMP': 'compound-governance-token',
        'SUSHI': 'sushi',
        'YFI': 'yearn-finance',
        'CRV': 'curve-dao-token',
        'SNX': 'havven',
        'MKR': 'maker',
        'BAL': 'balancer'
    }
    
    # Try exact match first
    if symbol.upper() in coingecko_ids:
        return coingecko_ids[symbol.upper()]
    
    # Try lowercase match
    if symbol.lower() in coingecko_ids:
        return coingecko_ids[symbol.lower()]
    
    # If no match found, try using the symbol as is (for newer tokens)
    return symbol.lower()

def fetch_social_media_sentiment(crypto, headers=None, api=None):
    """Fetch and analyze sentiment for a cryptocurrency from CoinGecko API."""
    try:
        # Get the correct CoinGecko ID
        coingecko_id = get_coingecko_id(crypto)
        
        # CoinGecko API endpoint for social sentiment
        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "community_data": "true",
            "developer_data": "false",
            "sparkline": "false"
        }
        
        print(f"🔍 Fetching sentiment from CoinGecko for {crypto} (ID: {coingecko_id})...")
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            
            # Extract social metrics
            community_data = data.get('community_data', {})
            
            # Calculate sentiment score from various metrics
            twitter_followers = community_data.get('twitter_followers', 0)
            reddit_subscribers = community_data.get('reddit_subscribers', 0)
            reddit_accounts_active_48h = community_data.get('reddit_accounts_active_48h', 0)
            
            # Calculate relative metrics
            twitter_growth = community_data.get('twitter_followers_change_24h', 0)
            reddit_growth = community_data.get('reddit_subscribers_change_24h', 0)
            
            # Calculate sentiment score (-1 to 1)
            # Weight the different factors
            follower_weight = 0.4
            growth_weight = 0.4
            activity_weight = 0.2
            
            # Normalize follower counts (assuming max values)
            max_followers = 1000000  # Adjust based on typical values
            follower_score = min(twitter_followers / max_followers, 1)
            
            # Calculate growth score (-1 to 1)
            growth_score = 0
            if twitter_followers > 0:
                growth_score += (twitter_growth / twitter_followers) * 0.5
            if reddit_subscribers > 0:
                growth_score += (reddit_growth / reddit_subscribers) * 0.5
            
            # Calculate activity score (0 to 1)
            max_active = 10000  # Adjust based on typical values
            activity_score = min(reddit_accounts_active_48h / max_active, 1)
            
            # Combine scores
            overall_sentiment = (
                follower_score * follower_weight +
                growth_score * growth_weight +
                activity_score * activity_weight
            )
            
            # Normalize to -1 to 1 range
            overall_sentiment = max(min(overall_sentiment * 2 - 1, 1), -1)
            
            print(f"✅ Sentiment for {crypto}: {overall_sentiment:.4f}")
            print(f"  Twitter Followers: {twitter_followers:,}")
            print(f"  Reddit Subscribers: {reddit_subscribers:,}")
            print(f"  Active Reddit Users: {reddit_accounts_active_48h:,}")
            print(f"  Growth Score: {growth_score:.4f}")
            
            # Cache the result
            cache_sentiment(crypto, overall_sentiment)
            return overall_sentiment
            
        elif response.status_code == 429:  # Rate limit
            print("⚠️ CoinGecko API rate limit reached. Using cached data if available.")
            return get_cached_sentiment(crypto) or 0
        else:
            print(f"⚠️ CoinGecko API error: {response.status_code}")
            print(f"  URL: {url}")
            return get_cached_sentiment(crypto) or 0

    except Exception as e:
        logging.error(f"⚠️ Error fetching sentiment from CoinGecko: {str(e)}")
        print(f"⚠️ Error fetching sentiment from CoinGecko: {str(e)}")
        return get_cached_sentiment(crypto) or 0


def fetch_fear_and_greed_index():
    """Fetch the Fear and Greed Index for market sentiment, handling errors properly."""
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()

            # Proper error handling for unexpected JSON structure
            if "data" not in data or not data["data"]:
                print("⚠️ Unexpected response format from Fear & Greed API. Using default neutral value.")
                return 50, "Neutral"

            fng_data = data["data"][0]

            # Validate the returned data structure
            if "value" not in fng_data or "value_classification" not in fng_data:
                print("⚠️ Missing required fields in Fear & Greed API response. Using default neutral value.")
                return 50, "Neutral"

            try:
                fng_value = int(fng_data["value"])
            except (ValueError, TypeError):
                print("⚠️ Invalid value in Fear & Greed API response. Using default neutral value.")
                fng_value = 50

            fng_score = fng_data["value_classification"]
            print(f"✅ Fear & Greed Index: {fng_value}, Sentiment: {fng_score}")
            return fng_value, fng_score
        else:
            print(
                f"⚠️ Failed to fetch Fear & Greed Index. Status code: {response.status_code}. Using default neutral value.")
            return 50, "Neutral"
    except Exception as e:
        print(f"⚠️ Error fetching Fear & Greed Index: {e}. Using default neutral value.")
        return 50, "Neutral"


def add_sentiment_to_dataframe(df, crypto, headers=None, api=None):
    """Add sentiment features to the cryptocurrency DataFrame."""
    if df.empty:
        return df

    # Fetch sentiment data
    fng_value, fng_score = fetch_fear_and_greed_index()
    social_media_sentiment = fetch_social_media_sentiment(crypto, headers, api)

    # Create copies of the sentiment values for all rows in the DataFrame
    df['fear_and_greed_index'] = fng_value if fng_value is not None else 50  # Default to neutral
    df[
        'social_media_sentiment'] = social_media_sentiment if social_media_sentiment is not None else 0  # Default to neutral

    # Log the sentiment values
    print(
        f"Added sentiment data: Fear & Greed Index = {fng_value}, Social Media Sentiment = {social_media_sentiment:.4f}")

    return df


def train_and_evaluate_models(X_train, y_train):
    """Train and evaluate both Random Forest and XGBoost models with cross-validation."""
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        print("⚠️ Empty training data. Cannot train models.")
        return None

    try:
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        y_train = np.nan_to_num(y_train, nan=1)  # Default to neutral for missing values

        # Initialize models
        rf_model = RandomForestClassifier(
                n_estimators=100,
            max_depth=6,
                random_state=42
            )
        
        xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )

        # Cross-validation for both models
        rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
        xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')

        # Train models on full dataset
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        # Calculate feature importance for both models
        rf_importance = rf_model.feature_importances_
        xgb_importance = xgb_model.feature_importances_

        # Create feature importance dictionaries
        feature_names = [f"feature_{i}" for i in range(len(rf_importance))]
        rf_feature_importance = dict(zip(feature_names, rf_importance.tolist()))
        xgb_feature_importance = dict(zip(feature_names, xgb_importance.tolist()))

        # Calculate model confidence using probability predictions
        rf_pred_proba = rf_model.predict_proba(X_train)
        xgb_pred_proba = xgb_model.predict_proba(X_train)
        
        rf_confidence = np.max(rf_pred_proba, axis=1)
        xgb_confidence = np.max(xgb_pred_proba, axis=1)

        return {
            'rf': {
                'model': rf_model,
                'cv_scores': rf_cv_scores,
                'feature_importance': rf_feature_importance,
                'confidence': rf_confidence
            },
            'xgb': {
                'model': xgb_model,
                'cv_scores': xgb_cv_scores,
                'feature_importance': xgb_feature_importance,
                'confidence': xgb_confidence
            }
        }

    except Exception as e:
        logging.error(f"⚠️ Error training models: {e}")
        print(f"⚠️ Error training models: {e}")
        return None


def compare_models(model_results, X_test, y_test):
    """Compare performance of Random Forest and XGBoost models."""
    if model_results is None:
        return None

    comparison = {}
    
    for model_name, results in model_results.items():
        model = results['model']
        pred = model.predict(X_test)
        
        # Calculate various metrics
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')
        
        # Calculate confidence scores
        pred_proba = model.predict_proba(X_test)
        confidence = np.max(pred_proba, axis=1)
        avg_confidence = np.mean(confidence)
        
        comparison[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': avg_confidence,
            'cv_mean': np.mean(results['cv_scores']),
            'cv_std': np.std(results['cv_scores'])
        }
    
    return comparison


def generate_market_summary(selected_cryptos, market_scenario, model_results):
    """Generate a filtered market summary report with actionable insights."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM indicators ORDER BY date DESC", conn)
        conn.close()

        # Filter for selected cryptocurrencies
        df = df[df['crypto'].isin(selected_cryptos)]

        if df.empty:
            return "⚠️ No data available for the selected assets and market scenario."

        summary = []
        for crypto in selected_cryptos:
            crypto_df = df[df['crypto'] == crypto]

            if crypto_df.empty:
                summary.append(f"{crypto}: ⚠️ No data available")
                continue

            # Get the latest data point for each crypto
            latest = crypto_df.iloc[0]  # Since we sorted by date DESC

            # Get model results for this crypto
            model_result = next((item for item in model_results if item["Asset"] == crypto), None)

            # Technical Analysis
            technical_indicators = []
            if 'macd' in latest:
                technical_indicators.append(f"MACD: {latest['macd']:.2f}")
            if 'rsi' in latest:
                technical_indicators.append(f"RSI: {latest['rsi']:.1f}")
            if 'stochastic_rsi' in latest:
                technical_indicators.append(f"StochRSI: {latest['stochastic_rsi']:.2f}")

            # Determine trend based on technical indicators
            technical_trend = "Bullish" if latest.get('macd', 0) > 0 and latest.get('rsi', 50) > 60 else \
                "Bearish" if latest.get('macd', 0) < 0 and latest.get('rsi', 50) < 40 else "Neutral"

            # Format trend information
            trend_info = f"Technical Analysis: {technical_trend}"
            if technical_indicators:
                trend_info += f" | {' | '.join(technical_indicators)}"

            # Market Scenario Analysis
            scenario_info = []
            if 'market_scenario' in latest:
                scenario_info.append(f"Scenario: {latest['market_scenario']}")

            if 'volatility_score' in latest:
                vol_level = "High" if latest['volatility_score'] > 1.5 else "Normal" if latest['volatility_score'] > 0.5 else "Low"
                scenario_info.append(f"Volatility: {vol_level}")

            if 'liquidity_score' in latest:
                liq_level = "High" if latest['liquidity_score'] > 1.5 else "Normal" if latest['liquidity_score'] > 0.5 else "Low"
                scenario_info.append(f"Liquidity: {liq_level}")

            # Sentiment Analysis
            sentiment_info = []
            if 'fear_and_greed_index' in latest:
                fng = latest['fear_and_greed_index']
                fng_category = "Extreme Fear" if fng <= 25 else "Fear" if fng <= 40 else \
                    "Neutral" if fng <= 60 else "Greed" if fng <= 75 else "Extreme Greed"
                sentiment_info.append(f"Fear & Greed: {fng_category} ({fng:.0f})")

            if 'social_media_sentiment' in latest:
                social = latest['social_media_sentiment']
                social_category = "Negative" if social < -0.1 else "Neutral" if social < 0.1 else "Positive"
                sentiment_info.append(f"Social: {social_category} ({social:.2f})")

            # ML Model Insights
            ml_info = []
            if model_result:
                ml_info.append(f"Model Accuracy: {model_result['Model Accuracy (%)']:.1f}%")
                ml_info.append(f"Model Confidence: {model_result['Model Confidence']*100:.1f}%")

                if latest.get('model_prediction') is not None:
                    model_trend = "Bullish" if latest['model_prediction'] == 2 else \
                        "Bearish" if latest['model_prediction'] == 0 else "Neutral"
                    model_confidence = latest.get('model_confidence', 0) * 100
                    ml_info.append(f"ML Prediction: {model_trend} ({model_confidence:.1f}% confidence)")

            # Feature Importance
            feature_insights = ""
            if model_result and 'Feature Importance' in model_result:
                rf_importance = model_result['Feature Importance']['rf']
                if rf_importance:
                    top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    feature_insights = "\nTop Features: " + ", ".join([f"{f[0]}: {f[1]:.2f}" for f in top_features])

            # Generate trading recommendation
            recommendation = "📈 Consider Buying on Dips" if technical_trend == "Bullish" and \
                                                            (not model_result or latest.get('model_prediction') == 2) else \
                "📉 Potential Short Signal" if technical_trend == "Bearish" and \
                                              (not model_result or latest.get('model_prediction') == 0) else \
                    "⚖️ Wait for confirmation"

            # Format the summary line
            summary_line = (
                f"{crypto} Analysis:\n"
                f"  → {trend_info}"
            )

            if scenario_info:
                summary_line += f"\n  → Market Conditions: {' | '.join(scenario_info)}"

            if sentiment_info:
                summary_line += f"\n  → Sentiment: {' | '.join(sentiment_info)}"

            if ml_info:
                summary_line += f"\n  → ML Insights: {' | '.join(ml_info)}"

            summary_line += f"\n  → {recommendation}"

            if feature_insights:
                summary_line += f"\n  → {feature_insights}"

            summary.append(summary_line)

        formatted_summary = "\n\n".join(summary)
        return formatted_summary

    except Exception as e:
        logging.error(f"⚠️ Error generating market summary: {e}")
        print(f"⚠️ Error generating market summary: {e}")
        return "⚠️ Error generating market summary. Check logs for details."


def backtest_trading_strategy(analyzed_cryptos):
    """Backtest and compare ML-assisted trading vs. indicator-only trading."""
    conn = sqlite3.connect(db_path)
    
    # Only get data for the cryptocurrencies we analyzed in this run
    placeholders = ','.join(['?'] * len(analyzed_cryptos))
    query = f"SELECT * FROM indicators WHERE crypto IN ({placeholders})"
    df = pd.read_sql(query, conn, params=analyzed_cryptos)
    conn.close()

    if df.empty:
        print("⚠️ No data found for the selected cryptocurrencies")
        return []

    # Convert all numeric columns to float
    numeric_columns = ['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi',
                      'fear_and_greed_index', 'social_media_sentiment', 'volatility_score', 
                      'liquidity_score', 'model_prediction', 'model_confidence']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Get the list of cryptocurrencies that were actually analyzed
    selected_assets = df['crypto'].unique().tolist()
    print(f"\nBacktesting for analyzed cryptocurrencies: {selected_assets}")
    
    # Filter to only include the cryptocurrencies we analyzed
    df = df[df['crypto'].isin(selected_assets)]

    results_list = []  # Initialize as a list
    for crypto in selected_assets:
        print(f"\nProcessing {crypto}...")
        crypto_df = df[df['crypto'] == crypto].copy()

        # Extract features and target
        feature_columns = ['macd', 'rsi', 'bollinger_upper', 'bollinger_lower', 'stochastic_rsi',
                           'fear_and_greed_index', 'social_media_sentiment', 'volatility_score', 'liquidity_score']

        # Ensure all required columns exist
        missing_columns = [col for col in feature_columns if col not in crypto_df.columns]
        if missing_columns:
            print(f"⚠️ Missing columns for {crypto}: {missing_columns}")
            continue

        # Drop any rows with NaN values
        crypto_df = crypto_df.dropna(subset=feature_columns)

        if len(crypto_df) == 0:
            print(f"⚠️ No valid data after cleaning for {crypto}")
            continue

        # Convert to numpy array and ensure float type
        features = crypto_df[feature_columns].values.astype(float)
        target = crypto_df['market_condition'].values.astype(float)

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)
        target = np.nan_to_num(target, nan=1)  # Default to neutral for missing values

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train and evaluate both models
        model_results = train_and_evaluate_models(X_train, y_train)

        if model_results is None:
            print(f"⚠️ Failed to train models for {crypto}")
            continue

        # Compare models
        comparison = compare_models(model_results, X_test, y_test)
        
        if comparison:
            print(f"\nResults for {crypto}:")
            print("\nModel Comparison:")
            for model_name, metrics in comparison.items():
                print(f"\n{model_name.upper()}:")
                print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
                print(f"Precision: {metrics['precision']*100:.2f}%")
                print(f"Recall: {metrics['recall']*100:.2f}%")
                print(f"F1 Score: {metrics['f1']*100:.2f}%")
                print(f"Average Confidence: {metrics['avg_confidence']*100:.2f}%")
                print(f"Cross-validation: {metrics['cv_mean']*100:.2f}% (±{metrics['cv_std']*100:.2f}%)")

        # Generate SHAP explanations for both models
        print("\nSHAP Explanations:")
        for model_name, results in model_results.items():
            try:
                feature_df = pd.DataFrame(features, columns=feature_columns)
                if model_name == 'rf':
                    explainer = shap.TreeExplainer(results['model'])
                else:  # xgb
                    explainer = shap.TreeExplainer(results['model'], feature_perturbation="auto")
                
                shap_values = explainer.shap_values(feature_df)

                if shap_values is not None and len(shap_values) > 0:
                    print(f"\n{model_name.upper()} SHAP Summary:")
                    plt.figure(figsize=(10, 8))
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[1], feature_df, plot_type="bar", show=False)
                    else:
                        shap.summary_plot(shap_values, feature_df, plot_type="bar", show=False)
                    plt.tight_layout()
                    plt.savefig(f'shap_summary_{crypto}_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                    plt.close()
            except Exception as e:
                print(f"⚠️ Error generating SHAP explanation for {model_name}: {e}")

        # Store results with feature importance
        results_list.append({
            "Asset": crypto,
            "Model Accuracy (%)": comparison['rf']['accuracy'] * 100,  # Use RF accuracy as primary
            "Model Confidence": float(np.mean(model_results['rf']['confidence'])),
            "Model Comparison": comparison,
            "Feature Importance": {
                "rf": model_results['rf']['feature_importance'],
                "xgb": model_results['xgb']['feature_importance']
            }
        })

        # Update database with predictions and confidence scores
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Update predictions for both models
        rf_predictions = model_results['rf']['model'].predict(X_test)
        xgb_predictions = model_results['xgb']['model'].predict(X_test)
        
        for i, (rf_pred, xgb_pred) in enumerate(zip(rf_predictions, xgb_predictions)):
            date = crypto_df.iloc[i]['date']
            cursor.execute("""
                UPDATE indicators 
                SET model_prediction = ?, model_confidence = ?, feature_importance = ?
                WHERE crypto = ? AND date = ?
            """, (
                float(rf_pred),  # Using RF predictions as primary
                float(model_results['rf']['confidence'][i]),
                json.dumps({
                    'rf': model_results['rf']['feature_importance'],
                    'xgb': model_results['xgb']['feature_importance']
                }),
                crypto,
                date.strftime('%Y-%m-%d')
            ))

        conn.commit()
        conn.close()

    return results_list  # Return the list of results


if __name__ == "__main__":
    # Get user input for cryptocurrencies
    cryptos = get_user_assets()
    
    # Clear old data for these cryptocurrencies
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for crypto in cryptos:
        print(f"Clearing old data for {crypto}...")
        cursor.execute("DELETE FROM indicators WHERE crypto = ?", (crypto,))
    conn.commit()
    conn.close()
    
    market_scenario = get_user_scenario()

    # Process each cryptocurrency with delay to respect rate limits
    for i, crypto in enumerate(cryptos):
        print(f"\n📊 Processing {crypto} ({i+1}/{len(cryptos)})...")

        df = fetch_data(crypto)
        if not df.empty:
            df = calculate_indicators(df)
            df = add_sentiment_to_dataframe(df, crypto)
            df = apply_market_scenario(df, market_scenario)
            store_results(crypto, df)

    print("\n📊 Market Summary Report:\n")
    model_results = backtest_trading_strategy(cryptos)  # Pass the analyzed cryptocurrencies
    print(generate_market_summary(cryptos, market_scenario, model_results))

    generate_commit_hash()