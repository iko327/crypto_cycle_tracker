# Stock Bot with 4-Year Market Cycle Analysis
#
# Features:
# - Analyzes MSTR, TSLA, NVDA, SOL-USD, BTC-USD for DCA opportunities
# - Adds 4-year cycle analysis for BTC-USD (halving) and ^GSPC (presidential)
# - Uses Grok API for sentiment, news, forecasts, and cycle analysis
# - Saves results to CSV and Markdown, commits to repo
# - Uses project name 'crypto_cycle_tracking' in outputs
#
# Requirements: See requirements.txt
# Setup: Configure EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD, GROK_API_KEY in GitHub Secrets

import os
import datetime
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from stockstats import StockDataFrame
import yfinance as yf
from openai import OpenAI
import json
import requests
import pandas as pd
import numpy as np
import time
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Configure logging
LOG_FILE = 'stock_bot.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting stock_bot.py")

# Configuration
STOCKS = ['MSTR', 'TSLA', 'NVDA', 'SOL-USD', 'BTC-USD', '^GSPC']
TICKER_NAMES = {
    'MSTR': 'MicroStrategy',
    'TSLA': 'Tesla',
    'NVDA': 'Nvidia',
    'SOL-USD': 'Solana',
    'BTC-USD': 'Bitcoin',
    '^GSPC': 'S&P 500'
}
DCA_AMOUNT = 100
POSITIONS_FILE = 'positions.json'
CSV_FILE = 'market_data_history.csv'
PROJECT_NAME = 'crypto_cycle_tracking'

# Load environment variables
EMAIL_FROM = os.getenv('EMAIL_FROM')
EMAIL_TO = os.getenv('EMAIL_TO')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
GROK_API_KEY = os.getenv('GROK_API_KEY')
if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD, GROK_API_KEY]):
    logger.error("Missing environment variables")
    raise ValueError("Required environment variables are not set")

# Initialize Grok API client
client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
logger.info("OpenAI client initialized")

def load_positions():
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return {}

def save_positions(positions):
    try:
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(positions, f, indent=4)
        logger.info("Positions saved")
    except Exception as e:
        logger.error(f"Error saving positions: {e}")

def update_position(ticker, action, price, amount=DCA_AMOUNT):
    positions = load_positions()
    if ticker not in positions:
        positions[ticker] = {'quantity': 0, 'total_cost': 0}
    
    pos = positions[ticker]
    if action == 'buy':
        quantity_bought = amount / price
        pos['quantity'] += quantity_bought
        pos['total_cost'] += amount
        pos['avg_cost'] = pos['total_cost'] / pos['quantity'] if pos['quantity'] > 0 else 0
        logger.info(f"Bought {quantity_bought:.6f} {ticker} at ${price:.2f}")
    elif action == 'sell':
        if pos['quantity'] > 0:
            quantity_sold = min(amount / price, pos['quantity'])
            pos['quantity'] -= quantity_sold
            proceeds = quantity_sold * price
            pos['total_cost'] -= (quantity_sold / (pos['total_cost'] / pos['avg_cost'] if pos['total_cost'] > 0 else 1)) * pos['avg_cost']
            pos['avg_cost'] = pos['total_cost'] / pos['quantity'] if pos['quantity'] > 0 else 0
            logger.info(f"Sold {quantity_sold:.6f} {ticker} at ${price:.2f}")
        else:
            logger.warning(f"No position to sell for {ticker}")
    
    positions[ticker] = pos
    save_positions(positions)

def get_ticker_name(ticker):
    return TICKER_NAMES.get(ticker, ticker)

def query_grok(prompt, model="grok-3-mini"):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Grok API retry {attempt+1}/3 failed: {e}")
            time.sleep(2)
    logger.error("Grok API failed after retries")
    return "{}"

def get_sentiment_score(ticker):
    name = get_ticker_name(ticker)
    prompt = f"""
    Analyze recent news and X posts about {name} (last 24 hours).
    Provide a sentiment score (0-1, 1 is strongly positive) and a 1-sentence summary.
    Output as JSON: {{"score": 0.85, "summary": "Positive EV demand outlook"}}
    """
    try:
        return json.loads(query_grok(prompt))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in sentiment for {ticker}")
        return {"score": 0.5, "summary": "Neutral sentiment"}

def get_news_summary(ticker):
    name = get_ticker_name(ticker)
    prompt = f"""
    Summarize top 3 news items for {name} today using real-time search.
    Output JSON: {{"summary": "Bullet 1\\nBullet 2", "risk_level": "low"}}
    """
    try:
        return json.loads(query_grok(prompt))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in news for {ticker}")
        return {"summary": "No news available", "risk_level": "low"}

def forecast_rebound(hist_df, ticker):
    try:
        csv_data = hist_df['close'].tail(10).to_csv()
        prompt = f"""
        Analyze 10-day close prices for {ticker}: {csv_data}
        Forecast >5% rebound in 3-5 days.
        Output JSON: {{"rebound_prob": 0.75, "forecast_price": 250.00, "confidence": "high"}}
        """
        return json.loads(query_grok(prompt))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in forecast for {ticker}")
        return {"rebound_prob": 0.5, "forecast_price": 0.0, "confidence": "low"}

def get_larsson_analysis(ticker, larsson_line, hist_df):
    csv_data = hist_df['close'].tail(10).to_csv()
    prompt = f"""
    Analyze Larsson Line trend for {ticker}: {larsson_line} (gold=uptrend, blue=downtrend, silver=neutral).
    Use 10-day closes: {csv_data}.
    Output JSON: {{"analysis": "Gold indicates uptrend, favoring buys on dips."}}
    """
    try:
        return json.loads(query_grok(prompt))['analysis']
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in Larsson analysis for {ticker}")
        return "Neutral trend analysis"

def get_cycle_analysis(ticker, data):
    name = get_ticker_name(ticker)
    if ticker == 'BTC-USD':
        days_since_halving = (datetime.date.fromisoformat(data['date']) - 
                            datetime.date(2024, 4, 19)).days
        prompt = f"""
        Project: {PROJECT_NAME}
        Analyze Bitcoin's position in the 4-year halving cycle (accumulation, growth, bubble, crash).
        Current date: {data['date']}, Days since halving (Apr 19, 2024): {days_since_halving}
        Price: ${data['current_price']:.2f}, RSI: {data['rsi']:.1f}, SMA: ${data['sma_50']:.2f}
        Dominance: {data['btc_dominance'] or 'N/A'}%, Puell Multiple: {data['puell_multiple']:.2f}
        Output JSON: {{"phase": "growth
