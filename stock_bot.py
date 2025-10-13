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

from datetime import datetime, date
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
        days_since_halving = (date.fromisoformat(data['date']) - 
                            date(2024, 4, 19)).days
        prompt = f"""
        Project: {PROJECT_NAME}
        Analyze Bitcoin's position in the 4-year halving cycle (accumulation, growth, bubble, crash).
        Current date: {data['date']}, Days since halving (Apr 19, 2024): {days_since_halving}
        Price: ${float(data['current_price']):.2f}, RSI: {float(data['rsi']):.1f}, SMA: ${float(data['sma_50']):.2f}
        Dominance: {data['btc_dominance'] or 'N/A'}%, Puell Multiple: {float(data['puell_multiple']):.2f}
        Output JSON: {{"phase": "growth", "analysis": "Post-halving rally likely, but watch for bubble signs."}}
        """
    elif ticker == '^GSPC':
        prompt = f"""
        Project: {PROJECT_NAME}
        Analyze S&P 500's position in the 4-year presidential election cycle (year 1-4, Wyckoff phases).
        Current date: {data['date']}, Year: {data['election_year']}
        Price: {float(data['current_price']):.2f}, RSI: {float(data['rsi']):.1f}, SMA: {float(data['sma_50']):.2f}
        VIX: {float(data['vix']):.2f}, Yield Curve: {float(data['yield_curve']):.2f}%
        Output JSON: {{"phase": "year 1", "analysis": "Early cycle volatility expected."}}
        """
    else:
        return {"phase": "N/A", "analysis": "Cycle analysis not applicable"}
    try:
        return json.loads(query_grok(prompt, model="grok-4"))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in cycle analysis for {ticker}")
        return {"phase": "N/A", "analysis": "Failed to compute cycle analysis"}

def fetch_btc_dominance():
    for attempt in range(3):
        try:
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()['data']['market_cap_percentage']['btc']
        except Exception as e:
            logger.error(f"Retry {attempt+1}/3 for BTC dominance: {e}")
            time.sleep(2)
    return None

def approximate_puell_multiple(btc_data):
    try:
        daily_revenue = btc_data['close'] * 6.25 / 144
        ma365_revenue = daily_revenue.rolling(window=365).mean()
        return float(daily_revenue.iloc[-1] / ma365_revenue.iloc[-1])
    except Exception as e:
        logger.error(f"Error computing Puell Multiple: {e}")
        return np.nan

def get_stock_data(ticker, retries=5):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y', auto_adjust=False)
            if hist.empty:
                logger.error(f"No data for {ticker}")
                time.sleep(3)
                continue
            hist.columns = [col.lower() for col in hist.columns]
            stock_df = StockDataFrame.retype(hist)
            sma_50 = float(hist['close'].rolling(window=50).mean().iloc[-1])
            current_price = float(stock.fast_info.get('last_price', hist['close'].iloc[-1]))
            rsi = float(stock_df['rsi_14'].iloc[-1])
            bb_lower = float(stock_df['boll_lb'].iloc[-1])
            bb_upper = float(stock_df['boll_ub'].iloc[-1])
            current_volume = float(hist['volume'].iloc[-1])
            avg_volume = float(hist['volume'].rolling(window=20).mean().iloc[-1])
            volatility = float(hist['close'].pct_change().rolling(window=20).std().iloc[-1] * 100)
            threshold = max(0.90, 0.95 - (volatility * 0.1))
            high_52w = float(hist['high'].max())
            low_52w = float(hist['low'].min())
            positions = load_positions()
            avg_cost = float(positions.get(ticker, {}).get('avg_cost', 0))
            hl2 = (hist['high'] + hist['low']) / 2
            v1 = smma(hl2, 15).iloc[-1]
            m1 = smma(hl2, 19).iloc[-1]
            m2 = smma(hl2, 25).iloc[-1]
            v2 = smma(hl2, 29).iloc[-1]
            p2 = ((v1 < m1) != (v1 < v2)) or ((m2 < v2) != (v1 < v2))
            p3 = (not p2) and (v1 < v2)
            p1 = (not p2) and (not p3)
            larsson_line = 'gold' if p1 else 'silver' if p2 else 'blue'
            larsson_analysis = get_larsson_analysis(ticker, larsson_line, hist)
            sentiment = get_sentiment_score(ticker)
            news = get_news_summary(ticker)
            forecast = forecast_rebound(hist, ticker)
            data = {
                'date': date.today().isoformat(),
                'ticker': ticker,
                'current_price': current_price,
                'sma_50': sma_50,
                'rsi': rsi,
                'bb_lower': bb_lower,
                'bb_upper': bb_upper,
                'volume_ratio': current_volume / avg_volume if avg_volume else 1,
                'threshold': threshold,
                'high_52w': high_52w,
                'low_52w': low_52w,
                'avg_cost': avg_cost,
                'dca_amount': DCA_AMOUNT,
                'percent_below_sma': (current_price - sma_50) / sma_50 * 100,
                'percent_from_high': (current_price - high_52w) / high_52w * 100,
                'percent_from_low': (current_price - low_52w) / low_52w * 100,
                'sentiment_score': sentiment.get('score', 0.5),
                'sentiment_summary': sentiment.get('summary', 'Neutral'),
                'news_summary': news.get('summary', 'No news'),
                'news_risk_level': news.get('risk_level', 'low'),
                'rebound_prob': forecast.get('rebound_prob', 0.5),
                'forecast_price': forecast.get('forecast_price', current_price),
                'forecast_confidence': forecast.get('confidence', 'medium'),
                'larsson_line': larsson_line,
                'larsson_analysis': larsson_analysis
            }
            if ticker == 'BTC-USD':
                data['btc_dominance'] = fetch_btc_dominance()
                data['puell_multiple'] = approximate_puell_multiple(hist)
            if ticker == '^GSPC':
                vix = float(yf.download('^VIX', period='1y', progress=False, auto_adjust=False)['Close'].iloc[-1])
                t10y = float(yf.download('^TNX', period='1mo', progress=False, auto_adjust=False)['Close'].iloc[-1])
                t2y = float(yf.download('^IRX', period='1mo', progress=False, auto_adjust=False)['Close'].iloc[-1] / 100)
                data['vix'] = vix
                data['yield_curve'] = t10y - t2y
                data['election_year'] = str(int(data['date'][:4]) % 4 + 1)
            cycle_analysis = get_cycle_analysis(ticker, data)
            data['cycle_phase'] = cycle_analysis.get('phase', 'N/A')
            data['cycle_analysis'] = cycle_analysis.get('analysis', 'N/A')
            buy_score_data = get_buy_score(data)
            sell_score_data = get_sell_score(data)
            data.update(buy_score_data)
            data.update(sell_score_data)
            return data
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            time.sleep(3)
    return None

def smma(series, length):
    smma_val = pd.Series(np.nan, index=series.index)
    smma_val.iloc[length-1] = series.iloc[:length].mean()
    for i in range(length, len(series)):
        smma_val.iloc[i] = (smma_val.iloc[i-1] * (length - 1) + series.iloc[i]) / length
    return smma_val

def get_buy_score(data):
    name = get_ticker_name(data['ticker'])
    prompt = f"""
    Evaluate DCA in (buy) opportunity for {name} based on:
    - Current: ${float(data['current_price']):.2f}, 50-day SMA: ${float(data['sma_50']):.2f} ({data['percent_below_sma']:.1f}% below)
    - RSI: {float(data['rsi']):.1f}, Below BB: {data['current_price'] <= data['bb_lower']}
    - Volatility Threshold: {data['threshold']:.2f}, % from 52w High: {data['percent_from_high']:.1f}%
    - Sentiment Score: {data['sentiment_score']:.2f}
    - Larsson Line Trend: {data['larsson_line']} (Analysis: {data['larsson_analysis']})
    - Suggested Amount: ${DCA_AMOUNT:.2f}
    Output JSON: {{"buy_score": 85, "recommendation": "Strong Buy", "reasoning": "Oversold with positive momentum"}}
    """
    try:
        return json.loads(query_grok(prompt, model="grok-4"))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in get_buy_score for {data['ticker']}")
        return {"buy_score": 50, "recommendation": "Neutral", "reasoning": "Failed to compute buy score"}

def get_sell_score(data):
    name = get_ticker_name(data['ticker'])
    positions = load_positions()
    avg_cost = positions.get(data['ticker'], {}).get('avg_cost', 0)
    profit_pct = (data['current_price'] - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
    percent_above_sma = (data['current_price'] - data['sma_50']) / data['sma_50'] * 100
    prompt = f"""
    Evaluate DCA out (sell) opportunity for {name} based on:
    - Current: ${float(data['current_price']):.2f}, 50-day SMA: ${float(data['sma_50']):.2f} ({percent_above_sma:.1f}% above)
    - Avg Cost: ${float(avg_cost):.2f} ({profit_pct:.1f}% profit)
    - RSI: {float(data['rsi']):.1f}, Above BB: {data['current_price'] >= data['bb_upper']}
    - Volatility Threshold: {data['threshold']:.2f}, % from 52w Low: {data['percent_from_low']:.1f}%
    - Sentiment Score: {data['sentiment_score']:.2f}
    - Larsson Line Trend: {data['larsson_line']} (Analysis: {data['larsson_analysis']})
    - Suggested Amount: ${DCA_AMOUNT:.2f}
    Output JSON: {{"sell_score": 85, "recommendation": "Strong Take Profit", "reasoning": "Overbought, lock in gains"}}
    """
    try:
        return json.loads(query_grok(prompt, model="grok-4"))
    except json.JSONDecodeError:
        logger.error(f"JSON decode error in get_sell_score for {data['ticker']}")
        return {"sell_score": 50, "recommendation": "Neutral", "reasoning": "Failed to compute sell score"}

def is_dca_opportunity(data):
    positions = load_positions()
    has_position = data['ticker'] in positions and positions[data['ticker']]['quantity'] > 0
    buy_weights = {
        'sma': 0.25, 'rsi': 0.2, 'bb': 0.1, 'volume': 0.1,
        'sentiment': 0.1, 'news_risk': 0.1, 'rebound': 0.1, 'larsson': 0.1
    }
    rsi_threshold_buy = 35 if data['ticker'] in ['MSTR', 'TSLA'] else 30
    volume_threshold_buy = 1.2 if data['ticker'] in ['MSTR', 'TSLA'] else 1.5
    buy_conditions = {
        'sma': data['current_price'] <= data['sma_50'] * data['threshold'],
        'rsi': data['rsi'] < rsi_threshold_buy,
        'bb': data['current_price'] <= data['bb_lower'],
        'volume': data['volume_ratio'] > volume_threshold_buy,
        'sentiment': data['sentiment_score'] >= 0.7,
        'news_risk': data['news_risk_level'] in ['low', 'medium'],
        'rebound': data['rebound_prob'] >= 0.6,
        'larsson': data['larsson_line'] == 'gold'
    }
    buy_combined_score = sum(buy_conditions[key] * weight * 100 for key, weight in buy_weights.items())
    
    sell_weights = {
        'sma_sell': 0.2, 'rsi_sell': 0.2, 'bb_sell': 0.15, 'profit': 0.15,
        'sentiment_sell': 0.1, 'news_risk_sell': 0.1, 'larsson_sell': 0.1
    }
    rsi_threshold_sell = 70 if data['ticker'] in ['MSTR', 'TSLA'] else 65
    profit_threshold = 15
    sell_conditions = {
        'sma_sell': data['current_price'] >= data['sma_50'] * 1.05,
        'rsi_sell': data['rsi'] > rsi_threshold_sell,
        'bb_sell': data['current_price'] >= data['bb_upper'],
        'profit': ((data['current_price'] - data['avg_cost']) / data['avg_cost'] * 100) >= profit_threshold if data['avg_cost'] > 0 else False,
        'sentiment_sell': data['sentiment_score'] <= 0.3,
        'news_risk_sell': data['news_risk_level'] in ['high', 'medium'],
        'larsson_sell': data['larsson_line'] == 'blue'
    }
    sell_combined_score = sum(sell_conditions[key] * weight * 100 for key, weight in sell_weights.items()) if has_position else 0
    
    data['buy_combined_score'] = buy_combined_score
    data['sell_combined_score'] = sell_combined_score
    data['has_position'] = has_position
    is_buy_opportunity = buy_combined_score >= 70 and data['buy_score'] >= 70 and data['percent_from_high'] > -20
    is_sell_opportunity = has_position and sell_combined_score >= 70 and data['sell_score'] >= 70
    data['opportunity_type'] = 'buy' if is_buy_opportunity else 'sell' if is_sell_opportunity else 'hold'
    
    if data['opportunity_type'] == 'buy':
        logger.info(f"{data['ticker']} Buy Opportunity: Buy Score {buy_combined_score:.1f}, DCA Score {data['buy_score']}")
    elif data['opportunity_type'] == 'sell':
        logger.info(f"{data['ticker']} Sell Opportunity: Sell Score {sell_combined_score:.1f}, DCA Score {data['sell_score']}")
    else:
        logger.info(f"{data['ticker']} Hold: Buy {buy_combined_score:.1f}, Sell {sell_combined_score:.1f}")
    
    return data['opportunity_type'] != 'hold'

def save_to_csv(all_data):
    try:
        csv_data = []
        for data in all_data:
            if data:
                csv_data.append({
                    'Date': data['date'],
                    'Ticker': data['ticker'],
                    'Price': data['current_price'],
                    'RSI': data['rsi'],
                    'SMA_50': data['sma_50'],
                    'Volume_Ratio': data['volume_ratio'],
                    'Sentiment_Score': data['sentiment_score'],
                    'Cycle_Phase': data['cycle_phase'],
                    'Buy_Score': data['buy_combined_score'],
                    'Sell_Score': data['sell_combined_score'],
                    'Opportunity': data['opportunity_type']
                })
        df = pd.DataFrame(csv_data)
        if os.path.exists(CSV_FILE):
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
        else:
            df.to_csv(CSV_FILE, mode='w', header=True, index=False)
        logger.info("Data saved to CSV")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")

def send_email(all_data, alerts, tickers=STOCKS):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = f"{PROJECT_NAME}: Daily DCA and Cycle Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    body = f"{PROJECT_NAME} Analysis for {', '.join(get_ticker_name(t) for t in tickers)}:\n\n"
    for data, ticker in zip(all_data, tickers):
        if data is None or not isinstance(data, dict):
            body += f"{get_ticker_name(ticker)}: Data unavailable\n\n"
            continue
        name = get_ticker_name(ticker)
        opp_type = data.get('opportunity_type', 'hold').title()
        status = f"{opp_type} Opportunity" if opp_type != 'hold' else "Hold"
        avg_cost = data.get('avg_cost', 0)
        profit_pct = (data['current_price'] - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
        body += f"""
        {name} ({ticker}) ({status}):
        - Price: ${float(data['current_price']):.2f} ({data['percent_below_sma']:.1f}% below SMA: ${float(data['sma_50']):.2f})
        - Avg Cost: ${float(avg_cost):.2f} ({profit_pct:.1f}% {'profit' if profit_pct > 0 else 'loss'})
        - RSI: {float(data['rsi']):.1f}, Volume: {data['volume_ratio']:.1f}x avg
        - Larsson Line: {data['larsson_line']} (Analysis: {data['larsson_analysis']})
        """
        if ticker in ['BTC-USD', '^GSPC']:
            body += f"- Cycle Phase: {data['cycle_phase']} ({data['cycle_analysis']})\n"
        if ticker == 'BTC-USD':
            body += f"- Dominance: {data['btc_dominance'] or 'N/A'}%, Puell Multiple: {float(data['puell_multiple']):.2f}\n"
        if ticker == '^GSPC':
            body += f"- VIX: {float(data['vix']):.2f}, Yield Curve: {float(data['yield_curve']):.2f}%\n"
        if data['opportunity_type'] == 'buy':
            body += f"""
        - Buy Score: {data['buy_combined_score']:.1f}/100 ({data['recommendation']}: {data['reasoning']})
            """
        elif data['opportunity_type'] == 'sell':
            body += f"""
        - Sell Score: {data['sell_combined_score']:.1f}/100 ({data['recommendation']}: {data['reasoning']})
            """
        body += f"""
        - Sentiment: {data['sentiment_summary']} (Score: {data['sentiment_score']:.2f})
        - News: {data['news_summary']} (Risk: {data['news_risk_level']})
        - Forecast: ${float(data['forecast_price']):.2f} ({data['rebound_prob']*100:.1f}% chance, {data['forecast_confidence']})
        - Suggested DCA: ${float(data['dca_amount']):.2f} ({opp_type})
        \n"""
    body += "Not financial advice. DYOR."
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        logger.info(f"Email sent with analysis for {len(all_data)} assets, {len(alerts)} opportunities")
    except Exception as e:
        logger.error(f"Email error: {e}")

def all_data_to_markdown(all_data):
    markdown = ""
    for data in all_data:
        if data is None or not isinstance(data, dict):
            markdown += f"{get_ticker_name(data['ticker'] if data else 'Unknown')}: Data unavailable\n\n"
            continue
        name = get_ticker_name(data['ticker'])
        opp_type = data.get('opportunity_type', 'hold').title()
        status = f"{opp_type} Opportunity" if opp_type != 'hold' else "Hold"
        avg_cost = data.get('avg_cost', 0)
        profit_pct = (data['current_price'] - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0
        markdown += f"""
## {name} ({data['ticker']}) ({status})
- **Price**: ${float(data['current_price']):.2f} ({data['percent_below_sma']:.1f}% below SMA: ${float(data['sma_50']):.2f})
- **Avg Cost**: ${float(avg_cost):.2f} ({profit_pct:.1f}% {'profit' if profit_pct > 0 else 'loss'})
- **RSI**: {float(data['rsi']):.1f}, **Volume**: {data['volume_ratio']:.1f}x avg
- **Larsson Line**: {data['larsson_line']} ({data['larsson_analysis']})
"""
        if data['ticker'] in ['BTC-USD', '^GSPC']:
            markdown += f"- **Cycle Phase**: {data['cycle_phase']} ({data['cycle_analysis']})\n"
        if data['ticker'] == 'BTC-USD':
            markdown += f"- **Dominance**: {data['btc_dominance'] or 'N/A'}%, **Puell Multiple**: {float(data['puell_multiple']):.2f}\n"
        if data['ticker'] == '^GSPC':
            markdown += f"- **VIX**: {float(data['vix']):.2f}, **Yield Curve**: {float(data['yield_curve']):.2f}%\n"
        if data['opportunity_type'] == 'buy':
            markdown += f"- **Buy Score**: {data['buy_combined_score']:.1f}/100 ({data['recommendation']}: {data['reasoning']})\n"
        elif data['opportunity_type'] == 'sell':
            markdown += f"- **Sell Score**: {data['sell_combined_score']:.1f}/100 ({data['recommendation']}: {data['reasoning']})\n"
        markdown += f"""
- **Sentiment**: {data['sentiment_summary']} (Score: {data['sentiment_score']:.2f})
- **News**: {data['news_summary']} (Risk: {data['news_risk_level']})
- **Forecast**: ${float(data['forecast_price']):.2f} ({data['rebound_prob']*100:.1f}% chance, {data['forecast_confidence']})
- **Suggested DCA**: ${float(data['dca_amount']):.2f} ({opp_type})
\n"""
    return markdown

def main():
    logger.info("Starting main function")
    alerts = []
    all_data = []
    for ticker in STOCKS:
        data = get_stock_data(ticker)
        all_data.append(data)
        if data and is_dca_opportunity(data):
            opp_type = data['opportunity_type']
            if opp_type == 'buy':
                update_position(ticker, 'buy', data['current_price'])
            elif opp_type == 'sell':
                update_position(ticker, 'sell', data['current_price'])
            alerts.append(data)
    save_to_csv(all_data)
    md_file = f"{PROJECT_NAME}_{date.today().isoformat()}.md"
    with open(md_file, 'w') as f:
        f.write(f"# {PROJECT_NAME} - Daily DCA and Cycle Analysis - {date.today().isoformat()}\n\n{all_data_to_markdown(all_data)}")
    send_email(all_data, alerts, STOCKS)

if __name__ == "__main__":
    main()
