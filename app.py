import streamlit as st
import feedparser
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from cachetools import TTLCache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import os


DEBUG = False 
# API Keys


PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"

# Perplexity API
def perplexity_generate(prompt, model="sonar-pro", temperature=0.7, max_tokens=1024):
    prompt = (prompt or "").strip()
    if not prompt:
        st.error("Perplexity API error: Tried to send empty prompt.")
        return "Error: Prompt is empty."

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful financial assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if DEBUG:
        print("\n--- Perplexity API Request ---")
        print("URL:", PERPLEXITY_API_URL)
        print("Headers:", headers)
        print("Payload:", json.dumps(payload, indent=2))

    try:
        response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload)

        if DEBUG:
            print("--- Perplexity API Response ---")
            print("Status Code:", response.status_code)
            print("Response Text:", response.text)

        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        try:
            error_msg = response.json()
        except Exception:
            error_msg = response.text
        st.error(f"Perplexity API error ({response.status_code}): {error_msg}")
        raise e
    except Exception as e:
        st.error(f"Unexpected error calling Perplexity API: {str(e)}")
        return "Error: Failed to call Perplexity API."

# Session Initialization 
def init_session():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = ["AAPL", "MSFT"]
    if 'risk_profile' not in st.session_state:
        st.session_state.risk_profile = "Medium"
    if 'query_logs' not in st.session_state:
        st.session_state.query_logs = []
    if 'admin_password' not in st.session_state:
        st.session_state.admin_password = ""

load_dotenv()

cache = TTLCache(maxsize=100, ttl=3600)
vader = SentimentIntensityAnalyzer()
finbert = pipeline("text-classification", model="yiyanghkust/finbert-tone")

# Fund Data 
FUND_DATA = {
    "Large Cap": [
        {"name": "ICICI Prudential Bluechip Fund", "returns": 12.5, "risk": "Low", "allocation": 30},
        {"name": "SBI Bluechip Fund", "returns": 11.8, "risk": "Low", "allocation": 25}
    ],
    "Mid Cap": [
        {"name": "Axis Midcap Fund", "returns": 15.2, "risk": "Medium", "allocation": 20},
        {"name": "Kotak Emerging Equity Fund", "returns": 16.1, "risk": "Medium", "allocation": 15}
    ],
    "Small Cap": [
        {"name": "Nippon India Small Cap Fund", "returns": 18.7, "risk": "High", "allocation": 10},
        {"name": "HDFC Small Cap Fund", "returns": 17.9, "risk": "High", "allocation": 5}
    ],
    "Sectoral": [
        {"name": "Tata Digital India Fund", "returns": 22.3, "risk": "Very High", "allocation": 5},
        {"name": "SBI Healthcare Opportunities Fund", "returns": 19.8, "risk": "Very High", "allocation": 5}
    ]
}

# Utility Functions 
def format_date(date_str):
    try:
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', ''))
            return dt.strftime("%b %d, %Y")
        return date_str[:10]
    except:
        return date_str

def get_realtime_stock_data(symbol):
    cache_key = f"realtime_{symbol}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API}"
        response = requests.get(url)
        data = response.json().get('Global Quote', {})
        if data and data.get('05. price'):
            result = {
                'symbol': symbol,
                'price': float(data.get('05. price', 0)),
                'change': float(data.get('10. change percent', "0%").rstrip('%')),
                'volume': int(data.get('06. volume', 0)),
                'source': 'Alpha Vantage'
            }
            cache[cache_key] = result
            return result
    except Exception as e:
        st.warning(f"Alpha Vantage failed: {str(e)}")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if not data.empty:
            open_price = data['Open'].iloc[-1]
            close_price = data['Close'].iloc[-1]
            change_pct = 0 if open_price == 0 else round((close_price - open_price) / open_price * 100, 2)
            result = {
                'symbol': symbol,
                'price': round(close_price, 2),
                'change': change_pct,
                'volume': int(data['Volume'].iloc[-1]),
                'source': 'Yahoo Finance'
            }
            cache[cache_key] = result
            return result
    except Exception as e:
        st.error(f"Failed to fetch data for {symbol}: {str(e)}")
    return {
        'symbol': symbol,
        'price': 0,
        'change': 0,
        'volume': 0,
        'source': 'Error'
    }


def get_historical_chart(symbol, period="1y"):
    cache_key = f"historical_{symbol}_{period}"
    if cache_key in cache:
        return cache[cache_key]
    period_map = {
        "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"
    }
    ticker = yf.Ticker(symbol)
    hist_data = ticker.history(period=period_map.get(period, "1y"))
    if hist_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No historical data available")
        return fig
    hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index,
        open=hist_data['Open'],
        high=hist_data['High'],
        low=hist_data['Low'],
        close=hist_data['Close'],
        name="Price"
    ))
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['SMA_20'],
        line=dict(color='orange', width=1),
        name="20-Day SMA"
    ))
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['SMA_50'],
        line=dict(color='blue', width=1),
        name="50-Day SMA"
    ))
    fig.update_layout(
        title=f"{symbol} Historical Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=500
    )
    cache[cache_key] = fig
    return fig


def compare_stocks(symbols, comparison_type, period="1y"):
    cache_key = f"compare_{'_'.join(sorted(symbols))}_{comparison_type}_{period}"
    if cache_key in cache:
        return cache[cache_key]
    period_map = {
        "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
        "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"
    }
    metrics = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(period=period_map.get(period, "1y"))
        if hist_data.empty:
            continue
        start_price = hist_data['Close'].iloc[0]
        end_price = hist_data['Close'].iloc[-1]
        returns = (end_price - start_price) / start_price * 100
        volatility = hist_data['Close'].pct_change().std() * np.sqrt(252)
        metrics.append({
            "Symbol": symbol,
            "Start Price": round(start_price, 2),
            "End Price": round(end_price, 2),
            "Returns (%)": round(returns, 2),
            "Volatility": round(volatility, 4),
            "Avg Volume": f"{int(hist_data['Volume'].mean()):,}"
        })
    metrics_df = pd.DataFrame(metrics).set_index("Symbol")
    fig = go.Figure()
    if comparison_type == "performance":
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(period=period_map.get(period, "1y"))
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['Close'],
                name=symbol,
                mode="lines"
            ))
        fig.update_layout(
            title="Performance Comparison (Normalized)",
            yaxis_title="Price ($)",
            xaxis_title="Date"
        )
    elif comparison_type == "volatility":
        pass  # Placeholder for volatility visualization
    result = {
        "chart": fig,
        "metrics": metrics_df
    }
    cache[cache_key] = result
    return result


def analyze_news_sentiment(symbol):
    cache_key = f"sentiment_{symbol}"
    if cache_key in cache:
        return cache[cache_key]

    # --- Map ticker to company name for better search ---
    company_names = {
        "AAPL": "Apple Inc",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc",
        "TSLA": "Tesla Inc",
        "AMZN": "Amazon.com Inc"
    }
    search_term = company_names.get(symbol.upper(), symbol)

    # --- API URLs ---
    news_api_url = f"https://newsapi.org/v2/everything?q={search_term}&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=10"
    alpha_news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={ALPHA_VANTAGE_API}"

    all_articles = []

    # --- NewsAPI ---
    try:
        response = requests.get(news_api_url)
        newsapi_articles = response.json().get('articles', [])
        print(f"[DEBUG] NewsAPI returned {len(newsapi_articles)} articles for {search_term}")
        for article in newsapi_articles:
            all_articles.append({
                "title": article.get('title', ''),
                "description": article.get('description', ''),
                "source": article.get('source', {}).get('name', 'Unknown'),
                "date": article.get('publishedAt', ''),
                "content": f"{article.get('title', '')}. {article.get('description', '')}"
            })
    except Exception as e:
        st.warning(f"NewsAPI failed: {str(e)}")

    #  Alpha Vantage News Sentiment 
    try:
        response = requests.get(alpha_news_url)
        alpha_articles = response.json().get('feed', [])
        print(f"[DEBUG] Alpha Vantage returned {len(alpha_articles)} articles for {symbol}")
        for article in alpha_articles:
            all_articles.append({
                "title": article.get('title', ''),
                "description": article.get('summary', ''),
                "source": article.get('source', 'Unknown'),
                "date": article.get('time_published', ''),
                "content": f"{article.get('title', '')}. {article.get('summary', '')}"
            })
    except Exception as e:
        st.warning(f"Alpha Vantage news failed: {str(e)}")

    #  Yahoo Finance RSS Fallback 
    if not all_articles:
        try:
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            print(f"[DEBUG] Yahoo Finance RSS returned {len(feed.entries)} articles for {symbol}")
            for entry in feed.entries:
                all_articles.append({
                    "title": entry.title,
                    "description": entry.get("summary", ""),
                    "source": "Yahoo Finance",
                    "date": entry.get("published", ""),
                    "content": f"{entry.title}. {entry.get('summary', '')}"
                })
        except Exception as e:
            st.warning(f"Yahoo Finance RSS failed: {str(e)}")

    #  No news
    if not all_articles:
        print(f"[DEBUG] No news found for {symbol}")
        return {
            "symbol": symbol,
            "compound_score": 0,
            "label": "Neutral",
            "top_news": [],
            "analysis": "No news found for analysis"
        }

    # Sentiment Analysis
    vader_scores = []
    finbert_scores = []
    top_news = []

    for article in all_articles[:10]:
        text = article["content"]
        vader_result = vader.polarity_scores(text)
        vader_scores.append(vader_result['compound'])

        try:
            finbert_result = finbert(text[:512])
            finbert_sentiment = 1 if finbert_result[0]['label'] == 'Positive' else -1
            finbert_scores.append(finbert_sentiment * finbert_result[0]['score'])
        except:
            finbert_scores.append(0)

        top_news.append({
            "title": article["title"],
            "source": article["source"],
            "date": format_date(article["date"]),
            "vader_score": vader_result['compound'],
            "sentiment": "Positive" if vader_result['compound'] > 0.05 else "Negative" if vader_result['compound'] < -0.05 else "Neutral"
        })

    # --- Combine Vader + FinBERT scores ---
    if finbert_scores:
        compound_score = (sum(vader_scores) * 0.4 + sum(finbert_scores) * 0.6) / (len(vader_scores) + len(finbert_scores))
    else:
        compound_score = sum(vader_scores) / len(vader_scores)

    # --- Label determination ---
    if compound_score > 0.2:
        label = "Strongly Positive"
    elif compound_score > 0.05:
        label = "Positive"
    elif compound_score < -0.2:
        label = "Strongly Negative"
    elif compound_score < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    top_news_sorted = sorted(top_news, key=lambda x: abs(x["vader_score"]), reverse=True)

    result = {
        "symbol": symbol,
        "compound_score": compound_score,
        "label": label,
        "top_news": top_news_sorted[:5],
        "analysis": generate_sentiment_analysis_text(compound_score, label, symbol)
    }

    cache[cache_key] = result
    return result


def generate_sentiment_analysis_text(score, label, symbol):
    if score > 0.3:
        return f"Strong positive sentiment detected for {symbol}. Recent news coverage is overwhelmingly favorable, which typically correlates with positive price movement in the short term."
    elif score > 0.1:
        return f"Generally positive sentiment for {symbol}. Market perception is favorable, though not uniformly so. Monitor for continuation of positive news flow."
    elif score > -0.1:
        return f"Neutral sentiment observed for {symbol}. News coverage is mixed with no strong directional bias. Technical factors may dominate price action."
    elif score > -0.3:
        return f"Negative sentiment detected for {symbol}. Caution warranted as recent news coverage has been unfavorable, which may pressure the stock."
    else:
        return f"Strong negative sentiment for {symbol}. Recent news is overwhelmingly negative, suggesting heightened risk of downward price movement."


# ---------- LLM Service Functions (Perplexity) ----------
def generate_comprehensive_report(stocks, query, risk_profile):
    REPORT_TEMPLATE = """
You are a senior financial analyst with 20 years of experience. Provide a comprehensive analysis 
based on the following context and user query. Your response should be professional, insightful, 
and formatted for easy reading with Markdown.


**Context:**
- Analyzing Stocks: {stocks}
- Risk Profile: {risk_profile}
- Current Date: {current_date}


**User Query:** {query}


**Analysis Guidelines:**
1. Start with an executive summary of key findings
2. Provide technical analysis (support/resistance levels, trends)
3. Include fundamental analysis (valuation, growth prospects)
4. Incorporate sentiment analysis from recent news
5. Offer specific recommendations based on risk profile
6. Conclude with actionable advice


**Response:**
"""
    prompt = REPORT_TEMPLATE.format(
        stocks=", ".join(stocks),
        risk_profile=risk_profile,
        current_date=datetime.now().strftime("%B %d, %Y"),
        query=query if query else "No specific query provided."
    )
    return perplexity_generate(prompt)


def generate_chat_response(question, context):
    CHAT_TEMPLATE = """
You are a helpful financial assistant named FinBot. Provide clear, concise answers to financial questions.
Use the following context to inform your response:


**Conversation History:**
{chat_history}


**Current Context:**
- Selected Stocks: {selected_stocks}
- User Risk Profile: {risk_profile}


**User Question:** {question}


Guidelines:
- Be professional but friendly
- Explain financial concepts simply when needed
- Always clarify when providing opinions vs facts
- Highlight risks when making recommendations
- Use bullet points for clarity when appropriate


**Response:**
"""
    formatted_history = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in context.get('chat_history', [])]
    )
    prompt = CHAT_TEMPLATE.format(
        chat_history=formatted_history,
        selected_stocks=", ".join(context.get('selected_stocks', [])),
        risk_profile=context.get('risk_profile', 'Medium'),
        question=question
    )
    return perplexity_generate(prompt)



def get_technical_analysis(stock):
    prompt = f"""
Provide a detailed technical analysis for {stock} stock. Include:
1. Key support and resistance levels
2. Trend analysis (short, medium, long term)
3. Volume analysis
4. Key technical indicators (RSI, MACD, etc.)
5. Overall technical outlook


Format your response with clear headings and bullet points.
"""
    return perplexity_generate(prompt)


# ---------- Recommendations and SIP ----------
def get_investment_recommendations(risk_profile):
    recommendations = {}
    if risk_profile == "Low":
        recommendations["Large Cap Funds"] = FUND_DATA["Large Cap"]
        recommendations["Balanced Funds"] = [
            {"name": "HDFC Balanced Advantage Fund", "returns": 10.9, "risk": "Low", "allocation": 45}
        ]
        recommendations["Debt Funds"] = [
            {"name": "ICICI Prudential Corporate Bond Fund", "returns": 8.2, "risk": "Very Low", "allocation": 25}
        ]
    elif risk_profile == "Medium":
        recommendations["Core Portfolio (Large Cap)"] = FUND_DATA["Large Cap"]
        recommendations["Growth Portfolio (Mid Cap)"] = FUND_DATA["Mid Cap"]
        recommendations["Satellite Portfolio"] = [
            {"name": "Parag Parikh Flexi Cap Fund", "returns": 14.5, "risk": "Medium", "allocation": 15},
            {"name": "Mirae Asset Hybrid Equity Fund", "returns": 13.2, "risk": "Medium", "allocation": 10}
        ]
    else:
        recommendations["Growth Portfolio (Mid Cap)"] = FUND_DATA["Mid Cap"]
        recommendations["Aggressive Growth (Small Cap)"] = FUND_DATA["Small Cap"]
        recommendations["Thematic Bets"] = FUND_DATA["Sectoral"]
        recommendations["International Diversification"] = [
            {"name": "Motilal Oswal NASDAQ 100 ETF", "returns": 16.8, "risk": "High", "allocation": 10},
            {"name": "Franklin India Feeder - US Opportunities", "returns": 15.3, "risk": "High", "allocation": 5}
        ]
    return recommendations


def calculate_sip_projection(monthly_investment, years, risk_profile):
    if risk_profile == "Low":
        expected_return = 0.10
    elif risk_profile == "Medium":
        expected_return = 0.12
    else:
        expected_return = 0.15
    months = years * 12
    future_value = 0
    invested = 0
    projection = {"years": [], "invested": [], "value": []}
    for month in range(1, months + 1):
        invested += monthly_investment
        future_value = (future_value + monthly_investment) * (1 + expected_return/12)
        if month % 12 == 0:
            projection["years"].append(month // 12)
            projection["invested"].append(invested)
            projection["value"].append(future_value)
    return projection


def get_stock_recommendations(risk_profile):
    if risk_profile == "Low":
        return ["HDFC Bank", "Reliance Industries", "Infosys"]
    elif risk_profile == "Medium":
        return ["Bajaj Finance", "Asian Paints", "Titan Company"]
    else:
        return ["Tesla", "Zomato", "Nykaa"]


# ---------- Admin Dashboard ----------
def admin_functions():
    st.header("Administrator Dashboard")
    tab1, tab2, tab3 = st.tabs(["Query Logs", "Data Management", "System"])
    with tab1:
        st.subheader("User Query Logs")
        if st.session_state.get('query_logs'):
            log_df = pd.DataFrame(st.session_state.query_logs)
            st.dataframe(log_df)
            if st.button("Export to CSV"):
                csv = log_df.to_csv(index=False)
                st.download_button(
                    label="Download Logs",
                    data=csv,
                    file_name=f"query_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No query logs available")
    with tab2:
        st.subheader("Data Source Management")
        st.write("Configure data sources and refresh intervals")
        data_sources = {
            "Alpha Vantage": True,
            "Yahoo Finance": True,
            "NewsAPI": True
        }
        for source, status in data_sources.items():
            col1, col2 = st.columns([3, 1])
            col1.write(source)
            col2.checkbox(f"Enable {source}", value=status, key=f"source_{source}")
        if st.button("Refresh All Data"):
            st.session_state.clear()
            st.success("All data sources refreshed")
    with tab3:
        st.subheader("System Configuration")
        st.write("Application settings and monitoring")
        current_memory = 512
        max_memory = st.slider("Max Memory (MB)", 256, 2048, current_memory)
        if st.button("Apply Settings"):
            st.success(f"Settings updated. New memory limit: {max_memory}MB")
        st.divider()
        st.write("**System Health**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Users", 15, "+2 today")
        col2.metric("API Calls", "1,243", "12% increase")
        col3.metric("Response Time", "0.4s", "0.1s improvement")


# --------- Expanded Stock List (US + India) ---------
STOCK_LIST = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Netflix (NFLX)": "NFLX",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "ICICI Bank (ICICIBANK.NS)": "ICICIBANK.NS",
    "Bajaj Finance (BAJFINANCE.NS)": "BAJFINANCE.NS",
    "Asian Paints (ASIANPAINT.NS)": "ASIANPAINT.NS",
}

def main():
    st.set_page_config(
        page_title="Financial AI Advisor Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_session()

    def is_admin():
        return st.session_state.admin_password == st.secrets.get("ADMIN_PASSWORD", "admin123")

    with st.sidebar:
        st.title("Financial AI Agent")
        if is_admin():
            st.success("Admin Mode Active")
            admin_functions()
        else:
            with st.expander("Admin Login"):
                st.session_state.admin_password = st.text_input("Admin Password", type="password")

        # Stock selector
        selected_names = st.multiselect(
            "Select Stocks",
            list(STOCK_LIST.keys()),
            default=[name for name in st.session_state.favorites if name in STOCK_LIST.keys()] or list(STOCK_LIST.keys())[:2]
        )
        selected_stocks = [STOCK_LIST[name] for name in selected_names]

        st.session_state.risk_profile = st.select_slider(
            "Your Risk Appetite",
            options=["Low", "Medium", "High"],
            value=st.session_state.risk_profile
        )
        analysis_period = st.selectbox(
            "Analysis Period",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
            index=4
        )
        if st.button("Update Dashboard"):
            st.rerun()

    st.title("📊 Financial AI Agent Pro")
    st.caption("Comprehensive Market Insights Powered by Generative AI")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Live Market",
        "🧠 AI Insights",
        "🔍 Comparison",
        "💡 Recommendations",
        "💬 Chat"
    ])

    # Tab 1: Live Market
    with tab1:
        st.header("Real-Time Market Dashboard")
        if not selected_stocks:
            st.warning("Please select at least one stock")
        else:
            for stock in selected_stocks:
                st.subheader(stock)
                realtime_data = get_realtime_stock_data(stock)
                delta = float(realtime_data.get('change', 0))
                price_label = f"₹{realtime_data['price']}" if ".NS" in stock else f"${realtime_data['price']}"
                st.metric(
                    label="Price",
                    value=price_label,
                    delta=f"{delta:.2f}%",
                    delta_color="inverse" if delta < 0 else "normal"
                )
                st.plotly_chart(
                    get_historical_chart(stock, analysis_period),
                    use_container_width=True
                )

    # Tab 2: AI Insights
    with tab2:
        st.header("AI-Powered Market Insights")
        st.subheader("📰 News Sentiment Analysis")
        if not selected_stocks:
            st.warning("Please select at least one stock to analyze sentiment.")
        else:
            for stock in selected_stocks:
                st.markdown(f"#### {stock}")
                with st.spinner(f"Analyzing {stock} sentiment..."):
                    sentiment = analyze_news_sentiment(stock)
                    sentiment_score = sentiment["compound_score"]
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=sentiment_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{stock} Sentiment"},
                        gauge={
                            'axis': {'range': [-1, 1]},
                            'steps': [
                                {'range': [-1, -0.5], 'color': "red"},
                                {'range': [-0.5, 0], 'color': "orange"},
                                {'range': [0, 0.5], 'color': "lightgreen"},
                                {'range': [0.5, 1], 'color': "green"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    with st.expander("Top News Headlines"):
                        for news in sentiment['top_news']:
                            st.caption(f"📰 {news['title']}")
                            st.caption(f"_{news['source']} - {news['date']}_")

        st.subheader(" Generative AI Analysis")
        insight_query = st.text_area(
            "Ask for specific analysis (e.g., 'Compare technical indicators for selected stocks')",
            height=100
        )
        if st.button("Generate AI Report") and selected_stocks:
            if not insight_query or not insight_query.strip():
                st.error("Please enter a specific analysis query before submitting.")
            else:
                with st.spinner("Generating comprehensive analysis..."):
                    report = generate_comprehensive_report(
                        stocks=selected_stocks,
                        query=insight_query,
                        risk_profile=st.session_state.risk_profile
                    )
                    st.markdown(report)
                    st.session_state.query_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "stocks": selected_stocks,
                        "query": insight_query,
                        "response": report[:500] + "..."
                    })

    # Tab 3: Comparisons
    with tab3:
        if len(selected_stocks) >= 2:
            st.header("Multi-Stock Comparison")
            comparison_type = st.radio(
                "Comparison Mode",
                ["Performance", "Volatility", "Valuation", "Custom Analysis"],
                horizontal=True
            )
            comparison_data = compare_stocks(
                selected_stocks,
                comparison_type.lower(),
                period=analysis_period
            )
            st.plotly_chart(comparison_data['chart'], use_container_width=True)
            with st.expander("Detailed Comparison Metrics"):
                st.dataframe(comparison_data['metrics'].style.background_gradient(cmap="RdYlGn"))
        else:
            st.warning("Select at least 2 stocks for comparison")

    with tab4:
        st.header("💡 Personalized SIP & Investment Recommendations")
        sip_type = st.selectbox(
            "Select SIP Type",
        ["Short Term (2-5 years)", "Medium Term (5-10 years)", "Long Term (10+ years)"]
        )
        # --- NEW: AI Suggested SIPs using Perplexity API ---
        if st.button("🔍 Get AI SIP Recommendations"):
            with st.spinner("Analyzing best SIP options for you..."):
                prompt = f"""
                You are a financial advisor. Based on the user's risk profile: {st.session_state.risk_profile}
                and SIP duration: {sip_type}, suggest the top 5 SIP mutual funds in India.
                Also provide their typical 5-year CAGR, risk level, and a short reason why each is recommended.
                Format response in bullet points.
                """
                ai_sip_recs = perplexity_generate(prompt)
                st.markdown("### 🧠 AI Suggested SIPs")
                st.markdown(ai_sip_recs)
        st.divider()
        # --- SIP Calculator ---
        st.subheader("📊 SIP Investment Calculator")
        monthly_investment = st.number_input(
           "Monthly Investment (₹)",
            min_value=1000,
            max_value=100000,
            value=5000,
            step=1000
        )
        investment_period = st.slider("Investment Period (years)", 1, 30, 5)
        if st.button("📈 Calculate SIP Projection"):
            projection = calculate_sip_projection(
                monthly_investment, investment_period, st.session_state.risk_profile
            )
            # Real-time profit calculation
            invested_amount = sum([monthly_investment] * investment_period * 12)
            estimated_value = projection['value'][-1]
            profit = estimated_value - invested_amount
            
            st.metric("Total Invested", f"₹{invested_amount:,.0f}")
            st.metric("Estimated Value", f"₹{estimated_value:,.0f}")
            st.metric("Estimated Profit", f"₹{profit:,.0f}")    
            
            # Chart
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=projection['years'],
                y=projection['invested'],
                name="Amount Invested",
                mode="lines",
                line=dict(color="blue")
            ))
            fig.add_trace(go.Scatter(
                x=projection['years'],
                y=projection['value'],
                name="Estimated Value",
                mode="lines",
                line=dict(color="green")
            ), secondary_y=False)
            fig.update_layout(title="SIP Projection Growth")
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        # --- NEW: SIP Chatbot inside Recommendations tab ---
        st.subheader("💬 Ask SIP Assistant")
        with st.form("sip_chat_input"):
            sip_user_input = st.text_input(
                "Ask about SIPs, mutual funds, or investment strategy:",
                 placeholder="e.g., Which SIP is best for 5 years?"
            )
            sip_submitted = st.form_submit_button("Send")
        if sip_submitted and sip_user_input.strip():
            st.session_state.chat_history.append({
                "role": "user",
                "content": sip_user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            with st.spinner("Getting expert financial advice..."):
                ai_response = generate_chat_response(
                    sip_user_input,
                    context={
                        "selected_stocks": selected_stocks,
                        "risk_profile": st.session_state.risk_profile,
                        "chat_history": st.session_state.chat_history[-5:]
                    }
                )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M")
            })
        # Display last few chat messages
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"👤 You ({msg['timestamp']}):")
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(f"🤖 AI ({msg['timestamp']}):")
                    st.markdown(msg["content"])
        st.divider()
        # --- Existing Risk-Based Recommendations ---
        st.subheader("📈 Investment Recommendations")
        recs = get_investment_recommendations(st.session_state.risk_profile)
        for category in recs:
            with st.expander(f"{category}"): 
                for fund in recs[category]:
                    cols = st.columns([3, 1, 1])
                    cols[0].markdown(f"**{fund['name']}**")
                    cols[1].metric("1Y Return", f"{fund['returns']}%")
                    cols[2].metric("Risk", fund['risk'])
                    st.progress(fund['allocation'] / 100)
                    st.caption(f"Suggested allocation: {fund['allocation']}% of portfolio")
                             
    # Tab 5: Chat Assistant
    with tab5:
        st.header("Financial AI Assistant")
        chat_container = st.container()
        with st.form("chat_input"):
            user_input = st.text_input(
                "Ask me anything about finance:",
                placeholder="e.g., Should I invest in tech stocks right now?",
                key="chat_input"
            )
            submitted = st.form_submit_button("Send")

        if submitted:
            if not user_input or not user_input.strip():
                st.error("Please enter a question to ask the assistant.")
            else:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                with st.spinner("Analyzing your query..."):
                    try:
                        ai_response = generate_chat_response(
                            user_input,
                            context={
                                "selected_stocks": selected_stocks,
                                "risk_profile": st.session_state.risk_profile,
                                "chat_history": st.session_state.chat_history[-5:]
                            }
                        )
                    except requests.exceptions.HTTPError:
                        st.error("The AI assistant could not process your request. Please try again later.")
                        ai_response = None
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                        ai_response = None

                    if ai_response:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": ai_response,
                            "timestamp": datetime.now().strftime("%H:%M")
                        })

        with chat_container:
            for msg in st.session_state.chat_history[-10:]:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(f"👤 You ({msg['timestamp']}):")
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(f"🤖 AI ({msg['timestamp']}):")
                        st.markdown(msg["content"])

    st.divider()
    st.caption(f"""
        Financial AI Agent Pro | University Project | 
        Last Updated: {datetime.now().strftime("%Y-%m-%d")}
    """)


if __name__ == "__main__":
    main()
