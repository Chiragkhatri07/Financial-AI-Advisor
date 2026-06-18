import streamlit as st
import feedparser
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from cachetools import TTLCache
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import threading
import tempfile
import json
import queue
import time
import concurrent.futures
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS (graceful fallbacks)
# ─────────────────────────────────────────────────────────────────────────────
# FinBERT loads lazily via cache_resource so it never blocks startup
@st.cache_resource(show_spinner=False)
def _load_finbert():
    try:
        from transformers import pipeline as hf_pipeline
        return hf_pipeline("text-classification", model="yiyanghkust/finbert-tone")
    except Exception:
        return None

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
try:
    ALPHA_VANTAGE_API = st.secrets.get("ALPHA_VANTAGE_API", os.getenv("ALPHA_VANTAGE_API", ""))
    NEWS_API_KEY      = st.secrets.get("NEWS_API_KEY", os.getenv("NEWS_API_KEY", ""))
    GEMINI_API_KEY    = st.secrets.get("GOOGLE_API_KEY", st.secrets.get("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))))
except Exception:
    ALPHA_VANTAGE_API = os.getenv("ALPHA_VANTAGE_API", "")
    NEWS_API_KEY      = os.getenv("NEWS_API_KEY", "")
    GEMINI_API_KEY    = os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY", ""))


# Configure Gemini once at startup
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini configured successfully.")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root {
    --bg-deep:     #080d14;
    --bg-card:     #0e1620;
    --bg-elevated: #141f2e;
    --border:      #1e2d42;
    --accent-gold: #c9a84c;
    --accent-teal: #0fd4b0;
    --accent-red:  #e05c6a;
    --accent-blue: #3b82f6;
    --text-primary:#e8edf4;
    --text-muted:  #6b7e96;
    --text-dim:    #3d5068;
    --font-display: 'DM Serif Display', serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --radius:       12px;
    --shadow:       0 8px 32px rgba(0,0,0,.5);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

h1 { font-family: var(--font-display) !important; font-size: 2.4rem !important; color: var(--text-primary) !important; letter-spacing: -0.5px; }
h2 { font-family: var(--font-display) !important; font-size: 1.7rem !important; color: var(--text-primary) !important; }
h3 { font-family: var(--font-body) !important; font-weight: 600 !important; color: var(--accent-gold) !important; letter-spacing: .4px; }

[data-testid="stTabs"] button {
    font-family: var(--font-body) !important; font-weight: 500 !important;
    color: var(--text-muted) !important; border-bottom: 2px solid transparent !important;
    transition: all .2s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-gold) !important; border-bottom: 2px solid var(--accent-gold) !important;
    background: transparent !important;
}
[data-testid="stMetric"] {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stMetricLabel"]  { color: var(--text-muted) !important; font-size: .78rem !important; text-transform: uppercase; letter-spacing: .8px; }
[data-testid="stMetricValue"]  { font-family: var(--font-mono) !important; font-size: 1.6rem !important; color: var(--text-primary) !important; }
[data-testid="stMetricDelta"]  { font-size: .85rem !important; font-family: var(--font-mono) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold) 0%, #a0742e 100%) !important;
    color: var(--bg-deep) !important; font-weight: 600 !important;
    font-family: var(--font-body) !important; border: none !important;
    border-radius: 8px !important; padding: 0.45rem 1.1rem !important;
    transition: opacity .2s, transform .15s !important;
    box-shadow: 0 2px 12px rgba(201,168,76,.25) !important;
}
.stButton > button:hover { opacity: .88; transform: translateY(-1px); }
.stButton > button[kind="secondary"] {
    background: var(--bg-elevated) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border) !important; box-shadow: none !important;
}
.stTextInput > div > div > input, .stTextArea textarea,
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
.stTextInput > div > div > input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,.15) !important;
}
.stNumberInput input {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
details {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; margin-bottom: .6rem !important;
}
summary { color: var(--text-primary) !important; font-weight: 500 !important; padding: .6rem .9rem !important; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
    overflow: hidden !important;
}
[data-testid="stChatMessage"] {
    background: var(--bg-elevated) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; margin-bottom: .5rem !important;
}
.stProgress > div > div { background: var(--accent-gold) !important; border-radius: 99px !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent-gold) !important; }
.stAlert { border-radius: var(--radius) !important; }
hr { border-color: var(--border) !important; opacity: .4; margin: 1.2rem 0; }
.stCaption, small { color: var(--text-muted) !important; font-size: .78rem !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
.sidebar-title { font-family: var(--font-display); font-size: 1.4rem; color: var(--accent-gold); letter-spacing: -0.3px; margin-bottom: .25rem; }
.sidebar-subtitle { font-size: .72rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 1.2rem; }
.hero-banner {
    background: linear-gradient(135deg, #0e1a2b 0%, #0a1520 60%, #06101a 100%);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 2rem 2.4rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(201,168,76,.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-family: var(--font-display); font-size: 2.1rem; color: var(--text-primary); margin: 0 0 .3rem; line-height: 1.2; }
.hero-accent { color: var(--accent-gold); }
.hero-sub { font-size: .88rem; color: var(--text-muted); font-weight: 300; letter-spacing: .3px; }
.stat-pill {
    display: inline-block; background: rgba(201,168,76,.1);
    border: 1px solid rgba(201,168,76,.2); color: var(--accent-gold);
    font-family: var(--font-mono); font-size: .72rem;
    padding: .18rem .6rem; border-radius: 99px; margin-right: .4rem;
}
.stock-card {
    background: var(--bg-elevated); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 1.2rem 1.4rem;
    margin-bottom: 1rem; transition: border-color .2s;
}
.stock-card:hover { border-color: var(--accent-gold); }
.section-label { font-size: .7rem; text-transform: uppercase; letter-spacing: 1.4px; color: var(--text-dim); font-weight: 500; margin-bottom: .5rem; }
.badge { display: inline-block; font-size: .72rem; font-weight: 600; padding: .2rem .7rem; border-radius: 99px; font-family: var(--font-mono); }
.badge-pos  { background: rgba(15,212,176,.12); color: var(--accent-teal); border: 1px solid rgba(15,212,176,.25); }
.badge-neg  { background: rgba(224,92,106,.12); color: var(--accent-red);  border: 1px solid rgba(224,92,106,.25); }
.badge-neu  { background: rgba(107,126,150,.12); color: var(--text-muted); border: 1px solid var(--border); }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-gold); }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1118",
    font=dict(family="DM Sans, sans-serif", color="#8da0b5", size=11),
    xaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42", zerolinecolor="#1e2d42"),
    yaxis=dict(gridcolor="#1e2d42", linecolor="#1e2d42", zerolinecolor="#1e2d42"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2d42"),
    colorway=["#c9a84c", "#0fd4b0", "#3b82f6", "#e05c6a", "#a78bfa", "#f59e0b"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHING & SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_cache():
    # keep a small TTL cache only for data not covered by st.cache_data
    return TTLCache(maxsize=100, ttl=300)

@st.cache_resource(show_spinner=False)
def get_vader():
    return SentimentIntensityAnalyzer()

cache = get_cache()
vader = get_vader()

# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA
# ─────────────────────────────────────────────────────────────────────────────
STOCK_LIST = {
    # US Stocks
    "Apple Inc":                     "AAPL",
    "Microsoft Corporation":         "MSFT",
    "Alphabet Inc (Google)":         "GOOGL",
    "Amazon.com Inc":                "AMZN",
    "Meta Platforms":                "META",
    "Tesla Inc":                     "TSLA",
    "NVIDIA Corporation":            "NVDA",
    "Netflix Inc":                   "NFLX",
    
    # Indian Stocks
    "Reliance Industries":           "RELIANCE.NS",
    "Tata Consultancy Services":     "TCS.NS",
    "HDFC Bank":                     "HDFCBANK.NS",
    "Infosys":                       "INFY.NS",
    "ICICI Bank":                    "ICICIBANK.NS",
    "Bajaj Finance":                 "BAJFINANCE.NS",
    "Asian Paints":                  "ASIANPAINT.NS",
    "Bharti Airtel":                 "BHARTIARTL.NS",
    "State Bank of India":           "SBIN.NS",
    "Life Insurance Corporation":    "LICI.NS",
    "ITC Limited":                   "ITC.NS",
    "Hindustan Unilever":            "HINDUNILVR.NS",
    "Larsen & Toubro":               "LT.NS",
    "Tata Steel":                    "TATASTEEL.NS",
    "Tata Motors":                   "TATAMOTORS.NS",
    "HCL Technologies":              "HCLTECH.NS",
    "Axis Bank":                     "AXISBANK.NS",
    "Maruti Suzuki":                 "MARUTI.NS",
    "Sun Pharmaceutical":            "SUNPHARMA.NS",
    "Kotak Mahindra Bank":           "KOTAKBANK.NS",
    "NTPC Limited":                  "NTPC.NS",
    "Oil and Natural Gas Corp":      "ONGC.NS",
    "Adani Enterprises":             "ADANIENT.NS",
    "Coal India":                    "COALINDIA.NS",
    "Power Grid Corporation":        "POWERGRID.NS",
    "Mahindra & Mahindra":           "M&M.NS",
    "Titan Company":                 "TITAN.NS",
    "UltraTech Cement":              "ULTRACEMCO.NS",
    "Wipro":                         "WIPRO.NS",
    "JSW Steel":                     "JSWSTEEL.NS",
}


FUND_DATA = {
    "Large Cap": [
        {"name": "ICICI Prudential Bluechip Fund", "returns": 12.5, "risk": "Low",       "allocation": 30},
        {"name": "SBI Bluechip Fund",               "returns": 11.8, "risk": "Low",       "allocation": 25},
    ],
    "Mid Cap": [
        {"name": "Axis Midcap Fund",               "returns": 15.2, "risk": "Medium",    "allocation": 20},
        {"name": "Kotak Emerging Equity Fund",     "returns": 16.1, "risk": "Medium",    "allocation": 15},
    ],
    "Small Cap": [
        {"name": "Nippon India Small Cap Fund",    "returns": 18.7, "risk": "High",      "allocation": 10},
        {"name": "HDFC Small Cap Fund",            "returns": 17.9, "risk": "High",      "allocation":  5},
    ],
    "Sectoral": [
        {"name": "Tata Digital India Fund",        "returns": 22.3, "risk": "Very High", "allocation":  5},
        {"name": "SBI Healthcare Opportunities",   "returns": 19.8, "risk": "Very High", "allocation":  5},
    ],
}

PERIOD_MAP = {
    "1 Week":   "5d",
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "5 Years":  "5y",
}

COMPANY_NAMES = {
    # US
    "AAPL": "Apple Inc",   "MSFT": "Microsoft",
    "GOOGL": "Alphabet",   "TSLA": "Tesla Inc",
    "AMZN": "Amazon",      "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    # Indian
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "SBIN.NS": "State Bank of India",
    "LICI.NS": "Life Insurance Corporation of India",
    "ITC.NS": "ITC Limited",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "LT.NS": "Larsen & Toubro",
    "BAJFINANCE.NS": "Bajaj Finance",
    "TATASTEEL.NS": "Tata Steel",
    "TATAMOTORS.NS": "Tata Motors",
    "HCLTECH.NS": "HCL Technologies",
    "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti Suzuki",
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "NTPC.NS": "NTPC Limited",
    "ONGC.NS": "ONGC India",
    "ADANIENT.NS": "Adani Enterprises",
    "COALINDIA.NS": "Coal India",
    "POWERGRID.NS": "Power Grid Corporation",
    "M&M.NS": "Mahindra & Mahindra",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "ASIANPAINT.NS": "Asian Paints",
    "WIPRO.NS": "Wipro",
    "JSWSTEEL.NS": "JSW Steel",
}

RISK_RETURN = {"Low": 0.10, "Medium": 0.12, "High": 0.15}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = dict(
        chat_history   = [],
        sip_chat       = [],
        risk_profile   = "Medium",
        query_logs     = [],
        admin_password = "",
        voice_enabled  = False,
        auto_speak     = False,
        mic_clicked    = False,
        portfolio      = {},
        portfolio_add_error = "",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_available_model() -> Optional[str]:
    """Try candidate model names and return the first that works."""
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return None

    candidates = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "models/gemini-2.0-flash",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]

    for name in candidates:
        try:
            m = genai.GenerativeModel(name)
            r = m.generate_content("Say OK", generation_config={"max_output_tokens": 5})
            if r.text:
                logger.info(f"Working Gemini model: {name}")
                return name
        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")

    # Fallback: enumerate API-reported models
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                model_id = m.name.replace("models/", "")
                logger.info(f"Fallback model from list: {model_id}")
                return model_id
    except Exception as e:
        logger.error(f"list_models failed: {e}")

    return None


def gemini_generate(prompt: str, max_tokens: int = 1024,
                    system_instruction: Optional[str] = None,
                    _retry: bool = False) -> str:
    # Retry flag prevents infinite recursion if API fails
    if not GEMINI_AVAILABLE:
        return "⚠️ google-generativeai not installed. Run: pip install google-generativeai"
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key missing."

    working_model = get_available_model()
    if working_model is None:
        return "⚠️ No working Gemini model found. Check your API key and quota."

    try:
        if system_instruction:
            model = genai.GenerativeModel(working_model, system_instruction=system_instruction)
        else:
            model = genai.GenerativeModel(working_model)

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.65, "max_output_tokens": max_tokens},
        )
        return response.text

    except Exception as e:
        # Clear cache and retry once to allow model re-probing
        if not _retry:
            st.cache_resource.clear()
            # Re-initialise singletons that were cleared
            global cache, vader
            cache = get_cache()
            vader = get_vader()
            return gemini_generate(prompt, max_tokens, system_instruction, _retry=True)
        return f"⚠️ Gemini API error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_realtime_stock_data(symbol: str) -> dict:
    """Fetch real-time quote. Cached 5 minutes via Streamlit to avoid redundant calls."""
    try:
        url = (
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
            f"&symbol={symbol}&apikey={ALPHA_VANTAGE_API}"
        )
        r = requests.get(url, timeout=5)
        d = r.json().get("Global Quote", {})
        if d.get("05. price"):
            return {
                "symbol": symbol,
                "price":  float(d["05. price"]),
                "change": float(d.get("10. change percent", "0%").rstrip("%")),
                "volume": int(d.get("06. volume", 0)),
                "source": "Alpha Vantage",
            }
    except Exception:
        pass

    try:
        ticker = yf.Ticker(symbol)
        data   = ticker.history(period="2d")   # 2d avoids empty results near market open
        if not data.empty:
            o, c = data["Open"].iloc[-1], data["Close"].iloc[-1]
            return {
                "symbol": symbol,
                "price":  round(c, 2),
                "change": round((c - o) / o * 100, 2) if o else 0,
                "volume": int(data["Volume"].iloc[-1]),
                "source": "Yahoo Finance",
            }
    except Exception:
        pass

    return {"symbol": symbol, "price": 0, "change": 0, "volume": 0, "source": "N/A"}


def get_realtime_batch(symbols: tuple) -> dict:
    """Fetch multiple quotes in parallel. Returns {symbol: quote_dict}."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as ex:
        results = list(ex.map(get_realtime_stock_data, symbols))
    return {sym: res for sym, res in zip(symbols, results)}


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_chart(symbol: str, period: str) -> go.Figure:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=PERIOD_MAP.get(period, "1y"))
    if hist.empty:
        fig = go.Figure()
        fig.update_layout(title="No data", **PLOTLY_THEME)
        return fig

    hist["SMA_20"] = hist["Close"].rolling(20).mean()
    hist["SMA_50"] = hist["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name="Price",
        increasing_line_color="#0fd4b0", decreasing_line_color="#e05c6a",
    ))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA_20"],
                             line=dict(color="#c9a84c", width=1.2), name="SMA 20"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["SMA_50"],
                             line=dict(color="#3b82f6", width=1.2), name="SMA 50"))
    fig.update_layout(
        title=dict(text=f"{COMPANY_NAMES.get(symbol, symbol)} · Price History", font=dict(size=14, color="#e8edf4")),
        xaxis_rangeslider_visible=False, hovermode="x unified", height=420,
        **PLOTLY_THEME,
    )
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def compare_stocks(symbols: tuple, period: str):
    metrics = []
    fig = go.Figure()
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=PERIOD_MAP.get(period, "1y"))
        if hist.empty:
            continue
        s, e = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        ret  = (e - s) / s * 100
        vol  = hist["Close"].pct_change().std() * np.sqrt(252)
        company_name = COMPANY_NAMES.get(symbol, symbol)
        metrics.append({
            "Company":     company_name,
            "Start (₹/$)": round(s, 2),
            "End (₹/$)":   round(e, 2),
            "Return %":    round(ret, 2),
            "Volatility":  round(vol, 4),
            "Avg Vol":     f"{int(hist['Volume'].mean()):,}",
        })
        norm = hist["Close"] / hist["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(x=hist.index, y=norm, name=company_name, mode="lines",
                                 line=dict(width=2)))
    fig.update_layout(
        title=dict(text="Normalised Performance (base 100)", font=dict(size=13, color="#e8edf4")),
        yaxis_title="Indexed (base=100)", hovermode="x unified", height=420,
        **PLOTLY_THEME,
    )

    # Guard against empty dataset or missing Symbol column
    if metrics:
        df = pd.DataFrame(metrics)
        if "Company" in df.columns:
            df = df.set_index("Company")
    else:
        df = pd.DataFrame()

    return fig, df

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT (parallel fetch)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_newsapi(symbol: str) -> list:
    term = COMPANY_NAMES.get(symbol.upper(), symbol)
    url  = (f"https://newsapi.org/v2/everything?q={term}"
            f"&apiKey={NEWS_API_KEY}&language=en&sortBy=publishedAt&pageSize=8")
    try:
        r = requests.get(url, timeout=6)
        arts = r.json().get("articles", [])
        return [{"title": a.get("title", ""), "source": a.get("source", {}).get("name", ""),
                 "date": a.get("publishedAt", ""),
                 "content": f"{a.get('title','')}. {a.get('description','')}"}
                for a in arts]
    except Exception:
        return []

def _fetch_alpha_news(symbol: str) -> list:
    url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
           f"&tickers={symbol}&apikey={ALPHA_VANTAGE_API}")
    try:
        r = requests.get(url, timeout=6)
        return [{"title": a.get("title", ""), "source": a.get("source", ""),
                 "date": a.get("time_published", ""),
                 "content": f"{a.get('title','')}. {a.get('summary','')}"}
                for a in r.json().get("feed", [])]
    except Exception:
        return []

def _fetch_yahoo_rss(symbol: str) -> list:
    try:
        feed = feedparser.parse(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US")
        return [{"title": e.title, "source": "Yahoo Finance",
                 "date": e.get("published", ""),
                 "content": f"{e.title}. {e.get('summary', '')}"}
                for e in feed.entries]
    except Exception:
        return []

def format_date(ds: str) -> str:
    try:
        if "T" in ds:
            return datetime.fromisoformat(ds.replace("Z", "")).strftime("%b %d, %Y")
        return ds[:10]
    except Exception:
        return ds


@st.cache_data(ttl=1800, show_spinner=False)
def analyze_news_sentiment(symbol: str) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f1 = ex.submit(_fetch_newsapi, symbol)
        f2 = ex.submit(_fetch_alpha_news, symbol)
        f3 = ex.submit(_fetch_yahoo_rss, symbol)
        articles = f1.result() + f2.result() + f3.result()

    if not articles:
        return {"symbol": symbol, "compound_score": 0, "label": "Neutral",
                "top_news": [], "analysis": "No news found."}

    v_scores, top_news = [], []
    for art in articles[:10]:
        txt = art["content"]
        vs  = vader.polarity_scores(txt)["compound"]
        v_scores.append(vs)
        top_news.append({
            "title":       art["title"],
            "source":      art["source"],
            "date":        format_date(art["date"]),
            "vader_score": vs,
            "sentiment":   "Positive" if vs > .05 else "Negative" if vs < -.05 else "Neutral",
        })

    compound = sum(v_scores) / len(v_scores)
    label = ("Strongly Positive" if compound > .2 else
             "Positive"          if compound > .05 else
             "Strongly Negative" if compound < -.2 else
             "Negative"          if compound < -.05 else "Neutral")

    top_news.sort(key=lambda x: abs(x["vader_score"]), reverse=True)
    return {"symbol": symbol, "compound_score": compound, "label": label,
            "top_news": top_news[:5],
            "analysis": _sentiment_text(compound, symbol)}

def _sentiment_text(score: float, symbol: str) -> str:
    if score > .3:   return f"📈 Strong positive sentiment for {symbol}. Favorable coverage may correlate with near-term price strength."
    if score > .1:   return f"🟢 Positive sentiment for {symbol}. News flow is broadly constructive."
    if score > -.1:  return f"⚪ Neutral sentiment for {symbol}. Mixed signals; technicals may dominate."
    if score > -.3:  return f"🟡 Slightly negative sentiment for {symbol}. Monitor for further deterioration."
    return             f"🔴 Strong negative sentiment for {symbol}. Recent coverage suggests elevated downside risk."


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI GENERATION WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_chat_response(question: str, context: dict) -> str:
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in context.get("chat_history", [])[-4:]
    )
    system_prompt = (
        "You are FinBot, a knowledgeable financial assistant for Indian and global markets. "
        "Always give complete, fully-formed answers. Never cut off mid-sentence. "
        "Use markdown formatting, bullet points, and clear structure in your responses."
    )
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context: Stocks={', '.join(context.get('selected_stocks', []))}, "
        f"Risk={context.get('risk_profile','Medium')}\n\n"
        f"User question: {question}\n\n"
        "Provide a complete and helpful answer. Use bullet points where appropriate. "
        "Always finish your response fully."
    )
    return gemini_generate(user_prompt, max_tokens=1000, system_instruction=system_prompt)


def generate_comprehensive_report(stocks: list, query: str, risk_profile: str) -> str:
    system_prompt = (
        "You are a senior financial analyst. Write complete, professional analysis in markdown. "
        "Always finish every section fully. Never cut off mid-sentence. "
        "Use ## headers and bullet points for clarity."
    )
    user_prompt = (
        f"Stocks under analysis: {', '.join(stocks)}\n"
        f"Investor risk profile: {risk_profile}\n"
        f"Report date: {datetime.now().strftime('%B %d, %Y')}\n"
        f"Analyst query: {query}\n\n"
        "Write a complete report with ALL of these sections:\n"
        "## 1. Executive Summary\n"
        "## 2. Key Technical Indicators\n"
        "## 3. News and Sentiment Outlook\n"
        "## 4. Risk Assessment\n"
        "## 5. Recommendation\n\n"
        "Complete every section fully before ending your response."
    )
    return gemini_generate(user_prompt, max_tokens=1800, system_instruction=system_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# SIP CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────
def calculate_sip_projection(monthly: float, years: int, risk: str) -> dict:
    rate     = RISK_RETURN.get(risk, 0.12) / 12
    fv       = 0.0
    invested = 0.0
    proj     = {"years": [], "invested": [], "value": []}
    for m in range(1, years * 12 + 1):
        invested += monthly
        fv = (fv + monthly) * (1 + rate)
        if m % 12 == 0:
            proj["years"].append(m // 12)
            proj["invested"].append(invested)
            proj["value"].append(round(fv, 0))
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# INVESTMENT RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_investment_recommendations(risk: str) -> dict:
    if risk == "Low":
        return {
            "Large Cap Funds": FUND_DATA["Large Cap"],
            "Balanced Funds":  [{"name": "HDFC Balanced Advantage Fund", "returns": 10.9, "risk": "Low", "allocation": 45}],
            "Debt Funds":      [{"name": "ICICI Prudential Corporate Bond Fund", "returns": 8.2, "risk": "Very Low", "allocation": 25}],
        }
    if risk == "Medium":
        return {
            "Core (Large Cap)": FUND_DATA["Large Cap"],
            "Growth (Mid Cap)": FUND_DATA["Mid Cap"],
            "Satellite":        [
                {"name": "Parag Parikh Flexi Cap Fund",     "returns": 14.5, "risk": "Medium", "allocation": 15},
                {"name": "Mirae Asset Hybrid Equity Fund",  "returns": 13.2, "risk": "Medium", "allocation": 10},
            ],
        }
    return {
        "Growth (Mid Cap)":       FUND_DATA["Mid Cap"],
        "Aggressive (Small Cap)": FUND_DATA["Small Cap"],
        "Thematic":               FUND_DATA["Sectoral"],
        "International":          [
            {"name": "Motilal Oswal NASDAQ 100 ETF",              "returns": 16.8, "risk": "High", "allocation": 10},
            {"name": "Franklin India Feeder – US Opportunities",  "returns": 15.3, "risk": "High", "allocation":  5},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────────────────────
def speak_text(text: str):
    if not text:
        return
    if not GTTS_AVAILABLE:
        st.info(text)
        return
    try:
        tts = gTTS(text=text[:1500], lang="en", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
        tts.save(tmp_path)
        with open(tmp_path, "rb") as f:   # Read file contents before deleting
            audio = f.read()
        os.unlink(tmp_path)
        st.audio(audio, format="audio/mp3")
    except Exception as e:
        st.warning(f"TTS failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STT
# ─────────────────────────────────────────────────────────────────────────────
def listen_for_speech(timeout: int = 10) -> Optional[str]:
    if not SPEECH_AVAILABLE:
        st.error("speech_recognition not installed.")
        return None
    recognizer = sr.Recognizer()
    recognizer.energy_threshold         = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold          = 0.8
    placeholder = st.empty()
    try:
        with sr.Microphone() as src:
            placeholder.info("🎤 Listening … speak now")
            recognizer.adjust_for_ambient_noise(src, duration=0.8)
            audio = recognizer.listen(src, timeout=timeout, phrase_time_limit=15)
        placeholder.info("🔍 Processing …")
        text = recognizer.recognize_google(audio)
        placeholder.success(f"✅ Heard: {text[:60]}")
        time.sleep(1.5)
        placeholder.empty()
        return text
    except sr.WaitTimeoutError:
        placeholder.warning("⏱ Timeout – please try again.")
    except sr.UnknownValueError:
        placeholder.error("Could not understand audio.")
    except Exception as e:
        placeholder.error(f"Mic error: {e}")
    time.sleep(2)
    placeholder.empty()
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
def admin_panel():
    st.markdown("### 🛡️ Administrator Panel")
    t1, t2, t3 = st.tabs(["Query Logs", "Data Sources", "System Health"])
    with t1:
        if st.session_state.query_logs:
            df = pd.DataFrame(st.session_state.query_logs)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇ Export CSV", csv,
                               f"logs_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        else:
            st.info("No query logs yet.")
    with t2:
        for src in ["Alpha Vantage", "Yahoo Finance", "NewsAPI"]:
            c1, c2 = st.columns([4, 1])
            c1.write(src)
            c2.checkbox("Enable", value=True, key=f"src_{src}")
        if st.button("🔄 Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared.")
    with t3:
        c1, c2, c3 = st.columns(3)
        c1.metric("Active Users",  15,      "+2")
        c2.metric("API Calls",   "1,243", "+12%")
        c3.metric("Latency",     "0.4s",  "-0.1s")


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO UTILITIES & CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
USD_INR_RATE = 83.5

def get_usd_value(symbol: str, price: float) -> float:
    """Helper to convert Indian stock prices (ending in .NS) to USD."""
    if symbol.upper().endswith(".NS"):
        return price / USD_INR_RATE
    return price

def get_inr_value(symbol: str, price: float) -> float:
    """Helper to convert US stock prices to INR."""
    if not symbol.upper().endswith(".NS"):
        return price * USD_INR_RATE
    return price

@st.cache_data(ttl=3600, show_spinner=False)
def get_portfolio_historical_data(portfolio_symbols: tuple, period: str) -> pd.DataFrame:
    """Fetches historical close prices for all portfolio stocks and merges them."""
    if not portfolio_symbols:
        return pd.DataFrame()
    
    yperiod = PERIOD_MAP.get(period, "1y")
    data_dict = {}
    
    for symbol in portfolio_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=yperiod)
            if not hist.empty:
                data_dict[symbol] = hist["Close"]
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            
    if not data_dict:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_dict)
    df = df.ffill().bfill()
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_benchmark_historical_data(period: str) -> pd.Series:
    """Fetches SPY historical close prices as USD benchmark."""
    try:
        yperiod = PERIOD_MAP.get(period, "1y")
        ticker = yf.Ticker("SPY")
        hist = ticker.history(period=yperiod)
        if not hist.empty:
            return hist["Close"]
    except Exception as e:
        logger.error(f"Error fetching benchmark data: {e}")
    return pd.Series()

def calculate_portfolio_historical_value(df_closes: pd.DataFrame, portfolio: dict) -> pd.Series:
    """Calculates historical value of the portfolio in USD (or native)."""
    if df_closes.empty:
        return pd.Series()
        
    val_series = pd.Series(0.0, index=df_closes.index)
    for symbol in df_closes.columns:
        if symbol in portfolio:
            shares = portfolio[symbol]["shares"]
            prices = df_closes[symbol]
            if symbol.endswith(".NS"):
                prices = prices / USD_INR_RATE
            val_series += prices * shares
            
    return val_series

def calculate_portfolio_metrics(portfolio_val: pd.Series, benchmark_val: pd.Series) -> dict:
    """Calculates standard risk/return metrics: Sharpe, volatility, max drawdown, beta."""
    metrics = {
        "annualized_return": 0.0,
        "annualized_vol": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "beta": 1.0,
    }
    if portfolio_val.empty or len(portfolio_val) < 5:
        return metrics
        
    p_returns = portfolio_val.pct_change().dropna()
    
    total_ret = (portfolio_val.iloc[-1] / portfolio_val.iloc[0]) - 1.0
    days = (portfolio_val.index[-1] - portfolio_val.index[0]).days
    if days > 0:
        metrics["annualized_return"] = ((1.0 + total_ret) ** (365.25 / days)) - 1.0
    else:
        metrics["annualized_return"] = 0.0
        
    metrics["annualized_vol"] = p_returns.std() * np.sqrt(252)
    
    rf = 0.03
    if metrics["annualized_vol"] > 0:
        metrics["sharpe_ratio"] = (metrics["annualized_return"] - rf) / metrics["annualized_vol"]
        
    cum_max = portfolio_val.cummax()
    drawdown = (portfolio_val - cum_max) / cum_max
    metrics["max_drawdown"] = drawdown.min() * 100.0
    
    if not benchmark_val.empty:
        b_returns = benchmark_val.pct_change().dropna()
        aligned_df = pd.DataFrame({"p": p_returns, "b": b_returns}).dropna()
        if len(aligned_df) > 5:
            cov = np.cov(aligned_df["p"], aligned_df["b"])[0, 1]
            b_var = aligned_df["b"].var()
            if b_var > 0:
                metrics["beta"] = cov / b_var
                
    return metrics

def run_mpt_optimization(df_closes: pd.DataFrame, num_portfolios: int = 1000) -> dict:
    """Runs a Monte Carlo simulation of weights to find optimal allocation (Max Sharpe)."""
    results = {
        "frontier_vol": [],
        "frontier_ret": [],
        "frontier_sharpe": [],
        "frontier_weights": [],
        "max_sharpe_idx": 0,
        "max_sharpe": 0.0,
        "optimal_weights": {},
        "symbols": list(df_closes.columns)
    }
    
    symbols = list(df_closes.columns)
    n_assets = len(symbols)
    if n_assets == 0:
        return results
        
    usd_closes = df_closes.copy()
    for col in usd_closes.columns:
        if col.endswith(".NS"):
            usd_closes[col] = usd_closes[col] / USD_INR_RATE
            
    daily_returns = usd_closes.pct_change().dropna()
    if daily_returns.empty or len(daily_returns) < 5:
        return results
        
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    
    rf = 0.03
    
    vol_arr = np.zeros(num_portfolios)
    ret_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    weights_arr = np.zeros((num_portfolios, n_assets))
    
    for idx in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_arr[idx, :] = weights
        p_ret = np.sum(mean_returns * weights)
        ret_arr[idx] = p_ret
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        vol_arr[idx] = p_vol
        p_sharpe = (p_ret - rf) / p_vol
        sharpe_arr[idx] = p_sharpe
        
    max_idx = sharpe_arr.argmax()
    
    results["frontier_vol"] = vol_arr
    results["frontier_ret"] = ret_arr
    results["frontier_sharpe"] = sharpe_arr
    results["frontier_weights"] = weights_arr
    results["max_sharpe_idx"] = max_idx
    results["max_sharpe"] = sharpe_arr[max_idx]
    
    opt_weights_list = weights_arr[max_idx, :]
    results["optimal_weights"] = {symbols[i]: opt_weights_list[i] for i in range(n_assets)}
    
    return results

def generate_ai_portfolio_review(portfolio: dict, metrics: dict, risk_profile: str) -> str:
    """Uses Gemini to generate a tailored report evaluating portfolio allocation and risk."""
    system_prompt = (
        "You are a senior portfolio manager and investment strategist. Write a complete, "
        "professional analysis of the user's investment portfolio in markdown. "
        "Always finish every section fully. Never cut off mid-sentence. "
        "Use ## headers and bullet points for clarity."
    )
    
    holdings_text = ""
    total_val = 0.0
    for symbol, info in portfolio.items():
        shares = info["shares"]
        buy_price = info["buy_price"]
        rt = get_realtime_stock_data(symbol)
        curr_price = rt["price"]
        
        ccy = "₹" if symbol.endswith(".NS") else "$"
        cost = shares * buy_price
        curr_val = shares * curr_price
        
        usd_val = curr_val / USD_INR_RATE if symbol.endswith(".NS") else curr_val
        total_val += usd_val
        holdings_text += f"- **{symbol}**: {shares} shares @ average cost {ccy}{buy_price:.2f} (Current Price: {ccy}{curr_price:.2f}, Cost: {ccy}{cost:,.2f}, Value: {ccy}{curr_val:,.2f})\n"
        
    weights_text = ""
    for symbol, info in portfolio.items():
        rt = get_realtime_stock_data(symbol)
        curr_val = info["shares"] * rt["price"]
        usd_val = curr_val / USD_INR_RATE if symbol.endswith(".NS") else curr_val
        weight = (usd_val / total_val) * 100 if total_val > 0 else 0
        weights_text += f"- **{symbol}**: {weight:.1f}% weight\n"

    user_prompt = f"""Evaluate the following investment portfolio for an investor with a **{risk_profile}** risk appetite.
    
### Current Portfolio Holdings:
{holdings_text}

### Asset Allocation Weights:
{weights_text}

### Portfolio Analytics:
- Annualized Return: {metrics['annualized_return']*100:.2f}%
- Annualized Volatility: {metrics['annualized_vol']*100:.2f}%
- Sharpe Ratio (Risk-Adjusted Return): {metrics['sharpe_ratio']:.2f}
- Maximum Drawdown: {metrics['max_drawdown']:.2f}%
- Portfolio Beta vs. S&P 500: {metrics['beta']:.2f}

Write a comprehensive portfolio review covering:
## 1. Executive Allocation Summary
Summarize the current portfolio structure and whether it's well-diversified.

## 2. Risk & Return Profile Analysis
Analyze the portfolio metrics (annualized return, volatility, Sharpe ratio, beta, and max drawdown) in the context of the user's selected risk appetite ({risk_profile}). Highlight if they are taking too much or too little risk.

## 3. Diversification & Sector Allocation
Evaluate the concentration risk (e.g. US Tech vs. Indian Large Caps).

## 4. Rebalancing Recommendations
Provide specific, actionable instructions on what weight adjustments should be made (e.g. increase exposure to large caps, reduce highly volatile tech, or reallocate across sectors) to optimize their risk-adjusted return.

Ensure the analysis is highly professional, thorough, and complete. Finish all sections before ending the output.
"""
    return gemini_generate(user_prompt, max_tokens=1800, system_instruction=system_prompt)

# ─────────────────────────────────────────────────────────────────────────────
# CURRENCY CONVERTER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

CURRENCY_LIST = [
    "USD", "EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD",
    "SGD", "MXN", "BRL", "KRW", "SEK", "NOK", "DKK", "NZD", "AED", "SAR",
    "THB", "MYR", "IDR", "PHP", "PKR", "ZAR", "TRY", "RUB", "PLN", "CZK",
]

CURRENCY_FLAGS = {
    "USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "INR": "🇮🇳",
    "JPY": "🇯🇵", "CAD": "🇨🇦", "AUD": "🇦🇺", "CHF": "🇨🇭",
    "CNY": "🇨🇳", "HKD": "🇭🇰", "SGD": "🇸🇬", "MXN": "🇲🇽",
    "BRL": "🇧🇷", "KRW": "🇰🇷", "SEK": "🇸🇪", "NOK": "🇳🇴",
    "DKK": "🇩🇰", "NZD": "🇳🇿", "AED": "🇦🇪", "SAR": "🇸🇦",
    "THB": "🇹🇭", "MYR": "🇲🇾", "IDR": "🇮🇩", "PHP": "🇵🇭",
    "PKR": "🇵🇰", "ZAR": "🇿🇦", "TRY": "🇹🇷", "RUB": "🇷🇺",
    "PLN": "🇵🇱", "CZK": "🇨🇿",
}

CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥",
    "CAD": "C$", "AUD": "A$", "CHF": "Fr", "CNY": "¥", "HKD": "HK$",
    "SGD": "S$", "MXN": "Mex$", "BRL": "R$", "KRW": "₩", "SEK": "kr",
    "NOK": "kr", "DKK": "kr", "NZD": "NZ$", "AED": "د.إ", "SAR": "ر.س",
    "THB": "฿", "MYR": "RM", "IDR": "Rp", "PHP": "₱", "PKR": "₨",
    "ZAR": "R", "TRY": "₺", "RUB": "₽", "PLN": "zł", "CZK": "Kč",
}

@st.cache_data(ttl=600, show_spinner=False)
def get_fx_rate(from_ccy: str, to_ccy: str) -> float:
    """Fetch live exchange rate via Yahoo Finance currency pair."""
    if from_ccy == to_ccy:
        return 1.0
    try:
        pair = f"{from_ccy}{to_ccy}=X"
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="2d")
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 6)
    except Exception as e:
        logger.warning(f"FX rate fetch failed for {from_ccy}/{to_ccy}: {e}")
    return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_history(from_ccy: str, to_ccy: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch historical exchange rate data for a currency pair."""
    try:
        pair = f"{from_ccy}{to_ccy}=X"
        ticker = yf.Ticker(pair)
        hist = ticker.history(period=period)
        if not hist.empty:
            return hist[["Close"]].rename(columns={"Close": "Rate"})
    except Exception as e:
        logger.warning(f"FX history fetch failed: {e}")
    return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
@st.cache_data(ttl=600, show_spinner=False)
def get_popular_fx_rates_cached() -> list:
    """Fetch the 10 popular USD pairs in parallel. Cached 10 min."""
    popular = [
        ("EUR", "Euro"), ("GBP", "British Pound"), ("INR", "Indian Rupee"),
        ("JPY", "Japanese Yen"), ("CAD", "Canadian Dollar"),
        ("AUD", "Australian Dollar"), ("CHF", "Swiss Franc"),
        ("CNY", "Chinese Yuan"), ("SGD", "Singapore Dollar"), ("AED", "UAE Dirham"),
    ]
    ccys = [c for c, _ in popular]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        rates = list(ex.map(lambda c: get_fx_rate("USD", c), ccys))
    return [(ccy, name, r) for (ccy, name), r in zip(popular, rates)]


@st.cache_data(ttl=600, show_spinner=False)
def get_cross_rates_cached() -> list:
    """Fetch the cross-rate table in parallel. Cached 10 min."""
    cross_ccys = ["EUR", "GBP", "INR", "JPY", "CAD", "AUD", "CHF", "CNY", "SGD", "HKD"]
    names = {
        "EUR": "Euro", "GBP": "British Pound", "INR": "Indian Rupee",
        "JPY": "Japanese Yen", "CAD": "Canadian Dollar",
        "AUD": "Australian Dollar", "CHF": "Swiss Franc",
        "CNY": "Chinese Yuan", "SGD": "Singapore Dollar", "HKD": "HK Dollar",
    }
    def _fetch_pair(c):
        return c, get_fx_rate(c, "USD"), get_fx_rate("USD", c)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_fetch_pair, cross_ccys))
    rows = []
    for c, r_to, r_from in results:
        rows.append({
            "Currency": f"{CURRENCY_FLAGS.get(c, '')} {c}",
            "Name": names.get(c, c),
            "1 USD → X": f"{r_from:,.4g}" if r_from else "N/A",
            "1 X → USD": f"{r_to:,.6g}"   if r_to   else "N/A",
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FinAdvisor AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 FinAdvisor AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Powered by Google Gemini</div>', unsafe_allow_html=True)

        working_model = get_available_model()
        if working_model:
            st.success(f"✅ Model: `{working_model}`")
        else:
            st.error("❌ No working Gemini model. Check your API key.")

        # Render login form in the sidebar
        def is_admin() -> bool:
            try:
                admin_pw = st.secrets.get("ADMIN_PASSWORD", os.getenv("ADMIN_PASSWORD", "admin123"))
            except Exception:
                admin_pw = os.getenv("ADMIN_PASSWORD", "admin123")
            return st.session_state.admin_password == admin_pw

        with st.expander("🔐 Admin Login"):
            st.session_state.admin_password = st.text_input(
                "Password", type="password", key="admin_pw")
        if is_admin():
            st.success("🛡️ Admin Active")

        st.markdown("---")

        selected_names = st.multiselect(
            "📌 Select Stocks",
            list(STOCK_LIST.keys()),
            default=list(STOCK_LIST.keys())[:3],
        )
        selected_stocks = [STOCK_LIST[n] for n in selected_names]

        st.session_state.risk_profile = st.select_slider(
            "⚖️ Risk Appetite",
            options=["Low", "Medium", "High"],
            value=st.session_state.risk_profile,
        )

        analysis_period = st.selectbox(
            "📅 Period",
            list(PERIOD_MAP.keys()),
            index=4,
        )

        st.markdown("---")

        with st.expander("🎙️ Voice Settings"):
            st.session_state.voice_enabled = st.checkbox(
                "Enable Voice", value=st.session_state.voice_enabled)
            if st.session_state.voice_enabled:
                st.session_state.auto_speak = st.checkbox(
                    "Auto-speak responses", value=st.session_state.auto_speak)
                test_text = st.text_input("Test phrase", "Markets are looking strong today.")
                if st.button("🔊 Test TTS"):
                    speak_text(test_text)

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Render the admin panel in the main content area (st.tabs is unsupported in sidebar)
    if is_admin():
        admin_panel()
        st.markdown("---")

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-title">Your <span class="hero-accent">AI-Powered</span><br>Financial Advisor</div>
        <div class="hero-sub">Real-time markets · Sentiment analysis · Personalised SIP planning · AI insights</div>
        <br>
        <span class="stat-pill">{datetime.now().strftime('%b %d, %Y')}</span>
        <span class="stat-pill">Risk: {st.session_state.risk_profile}</span>
        <span class="stat-pill">{len(selected_stocks)} stocks selected</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈  Live Market",
        "🧠  AI Insights",
        "💼  Portfolio Analysis",
        "🔍  Comparison",
        "💡  Recommendations",
        "💱  Currency Converter",
        "💬  Chat",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — LIVE MARKET
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        if not selected_stocks:
            st.warning("Select at least one stock in the sidebar.")
        else:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                rt_results = list(ex.map(get_realtime_stock_data, selected_stocks))

            cols = st.columns(min(len(selected_stocks), 4))
            for i, (sym, rt) in enumerate(zip(selected_stocks, rt_results)):
                ccy = "₹" if ".NS" in sym else "$"
                company_name = COMPANY_NAMES.get(sym, sym)
                with cols[i % 4]:
                    color = "var(--accent-teal)" if rt["change"] >= 0 else "var(--accent-red)"
                    arrow = "▲" if rt["change"] >= 0 else "▼"
                    st.markdown(f"""
                    <div class="stock-card">
                        <div class="section-label" style="text-overflow:ellipsis;overflow:hidden;white-space:nowrap;" title="{company_name}">{company_name}</div>
                        <div style="font-family:var(--font-mono);font-size:1.6rem;color:var(--text-primary);">
                            {ccy}{rt['price']:,.2f}
                        </div>
                        <div style="color:{color};font-family:var(--font-mono);font-size:.88rem;margin-top:.2rem;">
                            {arrow} {rt['change']:+.2f}%
                        </div>
                        <div style="color:var(--text-dim);font-size:.72rem;margin-top:.3rem;">
                            Vol: {rt['volume']:,} · {rt['source']}
                        </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            for sym in selected_stocks:
                st.markdown(f"#### {COMPANY_NAMES.get(sym, sym)} — Price Chart")
                fig = get_historical_chart(sym, analysis_period)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — AI INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        if not selected_stocks:
            st.warning("Select stocks first.")
        else:
            st.markdown("### 📰 News Sentiment Analysis")
            for sym in selected_stocks:
                with st.expander(f"Sentiment — {COMPANY_NAMES.get(sym, sym)}", expanded=(sym == selected_stocks[0])):
                    with st.spinner(f"Fetching sentiment for {sym}…"):
                        sent = analyze_news_sentiment(sym)

                    col_gauge, col_info = st.columns([1, 2])
                    with col_gauge:
                        # Determine bar color dynamically based on sentiment score
                        score = sent["compound_score"]
                        bar_color = "#0fd4b0" if score > 0.05 else "#e05c6a" if score < -0.05 else "#6b7e96"

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=round(score, 3),
                            domain={"x": [0, 1], "y": [0, 1]},
                            number={"font": {"color": "#e8edf4", "family": "DM Mono"}},
                            gauge={
                                "axis": {
                                    "range": [-1, 1],
                                    "tickmode": "array",
                                    "tickvals": [-1, -0.5, 0, 0.5, 1],
                                    "ticktext": ["-1", "-0.5", "0", "0.5", "1"],
                                    "ticks": "",
                                    "ticklen": 0,
                                    "tickfont": {"color": "#6b7e96", "size": 10}
                                },
                                "bar":       {"color": bar_color, "thickness": 0.65},
                                "bgcolor":   "#0a1118",
                                "bordercolor": "#1e2d42",
                                "steps": [
                                    {"range": [-1, -.2], "color": "#2a1018"},
                                    {"range": [-.2, .2],  "color": "#141f2e"},
                                    {"range": [.2, 1],   "color": "#0d2118"},
                                ],
                            },
                        ))
                        fig.update_layout(height=220, paper_bgcolor="rgba(0,0,0,0)",
                                          margin=dict(t=20, b=20, l=30, r=30))
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                    with col_info:
                        badge_cls = ("badge-pos" if "Positive" in sent["label"]
                                     else "badge-neg" if "Negative" in sent["label"]
                                     else "badge-neu")
                        st.markdown(f'<span class="badge {badge_cls}">{sent["label"]}</span>',
                                    unsafe_allow_html=True)
                        st.caption(sent["analysis"])
                        st.markdown("**Top Headlines**")
                        for n in sent["top_news"][:3]:
                            emoji = ("🟢" if n["sentiment"] == "Positive"
                                     else "🔴" if n["sentiment"] == "Negative" else "⚪")
                            title_trunc = n["title"][:80] + ("…" if len(n["title"]) > 80 else "")
                            st.markdown(f"{emoji} **{title_trunc}**")
                            st.caption(f"_{n['source']} · {n['date']}_")

            st.markdown("---")
            st.markdown("### 🤖 Generative AI Report")
            insight_query = st.text_area(
                "Describe what you'd like analysed:",
                placeholder="e.g. Compare momentum signals across my selected stocks and suggest entry points.",
                height=90,
            )
            if st.button("⚡ Generate AI Report"):
                if not insight_query.strip():
                    st.error("Please enter a query.")
                else:
                    with st.spinner("Generating analysis…"):
                        report = generate_comprehensive_report(
                            selected_stocks, insight_query, st.session_state.risk_profile)
                    st.markdown(report)
                    if st.session_state.voice_enabled and st.session_state.auto_speak:
                        speak_text(report[:600])
                    st.session_state.query_logs.append({
                        "timestamp": datetime.now().isoformat(),
                        "stocks":    selected_stocks,
                        "query":     insight_query,
                        "preview":   report[:300],
                    })

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — PORTFOLIO ANALYSIS
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 💼 Portfolio Tracking & Analysis")

        # ── Add / Remove Holdings Panel (always visible at top) ───────────────
        st.markdown("""
        <div style="background:var(--bg-elevated);border:1px solid var(--border);border-radius:var(--radius);padding:1.2rem 1.4rem;margin-bottom:1rem;">
            <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:1.4px;color:var(--text-dim);font-weight:500;margin-bottom:.6rem;">➕ Add or Update a Holding</div>
        </div>""", unsafe_allow_html=True)

        with st.form("add_holding_form", clear_on_submit=True):
            fc1, fc2, fc3 = st.columns([2, 1.2, 1.2])
            with fc1:
                symbol_input = st.text_input(
                    "Stock Ticker Symbol",
                    placeholder="e.g. AAPL, RELIANCE.NS, NVDA, TSLA…",
                    help="Enter any valid Yahoo Finance ticker. Append .NS for Indian NSE stocks (e.g. HDFCBANK.NS).",
                    key="port_symbol_text"
                )
            with fc2:
                shares_input = st.number_input(
                    "Number of Shares",
                    min_value=0.001,
                    value=10.0,
                    step=0.5,
                    format="%.3f",
                    key="port_shares_form"
                )
            with fc3:
                buy_input = st.number_input(
                    "Avg Purchase Price",
                    min_value=0.01,
                    value=100.0,
                    step=0.5,
                    format="%.2f",
                    help="Enter price in the stock's native currency (USD for US stocks, ₹ for .NS stocks).",
                    key="port_buy_form"
                )
            add_submitted = st.form_submit_button("➕ Add / Update Holding", use_container_width=True)

        if add_submitted:
            raw = symbol_input.strip().upper()
            if not raw:
                st.error("⚠️ Please enter a ticker symbol before submitting.")
            else:
                # Validate the symbol by fetching 2 days of data
                with st.spinner(f"Validating {raw}…"):
                    try:
                        test = yf.Ticker(raw).history(period="5d")
                        valid = not test.empty
                    except Exception:
                        valid = False
                if not valid:
                    st.error(f"❌ **{raw}** could not be found on Yahoo Finance. Double-check the ticker and try again.")
                else:
                    if "portfolio" not in st.session_state:
                        st.session_state.portfolio = {}
                    action = "Updated" if raw in st.session_state.portfolio else "Added"
                    st.session_state.portfolio[raw] = {
                        "shares": shares_input,
                        "buy_price": buy_input,
                    }
                    st.success(f"✅ **{action}** {raw} — {shares_input:g} shares @ {buy_input:,.2f}")
                    st.rerun()

        # ── Remove Holdings ──────────────────────────────────────────────────
        if st.session_state.get("portfolio"):
            with st.expander("🗑️ Remove a Holding"):
                existing_symbols = list(st.session_state.portfolio.keys())
                del_sym = st.selectbox(
                    "Select asset to remove",
                    options=existing_symbols,
                    key="port_del_sym"
                )
                if st.button("Remove from Portfolio", key="port_del_btn"):
                    del st.session_state.portfolio[del_sym]
                    st.success(f"Removed **{del_sym}** from your portfolio.")
                    st.rerun()

        st.markdown("---")

        # ── Portfolio dashboard (only when holdings exist) ────────────────────
        if not st.session_state.get("portfolio"):
            # Beautiful empty-state
            st.markdown("""
            <div style="text-align:center;padding:3.5rem 2rem;
                        background:var(--bg-elevated);border:1px dashed var(--border);
                        border-radius:var(--radius);margin-top:1rem;">
                <div style="font-size:3rem;margin-bottom:.5rem;">📂</div>
                <div style="font-family:var(--font-display);font-size:1.4rem;
                            color:var(--text-primary);margin-bottom:.5rem;">Your portfolio is empty</div>
                <div style="color:var(--text-muted);font-size:.88rem;max-width:420px;margin:0 auto;">
                    Use the form above to add your first stock holding.
                    Enter any valid ticker — US stocks like <b>AAPL</b>, <b>MSFT</b>,
                    or Indian NSE stocks like <b>RELIANCE.NS</b>, <b>HDFCBANK.NS</b>.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            portfolio = st.session_state.portfolio
            portfolio_symbols = tuple(portfolio.keys())

            # Fetch realtime prices in parallel
            with concurrent.futures.ThreadPoolExecutor() as ex:
                rt_results = {sym: res for sym, res in zip(
                    portfolio_symbols, ex.map(get_realtime_stock_data, portfolio_symbols))}

            # ── Valuation calculations ────────────────────────────────────────
            total_invested_usd = 0.0
            total_value_usd    = 0.0
            total_invested_inr = 0.0
            total_value_inr    = 0.0
            daily_change_value_usd = 0.0
            holdings_rows = []

            for symbol, info in portfolio.items():
                shares    = info["shares"]
                buy_price = info["buy_price"]
                rt        = rt_results.get(symbol, {"price": 0.0, "change": 0.0, "source": "N/A"})
                curr_price  = rt["price"]
                change_pct  = rt["change"]

                cost_native  = shares * buy_price
                value_native = shares * curr_price

                if symbol.endswith(".NS"):
                    cost_inr, value_inr = cost_native, value_native
                    cost_usd  = cost_inr  / USD_INR_RATE
                    value_usd = value_inr / USD_INR_RATE
                    prev_close_usd = (curr_price / (1 + change_pct / 100)) / USD_INR_RATE
                else:
                    cost_usd, value_usd = cost_native, value_native
                    cost_inr  = cost_usd  * USD_INR_RATE
                    value_inr = value_usd * USD_INR_RATE
                    prev_close_usd = curr_price / (1 + change_pct / 100)

                daily_gain_usd = value_usd - (prev_close_usd * shares)
                total_invested_usd     += cost_usd
                total_value_usd        += value_usd
                total_invested_inr     += cost_inr
                total_value_inr        += value_inr
                daily_change_value_usd += daily_gain_usd

                pnl_usd = value_usd - cost_usd
                pnl_pct = (pnl_usd / cost_usd * 100) if cost_usd > 0 else 0.0
                ccy = "₹" if symbol.endswith(".NS") else "$"
                holdings_rows.append({
                    "Company":             COMPANY_NAMES.get(symbol, symbol),
                    "Shares":              shares,
                    "Avg Buy Price":       f"{ccy}{buy_price:,.2f}",
                    "Current Price":       f"{ccy}{curr_price:,.2f}",
                    "Total Cost (USD)":    round(cost_usd, 2),
                    "Current Value (USD)": round(value_usd, 2),
                    "P&L (USD)":           round(pnl_usd, 2),
                    "P&L (%)":             round(pnl_pct, 2),
                })

            total_pnl_usd = total_value_usd - total_invested_usd
            total_pnl_pct = (total_pnl_usd / total_invested_usd * 100) if total_invested_usd > 0 else 0.0
            prev_total    = total_value_usd - daily_change_value_usd
            daily_change_pct = (daily_change_value_usd / prev_total * 100) if prev_total > 0 else 0.0

            # ── KPI Cards ────────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Portfolio Value", f"${total_value_usd:,.2f}",
                      f"₹{total_value_inr:,.0f} (Consolidated)", delta_color="normal")
            c2.metric("Total Invested Capital",  f"${total_invested_usd:,.2f}",
                      f"₹{total_invested_inr:,.0f}", delta_color="off")
            c3.metric("Total P&L", f"${total_pnl_usd:,.2f}",
                      f"{total_pnl_pct:+.2f}%",
                      delta_color="normal" if total_pnl_usd >= 0 else "inverse")
            c4.metric("Today's Gain / Loss", f"${daily_change_value_usd:,.2f}",
                      f"{daily_change_pct:+.2f}%",
                      delta_color="normal" if daily_change_value_usd >= 0 else "inverse")

            st.markdown("---")

            # ── Allocation & Comparison Charts ───────────────────────────────
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_donut = go.Figure(go.Pie(
                    labels=[r["Company"] for r in holdings_rows],
                    values=[r["Current Value (USD)"] for r in holdings_rows],
                    hole=0.4, hoverinfo="label+percent+value", textinfo="label+percent",
                    marker=dict(line=dict(color="#0e1620", width=2))
                ))
                fig_donut.update_layout(
                    title=dict(text="Asset Allocation (USD Value)", font=dict(size=14, color="#e8edf4")),
                    height=300, margin=dict(t=50, b=10, l=10, r=10), **PLOTLY_THEME)
                st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

            with col_chart2:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=[r["Company"] for r in holdings_rows],
                                         y=[r["Total Cost (USD)"] for r in holdings_rows],
                                         name="Invested Cost", marker_color="#3b82f6"))
                fig_bar.add_trace(go.Bar(x=[r["Company"] for r in holdings_rows],
                                         y=[r["Current Value (USD)"] for r in holdings_rows],
                                         name="Current Value", marker_color="#0fd4b0"))
                fig_bar.update_layout(
                    title=dict(text="Cost vs. Current Value (USD)", font=dict(size=14, color="#e8edf4")),
                    barmode="group", height=300, margin=dict(t=50, b=10, l=10, r=10), **PLOTLY_THEME)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")

            # ── Holdings Table ───────────────────────────────────────────────
            st.markdown("#### 📂 Current Holdings")
            st.dataframe(pd.DataFrame(holdings_rows).set_index("Company"), use_container_width=True)

            st.markdown("---")

            # ── Historical Performance ───────────────────────────────────────
            st.markdown(f"#### 📅 Historical Growth of Portfolio ({analysis_period})")
            with st.spinner("Calculating historical portfolio timeline…"):
                df_closes = get_portfolio_historical_data(portfolio_symbols, analysis_period)
                df_bench  = get_benchmark_historical_data(analysis_period)

                if not df_closes.empty:
                    portfolio_history = calculate_portfolio_historical_value(df_closes, portfolio)

                    if not portfolio_history.empty:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Scatter(
                            x=portfolio_history.index, y=portfolio_history.values,
                            name="Portfolio Value ($)", fill="tozeroy",
                            line=dict(color="#c9a84c", width=2),
                            fillcolor="rgba(201,168,76,0.05)"
                        ))

                        norm_p = (portfolio_history / portfolio_history.iloc[0]) * 100
                        fig_hist_norm = go.Figure()
                        fig_hist_norm.add_trace(go.Scatter(
                            x=norm_p.index, y=norm_p.values,
                            name="Your Portfolio", line=dict(color="#0fd4b0", width=2.5)
                        ))
                        if not df_bench.empty:
                            df_bench_aligned = df_bench.reindex(portfolio_history.index).ffill().bfill()
                            norm_b = (df_bench_aligned / df_bench_aligned.iloc[0]) * 100
                            fig_hist_norm.add_trace(go.Scatter(
                                x=norm_b.index, y=norm_b.values,
                                name="S&P 500 (SPY)",
                                line=dict(color="#3b82f6", width=1.5, dash="dash")
                            ))

                        fig_hist.update_layout(
                            title=dict(text="Consolidated Portfolio Value over Time ($)",
                                       font=dict(size=13, color="#e8edf4")),
                            yaxis_title="USD ($)", hovermode="x unified", height=320, **PLOTLY_THEME)
                        fig_hist_norm.update_layout(
                            title=dict(text="Growth vs. S&P 500 (Base 100)",
                                       font=dict(size=13, color="#e8edf4")),
                            yaxis_title="Indexed (Base 100)", hovermode="x unified", height=320, **PLOTLY_THEME)

                        col_h1, col_h2 = st.columns(2)
                        with col_h1:
                            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
                        with col_h2:
                            st.plotly_chart(fig_hist_norm, use_container_width=True, config={"displayModeBar": False})

                        # Risk & Performance metrics
                        p_metrics = calculate_portfolio_metrics(portfolio_history, df_bench)
                        st.markdown("#### ⚡ Portfolio Risk & Performance Analytics")
                        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
                        cm1.metric("Annualized Return",   f"{p_metrics['annualized_return']*100:.1f}%")
                        cm2.metric("Annualized Volatility", f"{p_metrics['annualized_vol']*100:.1f}%")
                        cm3.metric("Sharpe Ratio",        f"{p_metrics['sharpe_ratio']:.2f}")
                        cm4.metric("Max Drawdown",        f"{p_metrics['max_drawdown']:.1f}%")
                        cm5.metric("Beta vs. SPY",        f"{p_metrics['beta']:.2f}")

                        # MPT Optimization
                        st.markdown("---")
                        st.markdown("#### 🎛️ Modern Portfolio Theory (MPT) Optimization")
                        c_opt1, c_opt2 = st.columns([1.2, 1])
                        with c_opt1:
                            st.caption("Monte Carlo simulation of 1,000 random allocations to construct the Efficient Frontier. The star marks the Max Sharpe (optimal) portfolio.")
                            if st.button("🚀 Run Portfolio Optimization", use_container_width=True):
                                opt_results = run_mpt_optimization(df_closes, num_portfolios=1000)
                                if opt_results["frontier_vol"]:
                                    fig_ef = go.Figure()
                                    fig_ef.add_trace(go.Scatter(
                                        x=opt_results["frontier_vol"], y=opt_results["frontier_ret"],
                                        mode="markers",
                                        marker=dict(color=opt_results["frontier_sharpe"],
                                                    colorscale="Viridis", showscale=True,
                                                    colorbar=dict(title="Sharpe", thickness=12),
                                                    size=5, opacity=0.6),
                                        name="Simulated Portfolios", hoverinfo="text",
                                        text=[f"Return: {r:.1%}<br>Vol: {v:.1%}<br>Sharpe: {s:.2f}"
                                              for r, v, s in zip(opt_results["frontier_ret"],
                                                                  opt_results["frontier_vol"],
                                                                  opt_results["frontier_sharpe"])]
                                    ))
                                    opt_vol = opt_results["frontier_vol"][opt_results["max_sharpe_idx"]]
                                    opt_ret = opt_results["frontier_ret"][opt_results["max_sharpe_idx"]]
                                    fig_ef.add_trace(go.Scatter(
                                        x=[opt_vol], y=[opt_ret], mode="markers+text",
                                        marker=dict(color="#0fd4b0", size=14, symbol="star",
                                                    line=dict(color="#0e1620", width=2)),
                                        text=["Max Sharpe (Optimal)"], textposition="top center",
                                        name="Optimal Portfolio"
                                    ))
                                    total_val_opt = sum(
                                        portfolio[s]["shares"]
                                        * rt_results.get(s, {"price": 0})["price"]
                                        / (USD_INR_RATE if s.endswith(".NS") else 1)
                                        for s in opt_results["symbols"]
                                    )
                                    current_weights = [
                                        (portfolio[s]["shares"]
                                         * rt_results.get(s, {"price": 0})["price"]
                                         / (USD_INR_RATE if s.endswith(".NS") else 1))
                                        / total_val_opt if total_val_opt > 0 else 0
                                        for s in opt_results["symbols"]
                                    ]
                                    usd_closes = df_closes.copy()
                                    for col in usd_closes.columns:
                                        if col.endswith(".NS"):
                                            usd_closes[col] /= USD_INR_RATE
                                    cr = usd_closes.pct_change().dropna()
                                    curr_port_ret = np.sum(cr.mean() * 252 * current_weights)
                                    curr_port_vol = np.sqrt(np.dot(
                                        np.array(current_weights).T,
                                        np.dot(cr.cov() * 252, current_weights)))
                                    fig_ef.add_trace(go.Scatter(
                                        x=[curr_port_vol], y=[curr_port_ret], mode="markers+text",
                                        marker=dict(color="#e05c6a", size=12, symbol="circle",
                                                    line=dict(color="#0e1620", width=2)),
                                        text=["Current Portfolio"], textposition="bottom center",
                                        name="Current Portfolio"
                                    ))
                                    fig_ef.update_layout(
                                        title=dict(text="Efficient Frontier (Monte Carlo)",
                                                   font=dict(size=13, color="#e8edf4")),
                                        xaxis_title="Annualized Volatility",
                                        yaxis_title="Expected Annualized Return",
                                        height=360, margin=dict(t=50, b=10, l=10, r=10), **PLOTLY_THEME)
                                    st.plotly_chart(fig_ef, use_container_width=True,
                                                    config={"displayModeBar": False})
                                    with c_opt2:
                                        st.markdown("##### ⚖️ Rebalancing Suggestions")
                                        opt_data = []
                                        for s, w_opt in opt_results["optimal_weights"].items():
                                            idx_s  = opt_results["symbols"].index(s)
                                            w_curr = current_weights[idx_s]
                                            opt_data.append({"Asset": s,
                                                             "Current": f"{w_curr*100:.1f}%",
                                                             "Optimal": f"{w_opt*100:.1f}%",
                                                             "Δ": f"{(w_opt-w_curr)*100:+.1f}%"})
                                        st.dataframe(pd.DataFrame(opt_data).set_index("Asset"),
                                                     use_container_width=True)

                        # Gemini AI Review
                        st.markdown("---")
                        st.markdown("#### 🧠 Gemini AI Portfolio Review")
                        col_ai1, col_ai2 = st.columns([2, 1])
                        with col_ai1:
                            st.write("Let Gemini evaluate your holdings, risk metrics, and diversification against your risk appetite.")
                        with col_ai2:
                            run_ai_review = st.button("Generate AI Portfolio Analysis", use_container_width=True)
                        if run_ai_review:
                            with st.spinner("Gemini is analysing your portfolio…"):
                                review_text = generate_ai_portfolio_review(
                                    portfolio, p_metrics, st.session_state.risk_profile)
                            st.markdown("---")
                            st.markdown("### 🤖 Generative AI Portfolio Assessment")
                            st.markdown(review_text)
                            st.session_state.query_logs.append({
                                "timestamp": datetime.now().isoformat(),
                                "stocks":    list(portfolio.keys()),
                                "query":     "Generate AI Portfolio Analysis",
                                "preview":   review_text[:300],
                            })
                else:
                    st.warning("Could not fetch historical prices for this portfolio.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab4:
        if len(selected_stocks) < 2:
            st.info("Select at least 2 stocks for comparison.")
        else:
            st.markdown("### 🔍 Multi-Stock Comparison")
            fig, metrics_df = compare_stocks(tuple(selected_stocks), analysis_period)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            if not metrics_df.empty:
                st.markdown("#### Metrics")
                st.dataframe(
                    metrics_df.style.background_gradient(cmap="RdYlGn", axis=0),
                    use_container_width=True,
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("### 💡 Personalised Investment Recommendations")
        col_sip, col_recs = st.columns([1.2, 1])

        with col_sip:
            st.markdown("#### 📊 SIP Calculator")
            monthly = st.number_input("Monthly Investment (₹)", 1000, 100000, 5000, 500)
            years   = st.slider("Investment Horizon (years)", 1, 30, 10)

            if st.button("📈 Project SIP Returns"):
                proj      = calculate_sip_projection(monthly, years, st.session_state.risk_profile)
                total_inv = monthly * years * 12
                final_val = proj["value"][-1]
                profit    = final_val - total_inv
                gain_pct  = profit / total_inv * 100

                c1, c2, c3 = st.columns(3)
                c1.metric("Invested",   f"₹{total_inv:,.0f}")
                c2.metric("Est. Value", f"₹{final_val:,.0f}")
                c3.metric("Gain",       f"₹{profit:,.0f}", f"+{gain_pct:.1f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=proj["years"], y=proj["invested"],
                                         name="Invested", fill="tozeroy",
                                         line=dict(color="#3b82f6", width=2)))
                fig.add_trace(go.Scatter(x=proj["years"], y=proj["value"],
                                         name="Est. Value", fill="tonexty",
                                         line=dict(color="#0fd4b0", width=2)))
                fig.update_layout(title="SIP Growth Projection", xaxis_title="Years",
                                  yaxis_title="₹", height=300, **PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            sip_type = st.selectbox("SIP Duration", [
                "Short Term (2 to 5 years)",
                "Medium Term (5 to 10 years)",
                "Long Term (10 or more years)",
            ])
            if st.button("🤖 Get AI SIP Suggestions"):
                # Sanitize sip_type: strip special chars that can confuse the model
                sip_type_clean = (sip_type
                                  .replace("–", " to ")
                                  .replace("—", " to ")
                                  .replace("+", " or more"))

                with st.spinner("Consulting AI… please wait"):
                    system_instruction = (
                        "You are a certified mutual fund expert advisor in India. "
                        "Always respond with complete, well-formatted markdown. "
                        "Never truncate your response. Always finish all 5 fund entries."
                    )
                    prompt = f"""List exactly 5 SIP mutual funds suitable for Indian investors with the following profile:
- Risk appetite: {st.session_state.risk_profile}
- Investment duration: {sip_type_clean}

For EACH of the 5 funds, provide:
1. Fund name (bold)
2. Fund category
3. Approximate 5-year CAGR (%)
4. Risk level
5. One-line rationale

Format each fund like this example:

**1. Mirae Asset Large Cap Fund**
- Category: Large Cap
- 5-Year CAGR: ~14.5%
- Risk: Moderately High
- Why: Consistent performer with strong large-cap exposure, ideal for steady long-term growth.

---

Now list all 5 funds completely:"""

                    recs_ai = gemini_generate(
                        prompt,
                        max_tokens=1800,
                        system_instruction=system_instruction,
                    )

                # Display result in a styled container
                if recs_ai.startswith("⚠️"):
                    st.error(recs_ai)
                else:
                    st.markdown("#### 🤖 AI SIP Recommendations")
                    st.markdown(recs_ai)
                    st.caption(f"Generated for: {sip_type} · Risk: {st.session_state.risk_profile}")

        with col_recs:
            st.markdown("#### 📂 Recommended Funds")
            recs = get_investment_recommendations(st.session_state.risk_profile)
            for category, funds in recs.items():
                with st.expander(category):
                    for f in funds:
                        risk_color = ("#0fd4b0" if f["risk"] in ("Low", "Very Low")
                                      else "#c9a84c" if f["risk"] == "Medium"
                                      else "#e05c6a")
                        st.markdown(f"""
                        <div style="margin-bottom:.6rem;">
                            <div style="font-weight:600;font-size:.9rem;color:var(--text-primary);">{f['name']}</div>
                            <div style="display:flex;gap:1rem;margin:.2rem 0;font-size:.78rem;font-family:var(--font-mono);">
                                <span style="color:#0fd4b0;">↑ {f['returns']}% 1Y</span>
                                <span style="color:{risk_color};">⚠ {f['risk']}</span>
                                <span style="color:var(--text-muted);">⚙ {f['allocation']}% alloc</span>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        st.progress(f["allocation"] / 100)

        # Mini SIP chat
        st.markdown("---")
        st.markdown("#### 💬 Ask the SIP Assistant")

        # Capture input value before form clearing on submit
        with st.form("sip_form", clear_on_submit=True):
            sip_q    = st.text_input("Question", placeholder="Which SIP is best for 7 years?")
            sip_send = st.form_submit_button("Send")

        if sip_send and sip_q.strip():  # sip_q still holds value at this point
            st.session_state.sip_chat.append({
                "role": "user", "content": sip_q,
                "ts": datetime.now().strftime("%H:%M"),
            })
            with st.spinner("…"):
                ans = generate_chat_response(sip_q, {
                    "selected_stocks": selected_stocks,
                    "risk_profile":    st.session_state.risk_profile,
                    "chat_history":    st.session_state.sip_chat[-4:],
                })
            st.session_state.sip_chat.append({
                "role": "assistant", "content": ans,
                "ts": datetime.now().strftime("%H:%M"),
            })

        for msg in st.session_state.sip_chat[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 6 — CURRENCY CONVERTER
    # ════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown("### 💱 Live Currency Converter")
        st.caption("Real-time exchange rates powered by Yahoo Finance · Rates update every 10 minutes")
        st.markdown("---")

        # ── Main Converter ───────────────────────────────────────────────
        conv_col, info_col = st.columns([1.4, 1])

        with conv_col:
            st.markdown("""
            <div style="background:var(--bg-elevated);border:1px solid var(--border);
                        border-radius:var(--radius);padding:1.6rem 1.8rem;">
                <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:1.4px;
                            color:var(--text-dim);margin-bottom:1rem;">Convert Amount</div>
            </div>""", unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns([1.2, 2, 2])
            with cc1:
                amount = st.number_input(
                    "Amount", min_value=0.0, value=1000.0, step=100.0,
                    format="%.2f", key="fx_amount"
                )
            with cc2:
                from_ccy = st.selectbox(
                    "From",
                    CURRENCY_LIST,
                    index=CURRENCY_LIST.index("USD"),
                    format_func=lambda x: f"{CURRENCY_FLAGS.get(x, '')} {x}",
                    key="fx_from"
                )
            with cc3:
                to_ccy = st.selectbox(
                    "To",
                    CURRENCY_LIST,
                    index=CURRENCY_LIST.index("INR"),
                    format_func=lambda x: f"{CURRENCY_FLAGS.get(x, '')} {x}",
                    key="fx_to"
                )

            with st.spinner("Fetching live rate…"):
                rate = get_fx_rate(from_ccy, to_ccy)
                rate_inv = get_fx_rate(to_ccy, from_ccy)

            if rate > 0:
                converted = amount * rate
                sym_from = CURRENCY_SYMBOLS.get(from_ccy, from_ccy)
                sym_to   = CURRENCY_SYMBOLS.get(to_ccy,   to_ccy)
                flag_f   = CURRENCY_FLAGS.get(from_ccy, "")
                flag_t   = CURRENCY_FLAGS.get(to_ccy, "")

                st.markdown(f"""
                <div style="margin-top:1.4rem;padding:1.4rem 1.6rem;
                            background:linear-gradient(135deg,#0e1a2b 0%,#0a1520 100%);
                            border:1px solid var(--border);border-radius:var(--radius);">
                    <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.8rem;">
                        <span style="font-size:1.8rem;">{flag_f}</span>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:1rem;
                                        color:var(--text-muted);">{sym_from}{amount:,.2f} {from_ccy}</div>
                            <div style="color:var(--text-dim);font-size:.75rem;">You send</div>
                        </div>
                    </div>
                    <div style="border-left:2px solid var(--accent-gold);
                                padding-left:.8rem;margin:.6rem 0 .6rem 1.2rem;
                                color:var(--accent-gold);font-size:.78rem;font-family:var(--font-mono);">
                        1 {from_ccy} = {rate:,.6g} {to_ccy}
                    </div>
                    <div style="display:flex;align-items:center;gap:.8rem;">
                        <span style="font-size:1.8rem;">{flag_t}</span>
                        <div>
                            <div style="font-family:var(--font-mono);font-size:2rem;
                                        color:var(--accent-teal);font-weight:600;">
                                {sym_to}{converted:,.2f}
                            </div>
                            <div style="color:var(--text-dim);font-size:.75rem;">
                                {to_ccy} · Live rate
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Inverse rate
                st.markdown(f"""
                <div style="margin-top:.8rem;padding:.7rem 1rem;
                            background:var(--bg-elevated);border:1px solid var(--border);
                            border-radius:8px;font-family:var(--font-mono);font-size:.82rem;
                            color:var(--text-muted);">
                    <span style="color:var(--text-dim);">Inverse: </span>
                    1 {to_ccy} = {rate_inv:,.6g} {from_ccy} &nbsp;|&nbsp;
                    <span style="color:var(--text-dim);">Updated: </span>
                    {datetime.now().strftime('%H:%M:%S')}
                </div>""", unsafe_allow_html=True)
            else:
                st.error(f"❌ Could not fetch rate for {from_ccy}/{to_ccy}. Try a different pair.")

        # ── Popular Rates Panel (batch-fetched & cached) ─────────────────
        with info_col:
            st.markdown("#### 🌐 Popular Rates vs USD")
            for ccy, name, r in get_popular_fx_rates_cached():
                flag = CURRENCY_FLAGS.get(ccy, "")
                sym  = CURRENCY_SYMBOLS.get(ccy, ccy)
                if r > 0:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                                padding:.5rem .7rem;margin-bottom:.3rem;
                                background:var(--bg-elevated);border:1px solid var(--border);
                                border-radius:8px;">
                        <span style="font-size:.85rem;color:var(--text-primary);">{flag} {ccy}</span>
                        <span style="font-size:.72rem;color:var(--text-muted);">{name}</span>
                        <span style="font-family:var(--font-mono);font-size:.9rem;
                                    color:var(--accent-gold);">{sym}{r:,.4g}</span>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Historical Chart ───────────────────────────────────────────────
        st.markdown(f"#### 📅 Historical Rate: {from_ccy} → {to_ccy}")
        hist_period_map = {
            "1 Week": "5d", "1 Month": "1mo", "3 Months": "3mo",
            "6 Months": "6mo", "1 Year": "1y",
        }
        hc1, hc2 = st.columns([3, 1])
        with hc2:
            hist_period = st.selectbox(
                "Period", list(hist_period_map.keys()), index=1, key="fx_hist_period"
            )
        fx_hist = get_fx_history(from_ccy, to_ccy, hist_period_map[hist_period])

        if not fx_hist.empty:
            high_rate = fx_hist["Rate"].max()
            low_rate  = fx_hist["Rate"].min()
            avg_rate  = fx_hist["Rate"].mean()
            pct_change = (fx_hist["Rate"].iloc[-1] - fx_hist["Rate"].iloc[0]) / fx_hist["Rate"].iloc[0] * 100

            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric(f"Current Rate", f"{rate:,.6g}")
            sm2.metric(f"Period High",  f"{high_rate:,.6g}")
            sm3.metric(f"Period Low",   f"{low_rate:,.6g}")
            sm4.metric(f"Period Change", f"{pct_change:+.2f}%",
                       delta_color="normal" if pct_change >= 0 else "inverse")

            fig_fx = go.Figure()
            fig_fx.add_trace(go.Scatter(
                x=fx_hist.index,
                y=fx_hist["Rate"],
                mode="lines",
                name=f"{from_ccy}/{to_ccy}",
                fill="tozeroy",
                line=dict(color="#c9a84c", width=2),
                fillcolor="rgba(201,168,76,0.06)"
            ))
            # Add avg line
            fig_fx.add_hline(
                y=avg_rate,
                line_dash="dash",
                line_color="#3b82f6",
                annotation_text=f"Avg: {avg_rate:,.4g}",
                annotation_position="top right",
                annotation_font_color="#3b82f6"
            )
            fig_fx.update_layout(
                title=dict(
                    text=f"{from_ccy} → {to_ccy} Exchange Rate ({hist_period})",
                    font=dict(size=13, color="#e8edf4")
                ),
                xaxis_title="Date",
                yaxis_title=f"Rate ({to_ccy} per {from_ccy})",
                hovermode="x unified",
                height=380,
                **PLOTLY_THEME
            )
            with hc1:
                pass  # spacer
            st.plotly_chart(fig_fx, use_container_width=True, config={"displayModeBar": False})
        else:
            st.warning(f"Historical data unavailable for {from_ccy}/{to_ccy}. Try a different pair.")

        st.markdown("---")

        # ── Cross-Rate Table (batch-fetched & cached) ──────────────────────
        with st.expander("📈 Cross-Rate Table (Major Currencies vs USD)"):
            cross_rows = get_cross_rates_cached()
            if cross_rows:
                st.dataframe(
                    pd.DataFrame(cross_rows).set_index("Currency"),
                    use_container_width=True
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 7 — CHAT
    # ════════════════════════════════════════════════════════════════════════
    with tab7:
        st.markdown("### 💬 Financial AI Assistant")
        ctrl = st.columns([3, 1, 1, 1])

        with ctrl[0]:
            if st.session_state.voice_enabled:
                if st.button("🎤 Voice Input", type="primary", use_container_width=True):
                    voice_text = listen_for_speech(timeout=12)
                    if voice_text:
                        st.session_state.chat_history.append({
                            "role": "user", "content": voice_text,
                            "ts": datetime.now().strftime("%H:%M"), "voice": True,
                        })
                        with st.spinner("Thinking…"):
                            resp = generate_chat_response(voice_text, {
                                "selected_stocks": selected_stocks,
                                "risk_profile":    st.session_state.risk_profile,
                                "chat_history":    st.session_state.chat_history[-4:],
                            })
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": resp,
                            "ts": datetime.now().strftime("%H:%M"), "voice": True,
                        })
                        if st.session_state.auto_speak:
                            speak_text(resp)
                        st.rerun()

        with ctrl[1]:
            if st.session_state.voice_enabled and st.session_state.chat_history:
                last_ai = next((m for m in reversed(st.session_state.chat_history)
                                if m["role"] == "assistant"), None)
                if last_ai and st.button("🔊 Replay", use_container_width=True):
                    speak_text(last_ai["content"])

        with ctrl[2]:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with ctrl[3]:
            st.caption(f"{len(st.session_state.chat_history)} messages")

        if not st.session_state.chat_history:
            st.markdown("**Quick prompts:**")
            qcols = st.columns(4)
            prompts = [
                "📊 Market summary",
                "🎯 Advice for my risk profile",
                "📖 Explain SIP basics",
                "🔍 Review my stocks",
            ]
            for i, p in enumerate(prompts):
                with qcols[i]:
                    if st.button(p, key=f"qp_{i}", use_container_width=True):
                        st.session_state.chat_history.append({
                            "role": "user", "content": p.split(" ", 1)[1],
                            "ts": datetime.now().strftime("%H:%M"), "voice": False,
                        })
                        st.rerun()

        for i, msg in enumerate(st.session_state.chat_history[-12:]):
            icon = "🎤" if msg.get("voice") else ("👤" if msg["role"] == "user" else "🤖")
            with st.chat_message(msg["role"]):
                c1, c2 = st.columns([18, 2])
                with c1:
                    st.markdown(
                        f"<span style='color:var(--text-dim);font-size:.7rem;'>{icon} {msg['ts']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(msg["content"])
                with c2:
                    if st.session_state.voice_enabled and msg["role"] == "assistant":
                        if st.button("🔊", key=f"spk_{i}"):
                            speak_text(msg["content"])

        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([6, 1])
            with c1:
                user_input = st.text_input(
                    "Ask anything about markets, investing, or your portfolio…",
                    label_visibility="collapsed",
                    placeholder="e.g. What's the outlook for AAPL this quarter?",
                )
            with c2:
                send = st.form_submit_button("Send ➜", use_container_width=True)

        if send and user_input.strip():
            st.session_state.chat_history.append({
                "role": "user", "content": user_input,
                "ts": datetime.now().strftime("%H:%M"), "voice": False,
            })
            with st.spinner("Thinking…"):
                response = generate_chat_response(user_input, {
                    "selected_stocks": selected_stocks,
                    "risk_profile":    st.session_state.risk_profile,
                    "chat_history":    st.session_state.chat_history[-5:],
                })
            st.session_state.chat_history.append({
                "role": "assistant", "content": response,
                "ts": datetime.now().strftime("%H:%M"), "voice": False,
            })
            if st.session_state.voice_enabled and st.session_state.auto_speak:
                speak_text(response)
            st.rerun()

    # Footer 
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center;color:var(--text-dim);font-size:.72rem;'>"
        f"FinAdvisor AI · Powered by Google Gemini · "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"Data: Alpha Vantage & Yahoo Finance · Not financial advice.</div>",
        unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()