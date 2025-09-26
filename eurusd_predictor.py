from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Sentiment (VADER)
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Models: prefer xgboost if available, otherwise sklearn RandomForest
try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor  # type: ignore
    _HAS_XGB = False

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("forex_predictor.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Files & dirs
DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"
PRICES_CSV = os.path.join(DATA_DIR, "prices.csv")
NEWS_CSV = os.path.join(DATA_DIR, "news.csv")
CALENDAR_CSV = os.path.join(DATA_DIR, "calendar.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "dataset_features.csv")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
FEATCOLS_PATH = os.path.join(ARTIFACT_DIR, "features.json")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(ARTIFACT_DIR, "config.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------
# Configuration helpers
# ---------------------------
def load_config() -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    default_config: Dict[str, Any] = {
        "symbol": "EURUSD=X",
        "interval": "1d",
        "test_size_days": 60,
        "news_apis": {
            "newsapi": {"enabled": False, "api_key": "", "sources": "reuters,bloomberg", "page_size": 100},
            "alphavantage": {"enabled": False, "api_key": ""},
            "finnhub": {"enabled": False, "api_key": ""},
        },
        "economic_calendar": {  # REPLACED Alpha Vantage with FRED
            "financialmodelingprep": {"enabled": True, "api_key": "demo"},
            "fred": {"enabled": True, "api_key": "a457b5721ff407e9f2fea7766ad2674d"},  # Free FRED API key
        },
        "model_params": {
            "xgb": {"n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42},
            "rf": {"n_estimators": 400, "max_depth": 8, "random_state": 42},
        },
        "feature_lags": [1, 2, 3, 5, 7],
        "threshold": 0.0,
    }

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    default_config.update(loaded)
                    logger.info("Loaded configuration from file")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")

    return default_config

def save_config(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Saved configuration")
    except Exception as e:
        logger.error(f"Failed to save config: {e}")


# ---------------------------
# Utilities
# ---------------------------
def ts_now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_vader() -> None:
    """Ensure VADER lexicon is available."""
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except Exception:
        nltk.download("vader_lexicon")


def parse_datetime_guess(s: Any) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(s):
            return None
        return pd.to_datetime(s, utc=True, errors="coerce")
    except Exception:
        return None


def validate_dataframe(df: pd.DataFrame, required_cols: List[str], data_type: str) -> bool:
    if df is None or len(df) == 0:
        logger.warning(f"{data_type} dataframe is empty")
        return False
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"{data_type} missing columns: {missing}")
        return False
    
    return True


# ---------------------------
# Ingest: Prices (Yahoo Finance)
# ---------------------------
def ingest_prices(symbol: str = "EURUSD=X", interval: str = "1d", start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download historical prices (yfinance) and save to CSV."""
    logger.info(f"Downloading prices symbol={symbol} interval={interval} start={start} end={end}")
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=interval, start=start, end=end, auto_adjust=False)
        
        if df is None or df.empty:
            raise RuntimeError("No price data returned by yfinance")

        # Reset index and rename columns
        df = df.reset_index()
        df = df.rename(columns={'Date': 'timestamp'})
        
        # Ensure timestamp is datetime (UTC)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        
        # Select and rename columns we need
        df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        logger.info(f"Successfully downloaded {len(df)} rows")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Save to CSV
        if os.path.exists(PRICES_CSV):
            existing = pd.read_csv(PRICES_CSV)
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
            if not df.empty and not existing.empty:
                min_new = df["timestamp"].min()
                existing = existing[existing["timestamp"] < min_new]
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

        df.to_csv(PRICES_CSV, index=False)
        logger.info(f"Saved prices -> {PRICES_CSV} rows={len(df)}")
        return df

    except Exception as e:
        logger.error(f"Failed to download prices: {e}")
        if os.path.exists(PRICES_CSV):
            logger.info("Returning existing price CSV")
            return pd.read_csv(PRICES_CSV)
        raise


# ---------------------------
# News ingestion (NewsAPI, AlphaVantage, Finnhub)
# ---------------------------
@dataclass
class NewsItem:
    timestamp: pd.Timestamp
    source: str
    url: str
    headline: str


def fetch_news_from_newsapi(api_key: str, sources: str = "reuters,bloomberg", page_size: int = 100, days_back: int = 7) -> List[NewsItem]:
    base_url = "https://newsapi.org/v2/everything"
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "apiKey": api_key,
        "sources": sources,
        "q": "forex OR currency OR EUR USD OR FX",
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": "en",
    }
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items: List[NewsItem] = []
        for art in data.get("articles", []):
            ts = parse_datetime_guess(art.get("publishedAt"))
            if ts is None:
                continue
            source = art.get("source", {}).get("name", "Unknown")
            items.append(NewsItem(timestamp=ts, source=source, url=art.get("url", ""), headline=art.get("title", "")))
        logger.info(f"Fetched {len(items)} from NewsAPI")
        return items
    except Exception as e:
        logger.error(f"NewsAPI fetch failed: {e}")
        return []


def fetch_news_from_alphavantage(api_key: str, tickers: str = "EURUSD") -> List[NewsItem]:
    base_url = "https://www.alphavantage.co/query"
    params = {"function": "NEWS_SENTIMENT", "tickers": tickers, "apikey": api_key, "sort": "LATEST"}
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items: List[NewsItem] = []
        for feed in data.get("feed", []):
            ts = parse_datetime_guess(feed.get("time_published"))
            if ts is None:
                continue
            items.append(NewsItem(timestamp=ts, source=feed.get("source", "Unknown"), url=feed.get("url", ""), headline=feed.get("title", "")))
        logger.info(f"Fetched {len(items)} from AlphaVantage")
        return items
    except Exception as e:
        logger.error(f"AlphaVantage fetch failed: {e}")
        return []


def fetch_news_from_finnhub(api_key: str) -> List[NewsItem]:
    base_url = "https://finnhub.io/api/v1/news"
    params = {"category": "forex", "token": api_key}
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items: List[NewsItem] = []
        for art in data:
            # Finnhub provides unix timestamp in 'datetime'
            ts = None
            if "datetime" in art:
                try:
                    ts = pd.to_datetime(int(art["datetime"]), unit="s", utc=True)
                except Exception:
                    ts = None
            if ts is None:
                continue
            items.append(NewsItem(timestamp=ts, source=art.get("source", "Unknown"), url=art.get("url", ""), headline=art.get("headline", "")))
        logger.info(f"Fetched {len(items)} from Finnhub")
        return items
    except Exception as e:
        logger.error(f"Finnhub fetch failed: {e}")
        return []


def ingest_news(cfg: Dict[str, Any]) -> pd.DataFrame:
    news_config = cfg.get("news_apis", {})
    all_items: List[NewsItem] = []

    if news_config.get("newsapi", {}).get("enabled", False):
        nc = news_config["newsapi"]
        all_items.extend(fetch_news_from_newsapi(nc.get("api_key", ""), nc.get("sources", "reuters,bloomberg"), nc.get("page_size", 100)))

    if news_config.get("alphavantage", {}).get("enabled", False):
        nc = news_config["alphavantage"]
        all_items.extend(fetch_news_from_alphavantage(nc.get("api_key", "")))

    if news_config.get("finnhub", {}).get("enabled", False):
        nc = news_config["finnhub"]
        all_items.extend(fetch_news_from_finnhub(nc.get("api_key", "")))

    if not all_items:
        logger.warning("No news items fetched. Check API keys / configuration.")
        return pd.DataFrame()

    df = pd.DataFrame([{"timestamp": it.timestamp, "source": it.source, "url": it.url, "headline": it.headline} for it in all_items])
    # merge with existing dedup by headline+timestamp
    if os.path.exists(NEWS_CSV):
        ex = pd.read_csv(NEWS_CSV)
        ex["timestamp"] = pd.to_datetime(ex["timestamp"], utc=True, errors="coerce")
        df = pd.concat([ex, df], ignore_index=True)
        df = df.drop_duplicates(subset=["headline", "timestamp"]).sort_values("timestamp")
    df.to_csv(NEWS_CSV, index=False)
    logger.info(f"Saved news -> {NEWS_CSV} rows={len(df)}")
    return df


# ---------------------------
# Calendar ingestion & features - REPLACED Alpha Vantage with FRED
# ---------------------------
def fetch_calendar_from_fred(api_key: str) -> pd.DataFrame:
    """Fetch economic indicators from FRED (Federal Reserve Economic Data)"""
    events = []
    
    # FRED series IDs for key economic indicators with their importance
    fred_series = {
        "GDP": {"id": "GDP", "importance": "High", "name": "Gross Domestic Product"},
        "CPI": {"id": "CPIAUCSL", "importance": "High", "name": "Consumer Price Index"},
        "UNRATE": {"id": "UNRATE", "importance": "High", "name": "Unemployment Rate"},
        "PPI": {"id": "PPIACO", "importance": "Medium", "name": "Producer Price Index"},
        "RETAIL_SALES": {"id": "RSAFS", "importance": "Medium", "name": "Retail Sales"},
        "INDUSTRIAL_PROD": {"id": "INDPRO", "importance": "Medium", "name": "Industrial Production"},
        "HOUSING_STARTS": {"id": "HOUST", "importance": "Medium", "name": "Housing Starts"},
        "FEDFUNDS": {"id": "FEDFUNDS", "importance": "High", "name": "Federal Funds Rate"},
    }
    
    for series_name, series_info in fred_series.items():
        try:
            # Get series info and latest observations
            series_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_info["id"],
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5  # Get last 5 observations
            }
            
            resp = requests.get(series_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            observations = data.get("observations", [])
            for obs in observations:
                ts = parse_datetime_guess(obs.get("date"))
                if ts is None:
                    continue
                    
                # Only include recent data (last 2 years)
                if ts < (datetime.now(timezone.utc) - timedelta(days=365*2)):
                    continue
                    
                events.append({
                    "timestamp": ts,
                    "event": f"FRED {series_info['name']} Release",
                    "actual": obs.get("value"),
                    "forecast": None,
                    "previous": None,
                    "importance": series_info["importance"],
                    "source": "FRED"
                })
                
        except Exception as e:
            logger.warning(f"Failed to fetch FRED series {series_info['id']}: {e}")
            continue
    
    df = pd.DataFrame(events)
    if not df.empty:
        df = df.sort_values("timestamp", ascending=False)
    logger.info(f"Fetched {len(df)} calendar events from FRED")
    return df


def fetch_calendar_from_financialmodelingprep(api_key: str) -> pd.DataFrame:
    """Fetch economic calendar from Financial Modeling Prep"""
    base_url = "https://financialmodelingprep.com/api/v3/economic_calendar"
    params = {"apikey": api_key}
    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        events = []
        for ev in data:
            ts = parse_datetime_guess(ev.get("date"))
            if ts is None:
                continue
            events.append(
                {
                    "timestamp": ts,
                    "event": ev.get("event", ""),
                    "actual": ev.get("actual"),
                    "forecast": ev.get("forecast"),
                    "previous": ev.get("previous"),
                    "importance": ev.get("importance", "Medium"),
                    "source": "FinancialModelingPrep"
                }
            )
        df = pd.DataFrame(events)
        logger.info(f"Fetched calendar events from FinancialModelingPrep: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch FinancialModelingPrep calendar: {e}")
        return pd.DataFrame()


def ingest_calendar(cfg: Dict[str, Any]) -> pd.DataFrame:
    calendar_config = cfg.get("economic_calendar", {})
    df_list: List[pd.DataFrame] = []
    
    # Use FRED instead of FinancialModelingPrep
    if calendar_config.get("fred", {}).get("enabled", False):
        api_cfg = calendar_config["fred"]
        try:
            cal = fetch_calendar_from_fred(api_cfg.get("api_key", ""))
            if not cal.empty:
                df_list.append(cal)
                logger.info(f"Successfully fetched {len(cal)} events from FRED")
            else:
                logger.warning("FRED returned empty calendar data")
        except Exception as e:
            logger.error(f"FRED calendar fetch failed: {e}")

    # Skip FinancialModelingPrep if it's causing issues
    if calendar_config.get("financialmodelingprep", {}).get("enabled", False):
        logger.info("Skipping FinancialModelingPrep due to previous errors")
        # Optional: you can disable it permanently in config
        # cfg["economic_calendar"]["financialmodelingprep"]["enabled"] = False
        # save_config(cfg)

    if not df_list:
        logger.warning("No calendar data fetched. Creating empty calendar.")
        return pd.DataFrame(columns=["timestamp", "event", "actual", "forecast", "previous", "importance", "source"])

    df = pd.concat(df_list, ignore_index=True)
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    if os.path.exists(CALENDAR_CSV):
        try:
            ex = pd.read_csv(CALENDAR_CSV)
            ex["timestamp"] = pd.to_datetime(ex["timestamp"], utc=True, errors="coerce")
            df = pd.concat([ex, df], ignore_index=True)
            df = df.drop_duplicates(subset=["timestamp", "event"])
        except Exception as e:
            logger.warning(f"Error reading existing calendar file: {e}")
            
    df.to_csv(CALENDAR_CSV, index=False)
    logger.info(f"Saved calendar -> {CALENDAR_CSV} rows={len(df)}")
    return df


# ---------------------------
# NLP: sentiment aggregation
# ---------------------------
def compute_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    ensure_vader()
    sia = SentimentIntensityAnalyzer()
    df = df_news.copy()
    df["headline"] = df["headline"].astype(str)
    df["sent_compound"] = df["headline"].apply(lambda t: sia.polarity_scores(t)["compound"])
    df["date"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("D")
    agg = (
        df.groupby("date")
        .agg(
            news_count=("headline", "count"),
            sent_mean=("sent_compound", "mean"),
            sent_pos_share=("sent_compound", lambda x: (x > 0).mean()),
            sent_neg_share=("sent_compound", lambda x: (x < 0).mean()),
        )
        .reset_index()
    )
    return agg


# ---------------------------
# Calendar feature engineering
# ---------------------------
EVENT_MAP = {
    "gross domestic product": "GDP",
    "gdp": "GDP",
    "consumer price index": "CPI",
    "cpi": "CPI",
    "inflation": "CPI",
    "core cpi": "CoreCPI",
    "unemployment rate": "UnemploymentRate",
    "nonfarm payrolls": "NFP",
    "ecb interest rate decision": "ECBRateDecision",
    "ecb press conference": "ECBPressConf",
    "pmi": "PMI",
    "fred": "FRED",  # Added FRED events
}


def normalize_event_name(name: Any) -> str:
    s = str(name).strip().lower()
    for k, v in EVENT_MAP.items():
        if k in s:
            return v
    return "Other"


def calendar_features(df_cal: pd.DataFrame) -> pd.DataFrame:
    df = df_cal.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.floor("D")
    df["event_key"] = df.get("event", "").astype(str).apply(normalize_event_name)

    # surprise metric (actual - forecast) / |forecast|
    df["forecast_str"] = df["forecast"].astype(str)
    df["surprise"] = np.where(
        df["forecast_str"].str.len() > 0,
        (pd.to_numeric(df["actual"], errors="coerce") - pd.to_numeric(df["forecast"], errors="coerce"))
        / (np.abs(pd.to_numeric(df["forecast"], errors="coerce")) + 1e-9),
        np.nan,
    )

    imp = df["importance"].astype(str).str.lower()
    # avoid errors when importance is NaN: use na=False for contains
    df["importance_num"] = np.select(
        [
            imp.str.contains("high", na=False) | imp.str.contains("3", na=False) | imp.str.contains(r"\*\*\*", na=False),
            imp.str.contains("medium", na=False) | imp.str.contains("2", na=False) | imp.str.contains(r"\*\*", na=False),
            imp.str.contains("low", na=False) | imp.str.contains("1", na=False) | imp.str.contains(r"\*", na=False),
        ],
        [3, 2, 1],
        default=np.nan,
    )

    agg = (
        df.groupby(["date", "event_key"])
        .agg(cal_count=("event", "count"), cal_surprise_mean=("surprise", "mean"), cal_importance_mean=("importance_num", "mean"))
        .reset_index()
    )

    if agg.empty:
        return pd.DataFrame()

    wide = agg.pivot(index="date", columns="event_key", values=["cal_count", "cal_surprise_mean", "cal_importance_mean"])
    # flatten column index
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    return wide


# ---------------------------
# Price feature engineering
# ---------------------------
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Close" not in df.columns:
        raise KeyError("Price DataFrame missing 'Close' column for technical indicators")

    close = df["Close"].astype(float)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss.replace(0, np.nan))
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    sma20 = close.rolling(window=20, min_periods=1).mean()
    std20 = close.rolling(window=20, min_periods=1).std()
    df["bollinger_upper"] = sma20 + (std20 * 2)
    df["bollinger_lower"] = sma20 - (std20 * 2)
    df["bollinger_bandwidth"] = (df["bollinger_upper"] - df["bollinger_lower"]) / (sma20.replace(0, np.nan))

    return df


def price_features(df_prices: pd.DataFrame) -> pd.DataFrame:
    df = df_prices.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.floor("D")
    df = df.sort_values("timestamp")

    df = add_technical_indicators(df)

    df["ret_1d"] = df["Close"].pct_change()
    for w in [5, 10, 20]:
        df[f"ma_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()
        df[f"vol_{w}"] = df["Close"].pct_change().rolling(window=w, min_periods=1).std()

    # avoid zero division if Low is zero
    df["high_low_ratio"] = np.where(df["Low"] == 0, np.nan, df["High"] / df["Low"])
    df["close_open_ratio"] = np.where(df["Open"] == 0, np.nan, df["Close"] / df["Open"])

    feat = (
        df.groupby("date")
        .agg(
            Close=("Close", "last"),
            ret_1d=("ret_1d", "last"),
            ma_5=("ma_5", "last"),
            ma_10=("ma_10", "last"),
            ma_20=("ma_20", "last"),
            vol_5=("vol_5", "last"),
            vol_10=("vol_10", "last"),
            vol_20=("vol_20", "last"),
            rsi=("rsi", "last"),
            macd=("macd", "last"),
            macd_signal=("macd_signal", "last"),
            macd_hist=("macd_hist", "last"),
            bollinger_bandwidth=("bollinger_bandwidth", "last"),
            high_low_ratio=("high_low_ratio", "last"),
            close_open_ratio=("close_open_ratio", "last"),
        )
        .reset_index()
    )

    feat = feat.rename(columns={"Close": "close"})
    feat["y_next_ret"] = feat["ret_1d"].shift(-1)
    return feat


# ---------------------------
# Build dataset
# ---------------------------
def build_dataset() -> pd.DataFrame:
    if not os.path.exists(PRICES_CSV):
        raise FileNotFoundError("prices.csv not found — run ingest_prices first")

    prices = pd.read_csv(PRICES_CSV)
    # news / calendar optional
    news_agg = None
    if os.path.exists(NEWS_CSV):
        news = pd.read_csv(NEWS_CSV)
        news["timestamp"] = pd.to_datetime(news["timestamp"], utc=True, errors="coerce")
        news = news.dropna(subset=["timestamp"])
        if not news.empty:
            news_agg = compute_sentiment(news)

    cal_wide = None
    if os.path.exists(CALENDAR_CSV):
        cal = pd.read_csv(CALENDAR_CSV)
        cal["timestamp"] = pd.to_datetime(cal["timestamp"], utc=True, errors="coerce")
        cal = cal.dropna(subset=["timestamp"])
        if not cal.empty:
            cal_wide = calendar_features(cal)

    pf = price_features(prices)
    df = pf.copy()
    if news_agg is not None and not news_agg.empty:
        df = df.merge(news_agg, on="date", how="left")
    if cal_wide is not None and not cal_wide.empty:
        df = df.merge(cal_wide, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(FEATURES_CSV, index=False)
    logger.info(f"Saved features -> {FEATURES_CSV} rows={len(df)} cols={df.shape[1]}")
    return df


# ---------------------------
# Training & model
# ---------------------------
def train_model(test_size_days: int = 60) -> None:
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError("features CSV not found — run featurize first")

    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["y_next_ret"])
    
    if df.empty or len(df) < 10:
        raise ValueError("Not enough data to train")

    drop_cols = {"date", "y_next_ret"}
    X_cols = [c for c in df.columns if c not in drop_cols]

    # Fill NA with zeros (simple approach)
    df[X_cols] = df[X_cols].fillna(0.0)

    df = df.sort_values("date").reset_index(drop=True)
    
    # Flexible test size calculation
    min_test_size = min(5, max(1, len(df) // 5))  # At least 1, at most 20% of data
    if test_size_days > len(df) // 2:
        test_size_days = min_test_size
        logger.warning(f"Test size too large for available data. Adjusting to {test_size_days} days")
    
    cutoff = df["date"].max() - pd.Timedelta(days=test_size_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    # If test set is empty, use last few rows as test
    if test.empty:
        test_size = min(5, len(df) // 4)  # Use 25% of data for test, max 5 days
        test = df.tail(test_size)
        train = df.iloc[:-test_size]
        logger.warning(f"Using last {test_size} rows as test set")

    if train.empty or test.empty:
        # Fallback: use 80/20 split if time-based split fails
        test_size = max(1, len(df) // 5)
        test = df.tail(test_size)
        train = df.iloc[:-test_size]
        logger.warning(f"Using simple split: train={len(train)}, test={len(test)}")

    if len(train) < 5:
        raise ValueError(f"Not enough training data: only {len(train)} rows. Need more historical data.")

    logger.info(f"Training data: {len(train)} rows ({train['date'].min()} to {train['date'].max()})")
    logger.info(f"Test data: {len(test)} rows ({test['date'].min()} to {test['date'].max()})")

    X_train, y_train = train[X_cols].values, train["y_next_ret"].values
    X_test, y_test = test[X_cols].values, test["y_next_ret"].values

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Adjust cross-validation splits based on data size
    n_splits = min(3, max(2, len(train) // 3))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    if _HAS_XGB:
        model = XGBRegressor(random_state=42, n_jobs=4)
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42, n_jobs=4)
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5],
        }

    # Reduce iterations for small datasets
    n_iter = min(10, len(train) // 2)
    
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=n_iter, 
        scoring="neg_mean_squared_error", 
        cv=tscv, 
        verbose=1, 
        n_jobs=4, 
        random_state=42
    )

    logger.info("Starting hyperparameter tuning")
    random_search.fit(X_train_s, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    logger.info(f"Best params: {best_params}")

    preds = best_model.predict(X_test_s)
    # FIXED: Use numpy sqrt for RMSE calculation
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    logger.info(f"Test RMSE={rmse:.6f} R^2={r2:.4f} test_days={len(test)}")

    # save artifacts
    dump(best_model, MODEL_PATH)
    dump(scaler, SCALER_PATH)
    with open(FEATCOLS_PATH, "w") as f:
        json.dump({"feature_columns": X_cols, "best_params": best_params}, f, indent=2)

    logger.info(f"Saved model -> {MODEL_PATH} scaler -> {SCALER_PATH} features -> {FEATCOLS_PATH}")
# ---------------------------
# Predict & Backtest
# ---------------------------
def predict_next() -> None:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATCOLS_PATH) and os.path.exists(FEATURES_CSV)):
        raise FileNotFoundError("Model/scaler/features not found. Run train first.")

    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    with open(FEATCOLS_PATH, "r") as f:
        meta = json.load(f)
    X_cols = meta.get("feature_columns", [])
    if not X_cols:
        raise KeyError("No feature_columns in features.json")

    df[X_cols] = df[X_cols].fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)
    last_row = df.iloc[[-1]]

    scaler = load(SCALER_PATH)
    model = load(MODEL_PATH)

    X = last_row[X_cols].values
    Xs = scaler.transform(X)
    yhat_ret = model.predict(Xs)[0]

    last_close = float(last_row["close"].values[0])
    pred_close = last_close * (1.0 + float(yhat_ret))

    logger.info("[predict]")
    logger.info(f"  last date  : {last_row['date'].dt.date.values[0]}")
    logger.info(f"  last close : {last_close:.5f}")
    logger.info(f"  yhat ret   : {yhat_ret:.6f}")
    logger.info(f"  pred close : {pred_close:.5f} (next day)")


def backtest(sign_threshold: float = 0.0) -> Dict[str, Any]:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATCOLS_PATH) and os.path.exists(FEATURES_CSV)):
        raise FileNotFoundError("Model/scaler/features not found. Run train first.")

    df = pd.read_csv(FEATURES_CSV)
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    with open(FEATCOLS_PATH, "r") as f:
        meta = json.load(f)
    X_cols = meta.get("feature_columns", [])
    if not X_cols:
        raise KeyError("No feature_columns in features.json")

    df[X_cols] = df[X_cols].fillna(0.0)
    df = df.sort_values("date").reset_index(drop=True)
    scaler = load(SCALER_PATH)
    model = load(MODEL_PATH)

    X = df[X_cols].values
    Xs = scaler.transform(X)
    df["pred_ret"] = model.predict(Xs)

    df["signal"] = np.where(df["pred_ret"] > sign_threshold, 1, np.where(df["pred_ret"] < -sign_threshold, -1, 0))
    df["strategy_ret"] = df["signal"].shift(1) * df["y_next_ret"]

    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    df["cum_buyhold"] = (1 + df["ret_1d"].fillna(0)).cumprod()

    df = df.dropna(subset=["strategy_ret"])
    if df.empty:
        raise ValueError("Backtest produced no valid strategy returns (empty after shift)")

    total_return = df["cum_strategy"].iloc[-1] - 1
    buyhold_return = df["cum_buyhold"].iloc[-1] - 1
    sharpe_ratio = (df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-9) * np.sqrt(252))
    max_drawdown = (df["cum_strategy"] / df["cum_strategy"].cummax() - 1).min()
    win_rate = float((df["strategy_ret"] > 0).mean())

    results = {
        "final_strategy": float(df["cum_strategy"].iloc[-1]),
        "final_buyhold": float(df["cum_buyhold"].iloc[-1]),
        "total_return": float(total_return),
        "buyhold_return": float(buyhold_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "avg_daily_return": float(df["strategy_ret"].mean()),
        "std_daily_return": float(df["strategy_ret"].std()),
    }

    logger.info("Backtest results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.6f}")

    return results


# ---------------------------
# CLI
# ---------------------------
def main() -> None:
    config = load_config()

    parser = argparse.ArgumentParser(description="EURUSD predictor pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p_prices = sub.add_parser("ingest_prices")
    p_prices.add_argument("--symbol", default=config.get("symbol", "EURUSD=X"))
    p_prices.add_argument("--interval", default=config.get("interval", "1d"))
    p_prices.add_argument("--start", default=None)
    p_prices.add_argument("--end", default=None)

    sub.add_parser("ingest_news")
    sub.add_parser("ingest_calendar")
    sub.add_parser("featurize")

    p_train = sub.add_parser("train")
    p_train.add_argument("--test_days", type=int, default=config.get("test_size_days", 60))

    sub.add_parser("predict")

    p_bt = sub.add_parser("backtest")
    p_bt.add_argument("--threshold", type=float, default=config.get("threshold", 0.0))

    p_cfg = sub.add_parser("config")
    p_cfg.add_argument("--set", nargs=2, action="append", metavar=("KEY", "VALUE"), help="Set configuration values")

    args = parser.parse_args()

    if args.cmd == "ingest_prices":
        ingest_prices(symbol=args.symbol, interval=args.interval, start=args.start, end=args.end)
    elif args.cmd == "ingest_news":
        ingest_news(config)
    elif args.cmd == "ingest_calendar":
        ingest_calendar(config)
    elif args.cmd == "featurize":
        build_dataset()
    elif args.cmd == "train":
        build_dataset()
        train_model(test_size_days=args.test_days)
    elif args.cmd == "predict":
        predict_next()
    elif args.cmd == "backtest":
        backtest(sign_threshold=args.threshold)
    elif args.cmd == "config":
        if args.set:
            for key, value in args.set:
                # naive type coercion
                try:
                    if value.lower() == "true":
                        val = True
                    elif value.lower() == "false":
                        val = False
                    elif value.isdigit():
                        val = int(value)
                    else:
                        try:
                            val = float(value)
                        except Exception:
                            val = value
                except Exception:
                    val = value

                keys = key.split(".")
                cur = config
                for k in keys[:-1]:
                    if k not in cur or not isinstance(cur[k], dict):
                        cur[k] = {}
                    cur = cur[k]
                cur[keys[-1]] = val
            save_config(config)
            print("Configuration updated")
        else:
            print(json.dumps(config, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Pipeline failed with error", exc_info=exc)
        sys.exit(1)