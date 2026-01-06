"""
Stock data fetching service - Hybrid yahooquery + yfinance approach.

Uses:
- yahooquery: Batch quotes (current prices with pre/post market) - async native
- yfinance: Chart history with prepost=True for extended hours - via threadpool

Both libraries use the same unofficial Yahoo Finance API, but have different
strengths. This hybrid approach gives us the best of both worlds.

yahooquery advantages:
- Batch quotes in a single call (fast for dashboard price updates)
- Native async support
- Pre/post market prices in the quote data

yfinance advantages:
- prepost=True parameter for historical data with extended hours candles
- Simpler API for individual ticker history

Note: yfinance is synchronous so we use run_in_executor for FastAPI compatibility.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import pytz
from yahooquery import Ticker as YQTicker  # For batch quotes
import yfinance as yf  # For history with prepost=True

from ..config import settings
from ..logging_config import get_logger
from .cache import StockCache, get_stock_cache
from .price_history import PriceHistoryService, get_price_history_service
from .calculations import StockCalculations
from .candle_aggregator import CandleAggregator, get_candle_aggregator

logger = get_logger(__name__)

# US Eastern timezone for market hours
MARKET_TZ = pytz.timezone('US/Eastern')


class StockFetcher:
    """
    Hybrid stock data fetching service.
    
    Uses yahooquery for:
    - Batch price quotes (fast, includes pre/post market prices)
    - Ticker validation
    
    Uses yfinance for:
    - Historical chart data with prepost=True (extended hours candles)
    
    Both are wrapped appropriately for FastAPI async compatibility.
    """
    
    def __init__(
        self,
        cache: Optional[StockCache] = None,
        price_history: Optional[PriceHistoryService] = None,
        candle_aggregator: Optional[CandleAggregator] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize the stock fetcher.
        
        Args:
            cache: Stock cache instance.
            price_history: Price history service instance.
            candle_aggregator: Candle aggregator for live chart updates.
            max_workers: Max thread pool workers.
        """
        self._cache = cache or get_stock_cache()
        self._price_history = price_history or get_price_history_service()
        self._candle_aggregator = candle_aggregator or get_candle_aggregator()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or settings.yfinance_max_workers
        )
    
    # ============ DataFrame Utilities ============
    
    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns if present."""
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = [
                col[0] if isinstance(col, tuple) else col 
                for col in df.columns
            ]
        return df
    
    def _dataframe_to_daily_history(self, df: pd.DataFrame, ticker: str = None) -> List[Dict[str, Any]]:
        """Convert DataFrame to daily history dictionaries."""
        history = []
        df = self._flatten_columns(df)
        calc = StockCalculations()
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle yahooquery's multi-ticker DataFrame format
        # Symbol can be in the index (multi-index) or as a column
        if ticker:
            ticker = ticker.upper()
            
            # Check if symbol is in the multi-index
            if hasattr(df.index, 'names') and 'symbol' in df.index.names:
                df = df.reset_index()
                df = df[df['symbol'] == ticker].copy()
                df = df.drop(columns=['symbol'], errors='ignore')
            elif 'symbol' in df.columns:
                df = df[df['symbol'] == ticker].copy()
                df = df.drop(columns=['symbol'], errors='ignore')
        elif 'symbol' in df.columns:
            df = df.drop(columns=['symbol'], errors='ignore')
        
        # Reset index if date is in index
        if df.index.name == 'date' or (hasattr(df.index, 'names') and 'date' in df.index.names):
            df = df.reset_index()
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Get date from row or index
            if 'date' in df.columns:
                date_val = row['date']
            else:
                date_val = df.index[idx]
            
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]
            
            # yahooquery uses lowercase column names
            open_val = row.get('open') or row.get('Open', 0)
            high_val = row.get('high') or row.get('High', 0)
            low_val = row.get('low') or row.get('Low', 0)
            close_val = row.get('close') or row.get('Close', 0)
            volume_val = row.get('volume') or row.get('Volume', 0)
            
            history.append({
                "date": date_str,
                "open": round(calc.safe_float(open_val), 2),
                "high": round(calc.safe_float(high_val), 2),
                "low": round(calc.safe_float(low_val), 2),
                "close": round(calc.safe_float(close_val), 2),
                "volume": calc.safe_int(volume_val)
            })
        
        return history
    
    def _dataframe_to_intraday_history(self, df: pd.DataFrame, ticker: str = None) -> List[Dict[str, Any]]:
        """Convert DataFrame to intraday history with full timestamps."""
        history = []
        df = self._flatten_columns(df)
        calc = StockCalculations()
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Handle yahooquery's multi-ticker DataFrame format
        # Symbol can be in the index (multi-index) or as a column
        if ticker:
            ticker = ticker.upper()
            
            # Check if symbol is in the multi-index
            if hasattr(df.index, 'names') and 'symbol' in df.index.names:
                df = df.reset_index()
                df = df[df['symbol'] == ticker].copy()
                df = df.drop(columns=['symbol'], errors='ignore')
            elif 'symbol' in df.columns:
                df = df[df['symbol'] == ticker].copy()
                df = df.drop(columns=['symbol'], errors='ignore')
        elif 'symbol' in df.columns:
            df = df.drop(columns=['symbol'], errors='ignore')
        
        # Reset index if needed
        if df.index.name == 'date' or (hasattr(df.index, 'names') and 'date' in df.index.names):
            df = df.reset_index()
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Get timestamp
            if 'date' in df.columns:
                ts = row['date']
            else:
                ts = df.index[idx]
            
            # Convert to Python datetime
            if hasattr(ts, 'to_pydatetime'):
                timestamp = ts.to_pydatetime()
            else:
                timestamp = pd.Timestamp(ts).to_pydatetime()
            
            # Convert to US/Eastern timezone for display
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone(MARKET_TZ).replace(tzinfo=None)
            
            # yahooquery uses lowercase column names
            open_val = row.get('open') or row.get('Open', 0)
            high_val = row.get('high') or row.get('High', 0)
            low_val = row.get('low') or row.get('Low', 0)
            close_val = row.get('close') or row.get('Close', 0)
            volume_val = row.get('volume') or row.get('Volume', 0)
            
            history.append({
                "date": timestamp.strftime('%Y-%m-%d %H:%M'),
                "timestamp": timestamp,
                "open": round(calc.safe_float(open_val), 2),
                "high": round(calc.safe_float(high_val), 2),
                "low": round(calc.safe_float(low_val), 2),
                "close": round(calc.safe_float(close_val), 2),
                "volume": calc.safe_int(volume_val)
            })
        
        return history
    
    # ============ Empty Response ============
    
    @staticmethod
    def _empty_response(ticker: str) -> Dict[str, Any]:
        """Return empty response for failed/invalid tickers."""
        return {
            "ticker": ticker.upper(),
            "current_price": 0,
            "previous_close": 0,
            "change": 0,
            "change_pct": 0,
            "ytd_return": 0,
            "sma_200": None,
            "price_vs_sma": None,
            "high_52w": None,
            "low_52w": None,
            "history": []
        }
    
    # ============ Ticker Validation ============
    
    def _validate_ticker_sync(self, ticker: str) -> Tuple[bool, float]:
        """Quick validation that a ticker exists."""
        ticker = ticker.upper()
        
        # Check cache first
        cached = self._cache.get(f"stock:{ticker}")
        if cached and cached.get('current_price', 0) > 0:
            return True, cached['current_price']
        
        try:
            t = YQTicker(ticker)
            price_data = t.price.get(ticker, {})
            
            if isinstance(price_data, str):
                # Error message returned
                return False, 0.0
            
            price = price_data.get('regularMarketPrice') or price_data.get('previousClose')
            
            if price and float(price) > 0:
                return True, float(price)
            return False, 0.0
        except Exception as e:
            logger.warning(f"Validation failed for {ticker}: {e}")
            return False, 0.0
    
    async def validate_ticker(self, ticker: str) -> Tuple[bool, float]:
        """Async wrapper for ticker validation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._validate_ticker_sync, 
            ticker
        )
    
    # ============ Stock Data Fetching ============
    
    def _get_stock_data_sync(self, ticker: str) -> Dict[str, Any]:
        """Synchronous function to fetch stock data from yahooquery."""
        ticker = ticker.upper()
        calc = StockCalculations()
        
        # Check cache first
        cached = self._cache.get(f"stock:{ticker}")
        if cached:
            return cached
        
        for attempt in range(settings.yfinance_retries):
            try:
                t = YQTicker(ticker)
                
                # Get price info - yahooquery returns dict keyed by ticker
                price_data = t.price.get(ticker, {})
                
                if isinstance(price_data, str):
                    # Error message - invalid ticker
                    logger.warning(f"Invalid ticker {ticker}: {price_data}")
                    return self._empty_response(ticker)
                
                regular_price = price_data.get('regularMarketPrice', 0)
                previous_close = price_data.get('regularMarketPreviousClose') or price_data.get('previousClose') or regular_price
                
                # Check for pre/post market prices
                current_price = regular_price
                market_state = price_data.get('marketState', '')
                
                if market_state in ('PRE', 'PREPRE'):
                    pre_market = price_data.get('preMarketPrice')
                    if pre_market and pre_market > 0:
                        current_price = pre_market
                elif market_state in ('POST', 'POSTPOST', 'CLOSED'):
                    post_market = price_data.get('postMarketPrice')
                    if post_market and post_market > 0:
                        current_price = post_market
                
                if not current_price or current_price == 0:
                    if attempt < settings.yfinance_retries - 1:
                        continue
                    return self._empty_response(ticker)
                
                current_price = float(current_price)
                previous_close = float(previous_close) if previous_close else current_price
                
                # Calculate change
                change = current_price - previous_close
                change_pct = (change / previous_close * 100) if previous_close else 0
                
                # Get historical data for SMA
                end_date = datetime.now()
                start_date = end_date - timedelta(days=250)
                
                # Try database first
                history_data = self._price_history.get_daily_history(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # Fetch from API if not enough data
                if len(history_data) < 50:
                    hist = t.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        adj_ohlc=True
                    )
                    
                    if isinstance(hist, pd.DataFrame) and not hist.empty:
                        history_data = self._dataframe_to_daily_history(hist, ticker)
                        self._price_history.store_daily_history(ticker, history_data)
                
                # Calculate SMA(200)
                sma_200 = None
                price_vs_sma = None
                
                if history_data:
                    closes = [h['close'] for h in history_data if h.get('close')]
                    sma_200 = calc.calculate_sma(closes, 200)
                    if sma_200:
                        price_vs_sma = calc.calculate_price_vs_sma(current_price, sma_200)
                
                # Calculate YTD return
                ytd_return = calc.calculate_ytd_return(current_price, history_data)
                
                # Get 52-week high/low from summary_detail
                summary = t.summary_detail.get(ticker, {})
                high_52w = summary.get('fiftyTwoWeekHigh')
                low_52w = summary.get('fiftyTwoWeekLow')
                
                if not high_52w and history_data:
                    highs = [h['high'] for h in history_data if h.get('high')]
                    high_52w = max(highs) if highs else None
                if not low_52w and history_data:
                    lows = [h['low'] for h in history_data if h.get('low')]
                    low_52w = min(lows) if lows else None
                
                # Return last 30 days for mini chart
                history = history_data[-30:] if history_data else []
                
                result = {
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "previous_close": round(previous_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "ytd_return": round(ytd_return, 2),
                    "sma_200": round(sma_200, 2) if sma_200 else None,
                    "price_vs_sma": round(price_vs_sma, 2) if price_vs_sma else None,
                    "high_52w": round(float(high_52w), 2) if high_52w else None,
                    "low_52w": round(float(low_52w), 2) if low_52w else None,
                    "history": history
                }
                
                self._cache.set(f"stock:{ticker}", result)
                return result
                
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{settings.yfinance_retries} "
                    f"failed for {ticker}: {e}"
                )
                if attempt < settings.yfinance_retries - 1:
                    continue
        
        return self._empty_response(ticker)
    
    def _get_multiple_stocks_sync(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch data for multiple stocks using yahooquery's batch capabilities."""
        results = {}
        calc = StockCalculations()
        tickers = [t.upper() for t in tickers]
        
        # Separate cached from uncached
        uncached_tickers = []
        for ticker in tickers:
            cached = self._cache.get(f"stock:{ticker}")
            if cached:
                results[ticker] = cached
            else:
                uncached_tickers.append(ticker)
        
        if not uncached_tickers:
            return results
        
        try:
            # Batch fetch all uncached tickers at once with yahooquery
            t = YQTicker(uncached_tickers)
            
            # Get all price data in one call (includes pre/post market)
            all_prices = t.price
            all_summaries = t.summary_detail
            
            # Get historical data for all tickers (daily, no extended hours needed)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=250)
            
            all_history = t.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adj_ohlc=True
            )
            
            for ticker in uncached_tickers:
                try:
                    price_data = all_prices.get(ticker, {})
                    
                    if isinstance(price_data, str):
                        results[ticker] = self._empty_response(ticker)
                        continue
                    
                    regular_price = price_data.get('regularMarketPrice', 0)
                    previous_close = price_data.get('regularMarketPreviousClose') or price_data.get('previousClose') or regular_price
                    
                    # Check for pre/post market prices
                    current_price = regular_price
                    market_state = price_data.get('marketState', '')
                    
                    if market_state in ('PRE', 'PREPRE'):
                        pre_market = price_data.get('preMarketPrice')
                        if pre_market and pre_market > 0:
                            current_price = pre_market
                    elif market_state in ('POST', 'POSTPOST', 'CLOSED'):
                        post_market = price_data.get('postMarketPrice')
                        if post_market and post_market > 0:
                            current_price = post_market
                    
                    if not current_price or current_price == 0:
                        results[ticker] = self._empty_response(ticker)
                        continue
                    
                    current_price = float(current_price)
                    previous_close = float(previous_close) if previous_close else current_price
                    
                    change = current_price - previous_close
                    change_pct = (change / previous_close * 100) if previous_close else 0
                    
                    # Extract history for this ticker
                    history_data = []
                    if isinstance(all_history, pd.DataFrame) and not all_history.empty:
                        history_data = self._dataframe_to_daily_history(all_history, ticker)
                        if history_data:
                            self._price_history.store_daily_history(ticker, history_data)
                    
                    # Calculate SMA(200)
                    sma_200 = None
                    price_vs_sma = None
                    
                    if history_data:
                        closes = [h['close'] for h in history_data if h.get('close')]
                        sma_200 = calc.calculate_sma(closes, 200)
                        if sma_200:
                            price_vs_sma = calc.calculate_price_vs_sma(current_price, sma_200)
                    
                    ytd_return = calc.calculate_ytd_return(current_price, history_data)
                    
                    # Get 52-week high/low
                    summary = all_summaries.get(ticker, {})
                    high_52w = summary.get('fiftyTwoWeekHigh') if isinstance(summary, dict) else None
                    low_52w = summary.get('fiftyTwoWeekLow') if isinstance(summary, dict) else None
                    
                    if not high_52w and history_data:
                        highs = [h['high'] for h in history_data if h.get('high')]
                        high_52w = max(highs) if highs else None
                    if not low_52w and history_data:
                        lows = [h['low'] for h in history_data if h.get('low')]
                        low_52w = min(lows) if lows else None
                    
                    history = history_data[-30:] if history_data else []
                    
                    result = {
                        "ticker": ticker,
                        "current_price": round(current_price, 2),
                        "previous_close": round(previous_close, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "ytd_return": round(ytd_return, 2),
                        "sma_200": round(sma_200, 2) if sma_200 else None,
                        "price_vs_sma": round(price_vs_sma, 2) if price_vs_sma else None,
                        "high_52w": round(float(high_52w), 2) if high_52w else None,
                        "low_52w": round(float(low_52w), 2) if low_52w else None,
                        "history": history
                    }
                    
                    self._cache.set(f"stock:{ticker}", result)
                    results[ticker] = result
                    
                except Exception as e:
                    logger.warning(f"Error processing {ticker}: {e}")
                    results[ticker] = self._empty_response(ticker)
        
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fallback to individual fetches
            for ticker in uncached_tickers:
                if ticker not in results:
                    results[ticker] = self._get_stock_data_sync(ticker)
        
        return results
    
    async def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Async wrapper for fetching stock data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._get_stock_data_sync, 
            ticker
        )
    
    async def get_multiple_stocks(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch data for multiple stocks using batch API."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_multiple_stocks_sync,
            tickers
        )
    
    # ============ Batch Prices (Lightweight) ============
    
    def _get_batch_prices_sync(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get current prices for multiple tickers in one batch call.
        
        This is a lightweight endpoint that only fetches price data,
        not historical data or other metadata. Ideal for frequent polling.
        
        Also updates the candle aggregator and returns chart_point for
        real-time 1m chart updates.
        
        No caching - always fetches fresh data from Yahoo.
        """
        import time
        results = {}
        tickers = [t.upper() for t in tickers]
        
        if not tickers:
            return results
        
        try:
            # Single batch call for all tickers with yahooquery
            # This includes pre/post market prices automatically
            start = time.time()
            t = YQTicker(tickers)
            all_prices = t.price
            elapsed = time.time() - start
            logger.info(f"Yahoo prices fetch for {tickers}: {elapsed:.2f}s")
            
            for ticker in tickers:
                price_data = all_prices.get(ticker, {})
                
                if isinstance(price_data, str):
                    # Error message - invalid ticker
                    results[ticker] = {
                        "ticker": ticker,
                        "current_price": 0,
                        "previous_close": 0,
                        "change": 0,
                        "change_pct": 0,
                        "market_state": "ERROR",
                        "chart_point": None
                    }
                    continue
                
                regular_price = price_data.get('regularMarketPrice', 0)
                previous_close = (
                    price_data.get('regularMarketPreviousClose') or 
                    price_data.get('previousClose') or 
                    regular_price
                )
                
                # Check for pre/post market prices
                current_price = regular_price
                market_state = price_data.get('marketState', 'REGULAR')
                
                if market_state in ('PRE', 'PREPRE'):
                    pre_market = price_data.get('preMarketPrice')
                    if pre_market and pre_market > 0:
                        current_price = pre_market
                elif market_state in ('POST', 'POSTPOST', 'CLOSED'):
                    post_market = price_data.get('postMarketPrice')
                    if post_market and post_market > 0:
                        current_price = post_market
                
                current_price = float(current_price) if current_price else 0
                previous_close = float(previous_close) if previous_close else current_price
                
                change = current_price - previous_close
                change_pct = (change / previous_close * 100) if previous_close else 0
                
                # Update candle aggregator and get chart point
                chart_point = None
                if current_price > 0:
                    # Get volume if available (regularMarketVolume is cumulative for the day)
                    volume = price_data.get('regularMarketVolume', 0) or 0
                    chart_point = self._candle_aggregator.update_price(
                        ticker=ticker,
                        price=current_price,
                        volume=0  # Don't add cumulative volume to candle
                    )
                
                results[ticker] = {
                    "ticker": ticker,
                    "current_price": round(current_price, 2),
                    "previous_close": round(previous_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "market_state": market_state,
                    "chart_point": chart_point
                }
                
        except Exception as e:
            logger.error(f"Batch prices fetch failed: {e}")
            # Return empty results for all tickers on error
            for ticker in tickers:
                if ticker not in results:
                    results[ticker] = {
                        "ticker": ticker,
                        "current_price": 0,
                        "previous_close": 0,
                        "change": 0,
                        "change_pct": 0,
                        "market_state": "ERROR",
                        "chart_point": None
                    }
        
        return results
    
    async def get_batch_prices(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Async wrapper for batch price fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_batch_prices_sync,
            tickers
        )
    
    # ============ Incremental Intraday History ============
    
    def _get_incremental_intraday_sync(
        self,
        tickers: List[str],
        interval: str,
        since_timestamps: Dict[str, Optional[datetime]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch only NEW intraday data since each ticker's last known timestamp.
        
        This is the key optimization for incremental chart updates:
        - Find the oldest "since" timestamp across all tickers
        - Batch fetch from that point for all tickers
        - Filter results per-ticker to only truly new points
        - Store new points in database
        
        Args:
            tickers: List of stock tickers to fetch.
            interval: Time interval ('1m', '5m', '15m').
            since_timestamps: Dict of ticker -> last known timestamp (or None).
            
        Returns:
            Dict of ticker -> {new_points: [...], latest_timestamp: datetime}
        """
        tickers = [t.upper() for t in tickers]
        results = {ticker: {"new_points": [], "latest_timestamp": None} for ticker in tickers}
        
        if not tickers:
            return results
        
        # Find the oldest timestamp - we'll fetch from this point for all
        # Tickers with no data (None) need full recent data
        valid_timestamps = [ts for ts in since_timestamps.values() if ts is not None]
        
        if not valid_timestamps:
            # No existing data for any ticker - fetch recent period
            fetch_start = datetime.now() - timedelta(days=2)
        else:
            # Fetch from oldest known point (with small buffer for safety)
            oldest = min(valid_timestamps)
            fetch_start = oldest - timedelta(minutes=5)
        
        try:
            # Calculate period string
            days_back = (datetime.now() - fetch_start).days + 1
            days_back = max(1, min(days_back, 7))  # Clamp to 1-7 days
            
            logger.info(
                f"Incremental fetch for {len(tickers)} tickers, "
                f"interval={interval}, days={days_back}, prepost=True"
            )
            
            # Use yfinance for history with prepost=True (extended hours)
            # yfinance.download supports multiple tickers but NOT prepost
            # So we fetch individually (can be parallelized)
            for ticker in tickers:
                try:
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(
                        period=f"{days_back}d",
                        interval=interval,
                        prepost=True  # KEY: Include pre-market and after-hours
                    )
                    
                    if hist.empty:
                        continue
                    
                    ticker_history = self._dataframe_to_intraday_history(hist, ticker)
                    
                    if not ticker_history:
                        continue
                    
                    # Filter to only points newer than our last known timestamp
                    since_ts = since_timestamps.get(ticker)
                    if since_ts:
                        new_points = [
                            h for h in ticker_history 
                            if h['timestamp'] > since_ts
                        ]
                    else:
                        # No existing data - all points are new
                        new_points = ticker_history
                    
                    if new_points:
                        # Store new points in database
                        self._price_history.store_intraday_history(
                            ticker, interval, new_points
                        )
                        
                        # Get latest timestamp
                        latest_ts = max(p['timestamp'] for p in new_points)
                        
                        # Remove timestamp from response (not JSON serializable)
                        response_points = []
                        for p in new_points:
                            point = {k: v for k, v in p.items() if k != 'timestamp'}
                            response_points.append(point)
                        
                        results[ticker] = {
                            "new_points": response_points,
                            "latest_timestamp": latest_ts.isoformat(),
                            "count": len(response_points)
                        }
                        
                        logger.debug(
                            f"Incremental: {ticker} got {len(new_points)} new points"
                        )
                    else:
                        # No new points - return empty result with current timestamp
                        if since_ts:
                            results[ticker] = {
                                "new_points": [],
                                "latest_timestamp": since_ts.isoformat(),
                                "count": 0
                            }
                            logger.debug(f"Incremental: {ticker} no new data since {since_ts}")
                        
                except Exception as e:
                    logger.warning(f"Error processing incremental data for {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"Incremental intraday fetch failed: {e}")
        
        return results
    
    async def get_incremental_intraday(
        self,
        tickers: List[str],
        interval: str,
        since_timestamps: Optional[Dict[str, Optional[str]]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Async wrapper for incremental intraday fetching.
        
        If since_timestamps is not provided, will look up from database.
        """
        tickers = [t.upper() for t in tickers]
        
        # Convert string timestamps to datetime
        if since_timestamps:
            parsed_timestamps = {}
            for ticker, ts in since_timestamps.items():
                if ts:
                    try:
                        parsed_timestamps[ticker.upper()] = datetime.fromisoformat(ts)
                    except (ValueError, TypeError):
                        parsed_timestamps[ticker.upper()] = None
                else:
                    parsed_timestamps[ticker.upper()] = None
        else:
            # Look up from database
            parsed_timestamps = self._price_history.get_latest_timestamps_batch(
                tickers, interval
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_incremental_intraday_sync,
            tickers,
            interval,
            parsed_timestamps
        )
    
    # ============ Reference Close ============
    
    def _get_reference_close(
        self, 
        ticker: str, 
        on_or_before_date: datetime
    ) -> Optional[float]:
        """Get closing price from last trading day on or before given date."""
        ticker = ticker.upper()
        
        # Look back up to 10 days
        start_date = on_or_before_date - timedelta(days=10)
        end_date = on_or_before_date + timedelta(days=1)
        
        # Try stored data first
        stored = self._price_history.get_daily_history(
            ticker,
            start_date.strftime('%Y-%m-%d'),
            on_or_before_date.strftime('%Y-%m-%d')
        )
        
        if stored:
            return stored[-1]['close']
        
        # Fetch from API if not in database (daily data, yahooquery is fine)
        try:
            t = YQTicker(ticker)
            hist = t.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adj_ohlc=True
            )
            
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                history = self._dataframe_to_daily_history(hist, ticker)
                target_str = on_or_before_date.strftime('%Y-%m-%d')
                history = [h for h in history if h['date'] <= target_str]
                self._price_history.store_daily_history(ticker, history)
                return history[-1]['close'] if history else None
        except Exception as e:
            logger.warning(f"Error fetching reference close for {ticker}: {e}")
        
        return None
    
    # ============ Stock History ============
    
    def _get_stock_history_sync(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Get historical data for a stock."""
        ticker = ticker.upper()
        
        # Intraday configuration (interval, fetch_days)
        # User requirements mapped to closest yfinance-supported intervals:
        # - 1d: every 5 min (yfinance doesn't support 3m, 5m is closest)
        # - 3d: every 15 min (yfinance doesn't support 10m, 15m is closest)  
        # - 1w: every 30 min
        # - 1mo: every 60 min (hourly)
        intraday_config = {
            "1d": ("5m", 2),      # ~192 points per day (16 hrs * 12)
            "3d": ("15m", 5),     # ~192 points (3 days * 64)
            "1w": ("30m", 10),    # ~160 points (5 days * 32)
            "1mo": ("60m", 35),   # ~336 points (21 days * 16)
        }
        
        # Daily/longer periods configuration
        # - 3mo: daily data (yfinance doesn't support 4h)
        # - 6mo: daily
        # - 1y: daily
        # - ytd: daily
        # - 2y: daily (sampled every 2 days)
        # - 5y: weekly
        daily_periods = {
            "3mo": 90, 
            "6mo": 180, 
            "1y": 365, 
            "2y": 730,
            "5y": 1825,  # 5 years
        }
        
        end_time = datetime.now()
        reference_close = None
        
        # Handle YTD
        if period == "ytd":
            year_start = datetime(end_time.year, 1, 1)
            days_since_year_start = (end_time - year_start).days
            daily_periods["ytd"] = days_since_year_start
        
        # Handle intraday periods
        if period in intraday_config:
            return self._get_intraday_history(ticker, period, intraday_config)
        
        # Daily data for longer periods
        days = daily_periods.get(period, 365)
        start_date = end_time - timedelta(days=days)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_time.strftime('%Y-%m-%d')
        
        # Get stored history
        stored_history = self._price_history.get_daily_history(ticker, start_str, end_str)
        coverage = self._price_history.analyze_coverage(stored_history, start_str, end_str)
        
        # If complete coverage, return
        if (coverage["has_data"] and 
            not coverage["missing_start_range"] and 
            not coverage["missing_end_range"]):
            logger.debug(f"Using {len(stored_history)} stored daily records for {ticker}")
            reference_close = self._calculate_daily_reference(ticker, stored_history)
            return {
                "history": stored_history,
                "reference_close": reference_close,
                "is_complete": True,
                "expected_start": start_str,
                "actual_start": coverage["actual_start"]
            }
        
        # Fetch missing data
        all_history = {h['date']: h for h in stored_history}
        fetch_error = False
        
        try:
            t = YQTicker(ticker)
            
            # Fetch missing start range
            if coverage["missing_start_range"]:
                gap_start, gap_end = coverage["missing_start_range"]
                logger.info(f"Fetching historical gap for {ticker} from {gap_start} to {gap_end}")
                
                hist = t.history(
                    start=gap_start,
                    end=gap_end,
                    adj_ohlc=True
                )
                
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    new_history = self._dataframe_to_daily_history(hist, ticker)
                    self._price_history.store_daily_history(ticker, new_history)
                    for h in new_history:
                        all_history[h['date']] = h
            
            # Fetch missing end range
            if coverage["missing_end_range"]:
                gap_start, gap_end = coverage["missing_end_range"]
                logger.info(f"Fetching recent data for {ticker} from {gap_start} to {gap_end}")
                
                hist = t.history(
                    start=gap_start,
                    end=gap_end,
                    adj_ohlc=True
                )
                
                if isinstance(hist, pd.DataFrame) and not hist.empty:
                    new_history = self._dataframe_to_daily_history(hist, ticker)
                    self._price_history.store_daily_history(ticker, new_history)
                    for h in new_history:
                        all_history[h['date']] = h
                        
        except Exception as e:
            logger.error(f"Error fetching daily history for {ticker}: {e}")
            fetch_error = True
        
        # Sort and filter
        result = sorted(all_history.values(), key=lambda x: x['date'])
        result = [h for h in result if start_str <= h['date'] <= end_str]
        
        # Sample data based on period to reduce data points
        # - 2y: every 2 days (~365 points)
        # - 5y: every 5 days (~365 points) - weekly would be too sparse
        if period == "2y" and len(result) > 400:
            result = result[::2]  # Every 2nd day
        elif period == "5y" and len(result) > 400:
            result = result[::5]  # Every 5th day (~365 points for 5 years)
        
        reference_close = self._calculate_daily_reference(ticker, result)
        result_start = result[0]['date'] if result else None
        
        return {
            "history": result,
            "reference_close": reference_close,
            "is_complete": not fetch_error,
            "expected_start": start_str,
            "actual_start": result_start
        }
    
    def _get_intraday_history(
        self, 
        ticker: str, 
        period: str, 
        config: Dict[str, Tuple[str, int]]
    ) -> Dict[str, Any]:
        """
        Get intraday historical data with smart gap-filling.
        
        This method:
        1. Checks what data we already have in SQLite
        2. Analyzes coverage to find missing time ranges
        3. Only fetches missing data from yfinance API
        4. Merges and returns complete dataset
        
        This dramatically reduces API calls and bandwidth when data
        is already partially available in the database.
        """
        interval, fetch_days = config[period]
        
        # Use Eastern time for all date comparisons (market timezone)
        # This ensures proper "today" detection regardless of server timezone
        end_time_eastern = datetime.now(MARKET_TZ).replace(tzinfo=None)
        end_time = datetime.now()  # Keep UTC for API calls
        
        # Calculate the time range we need
        # Add buffer for timezone differences and market hours
        db_start_time = end_time - timedelta(days=fetch_days + 2)
        
        # Analyze what coverage we have
        coverage = self._price_history.analyze_intraday_coverage(
            ticker, interval, db_start_time, end_time
        )
        
        stored = coverage.get('stored_data', [])
        
        logger.info(
            f"Intraday coverage for {ticker} ({interval}): "
            f"stored={coverage['stored_count']}, "
            f"has_data={coverage['has_data']}, "
            f"needs_fetch={coverage['needs_fetch']}, "
            f"gaps={len(coverage.get('missing_ranges', []))}"
        )
        
        # Check if we need to fetch any data
        if coverage['needs_fetch']:
            try:
                # Determine fetch strategy based on gaps
                missing_ranges = coverage.get('missing_ranges', [])
                actual_end = coverage.get('actual_end')
                
                # Determine if we're missing today's data entirely
                # This is the common case when opening the app after it was closed
                # Compare using Eastern time since stored data is in Eastern
                missing_today = (
                    actual_end is not None and 
                    actual_end.date() < end_time_eastern.date()
                )
                
                if not coverage['has_data'] or len(missing_ranges) > 2 or missing_today:
                    # No data, too fragmented, or missing today's data - fetch full period
                    # Using full period fetch is more reliable for intraday data
                    logger.info(
                        f"Fetching full intraday ({interval}) data for {ticker}, "
                        f"{fetch_days} days with prepost=True"
                        f"{' (missing today)' if missing_today else ''}"
                    )
                    
                    yf_ticker = yf.Ticker(ticker)
                    hist = yf_ticker.history(
                        period=f"{fetch_days}d",
                        interval=interval,
                        prepost=True
                    )
                    
                    if not hist.empty:
                        new_history = self._dataframe_to_intraday_history(hist, ticker)
                        if new_history:
                            self._price_history.store_intraday_history(ticker, interval, new_history)
                            
                            # Merge with existing
                            all_history = {h['timestamp']: h for h in stored}
                            for h in new_history:
                                all_history[h['timestamp']] = h
                            stored = list(all_history.values())
                            logger.info(f"Fetched {len(new_history)} points for {ticker}")
                
                else:
                    # Fetch only missing ranges (smart gap-fill)
                    for gap_start, gap_end in missing_ranges:
                        gap_duration = gap_end - gap_start
                        
                        # Skip tiny gaps (< 5 minutes)
                        if gap_duration.total_seconds() < 300:
                            continue
                        
                        # Calculate days needed for this gap
                        gap_days = max(1, gap_duration.days + 1)
                        
                        logger.info(
                            f"Filling gap for {ticker}: {gap_start} to {gap_end} "
                            f"({gap_days} days)"
                        )
                        
                        try:
                            yf_ticker = yf.Ticker(ticker)
                            
                            # For gaps, use start/end dates instead of period
                            # Add buffer for timezone handling
                            start_str = (gap_start - timedelta(hours=1)).strftime('%Y-%m-%d')
                            end_str = (gap_end + timedelta(hours=1)).strftime('%Y-%m-%d')
                            
                            hist = yf_ticker.history(
                                start=start_str,
                                end=end_str,
                                interval=interval,
                                prepost=True
                            )
                            
                            if not hist.empty:
                                new_history = self._dataframe_to_intraday_history(hist, ticker)
                                
                                # Filter to only the gap range
                                gap_history = [
                                    h for h in new_history
                                    if gap_start <= h['timestamp'] <= gap_end
                                ]
                                
                                if gap_history:
                                    self._price_history.store_intraday_history(
                                        ticker, interval, gap_history
                                    )
                                    
                                    # Merge with existing
                                    all_history = {h['timestamp']: h for h in stored}
                                    for h in gap_history:
                                        all_history[h['timestamp']] = h
                                    stored = list(all_history.values())
                                    
                                    logger.info(
                                        f"Filled gap with {len(gap_history)} points for {ticker}"
                                    )
                        except Exception as e:
                            logger.warning(f"Failed to fill gap for {ticker}: {e}")
                            
            except Exception as e:
                logger.error(f"Error fetching intraday data for {ticker}: {e}")
        
        if stored:
            stored = sorted(stored, key=lambda x: x['timestamp'])
            result = self._price_history.filter_intraday_by_period(stored, period)
            
            if result:
                # Calculate reference close
                result_dates = sorted(set(
                    h['timestamp'].strftime('%Y-%m-%d') 
                    for h in result if h.get('timestamp')
                ))
                
                reference_close = None
                if result_dates:
                    first_chart_date = datetime.strptime(result_dates[0], '%Y-%m-%d')
                    reference_close = self._get_reference_close(
                        ticker, 
                        first_chart_date - timedelta(days=1)
                    )
                
                # Remove timestamp from response
                for h in result:
                    if 'timestamp' in h:
                        del h['timestamp']
                
                # Calculate expected start
                trading_days_needed = {"1d": 1, "3d": 3, "1w": 5, "1mo": 21}.get(period, 1)
                expected_start = end_time - timedelta(days=trading_days_needed + 2)
                
                return {
                    "history": result,
                    "reference_close": reference_close,
                    "is_complete": True,
                    "expected_start": expected_start.strftime('%Y-%m-%d'),
                    "actual_start": result_dates[0] if result_dates else None
                }
        
        return {
            "history": [],
            "reference_close": None,
            "is_complete": False,
            "expected_start": None,
            "actual_start": None
        }
    
    def _check_intraday_coverage(
        self, 
        stored: List[Dict[str, Any]], 
        end_time: datetime,
        interval: str = "1m"
    ) -> Tuple[bool, bool]:
        """Check if intraday data is recent and has good coverage."""
        have_recent = False
        have_good_coverage = False
        
        if not stored:
            return have_recent, have_good_coverage
        
        from collections import defaultdict
        
        # Parse interval to get staleness threshold in minutes
        try:
            staleness_threshold = int(interval.replace('m', '').replace('h', ''))
            if 'h' in interval:
                staleness_threshold *= 60
        except ValueError:
            staleness_threshold = 1
        
        # Group data by date
        by_date = defaultdict(list)
        for h in stored:
            date_key = h['timestamp'].strftime('%Y-%m-%d')
            by_date[date_key].append(h['timestamp'])
        
        if not by_date:
            return have_recent, have_good_coverage
        
        most_recent_date = max(by_date.keys())
        
        # Check if we're during potential trading hours
        current_hour = end_time.hour
        current_weekday = end_time.weekday()
        is_potential_trading_time = (
            current_weekday < 5 and
            4 <= current_hour < 20
        )
        
        # Get the most recent timestamp
        latest_timestamp = max(h['timestamp'] for h in stored)
        minutes_since_latest = (end_time - latest_timestamp).total_seconds() / 60
        
        if is_potential_trading_time:
            have_recent = minutes_since_latest <= staleness_threshold
            if not have_recent:
                logger.debug(
                    f"Intraday data is stale: latest is {latest_timestamp}, "
                    f"{minutes_since_latest:.1f} minutes ago (threshold: {staleness_threshold}m)"
                )
        else:
            days_since_latest = (end_time - latest_timestamp).days
            have_recent = days_since_latest < 4
        
        # Check time span coverage
        times = sorted(by_date[most_recent_date])
        if len(times) >= 2:
            time_span_hours = (times[-1] - times[0]).total_seconds() / 3600
            have_good_coverage = time_span_hours >= settings.intraday_min_coverage_hours
            logger.debug(
                f"Stored data spans {time_span_hours:.1f} hours on "
                f"{most_recent_date}, {len(times)} points"
            )
        
        return have_recent, have_good_coverage
    
    def _calculate_daily_reference(
        self, 
        ticker: str, 
        history: List[Dict[str, Any]]
    ) -> Optional[float]:
        """Calculate reference close from first day of chart data."""
        if history:
            sorted_history = sorted(history, key=lambda x: x['date'])
            first_date = sorted_history[0]['date']
            first_date_dt = datetime.strptime(first_date, '%Y-%m-%d')
            return self._get_reference_close(ticker, first_date_dt - timedelta(days=1))
        return None
    
    async def get_stock_history(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """Async wrapper for fetching stock history."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_stock_history_sync,
            ticker,
            period
        )
    
    # ============ Cleanup ============
    
    def cleanup_old_data(self) -> Tuple[int, int]:
        """Cleanup old price history data."""
        return self._price_history.cleanup_old_data()
    
    def clear_ticker_cache(self, ticker: str) -> int:
        """Clear intraday cache for a ticker."""
        return self._price_history.clear_intraday_cache(ticker)


# Singleton instance
_stock_fetcher: Optional[StockFetcher] = None


def get_stock_fetcher() -> StockFetcher:
    """Get the singleton stock fetcher instance."""
    global _stock_fetcher
    if _stock_fetcher is None:
        _stock_fetcher = StockFetcher()
    return _stock_fetcher
