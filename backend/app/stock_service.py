import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pandas as pd
import pytz
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session
from .models import PriceHistory, IntradayPriceHistory

# US Eastern timezone for market hours
MARKET_TZ = pytz.timezone('US/Eastern')

# Database connection for synchronous operations
DATABASE_URL = "sqlite:///./data/portfolio.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Thread pool for running yfinance calls (which are synchronous)
executor = ThreadPoolExecutor(max_workers=10)

# Lock to serialize yfinance calls - yfinance is NOT thread-safe and concurrent
# downloads will cause data to cross between tickers
_yfinance_lock = threading.Lock()

# Simple in-memory cache for live data
_cache: dict[str, tuple[dict, float]] = {}
CACHE_TTL = 60  # 60 seconds cache for live price data

# Data retention period
DATA_RETENTION_DAYS = 730  # 2 years


def _get_from_cache(key: str) -> Optional[dict]:
    """Get data from cache if not expired."""
    if key in _cache:
        data, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return data
    return None


def _set_cache(key: str, data: dict):
    """Store data in cache."""
    _cache[key] = (data, time.time())


def _safe_float(value) -> float:
    """Safely convert a value to float, handling Series and other types."""
    if pd.isna(value):
        return 0.0
    if hasattr(value, 'iloc'):
        return float(value.iloc[0]) if len(value) > 0 else 0.0
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def _safe_int(value) -> int:
    """Safely convert a value to int, handling Series and other types."""
    if pd.isna(value):
        return 0
    if hasattr(value, 'iloc'):
        return int(value.iloc[0]) if len(value) > 0 else 0
    if hasattr(value, 'item'):
        return int(value.item())
    return int(value)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present (common with yf.download for single ticker)."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


# ============ Data Cleanup ============

def cleanup_old_data():
    """Delete price history data older than 2 years to save storage."""
    cutoff_date = datetime.now() - timedelta(days=DATA_RETENTION_DAYS)
    cutoff_str = cutoff_date.strftime('%Y-%m-%d')
    
    with Session(engine) as session:
        # Delete old daily data
        daily_deleted = session.execute(
            delete(PriceHistory).where(PriceHistory.date < cutoff_str)
        )
        
        # Delete old intraday data
        intraday_deleted = session.execute(
            delete(IntradayPriceHistory).where(IntradayPriceHistory.timestamp < cutoff_date)
        )
        
        session.commit()
        
        if daily_deleted.rowcount > 0 or intraday_deleted.rowcount > 0:
            print(f"Cleaned up {daily_deleted.rowcount} daily and {intraday_deleted.rowcount} intraday records older than 2 years")


def clear_intraday_cache(ticker: str) -> int:
    """Clear all intraday cache data for a specific ticker. Returns number of records deleted."""
    ticker = ticker.upper()
    
    with Session(engine) as session:
        result = session.execute(
            delete(IntradayPriceHistory).where(IntradayPriceHistory.ticker == ticker)
        )
        session.commit()
        deleted_count = result.rowcount
        print(f"Cleared {deleted_count} intraday records for {ticker}")
        return deleted_count


# ============ Daily Price History Functions ============

def _get_stored_history(ticker: str, start_date: str, end_date: str) -> list[dict]:
    """Get daily price history from database for a date range."""
    ticker = ticker.upper()
    with Session(engine) as session:
        result = session.execute(
            select(PriceHistory)
            .where(PriceHistory.ticker == ticker)
            .where(PriceHistory.date >= start_date)
            .where(PriceHistory.date <= end_date)
            .order_by(PriceHistory.date)
        )
        rows = result.scalars().all()
        return [
            {
                "date": row.date,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume
            }
            for row in rows
        ]


def _store_history(ticker: str, history: list[dict]):
    """Store daily price history in database. Skips duplicates."""
    if not history:
        return
    
    ticker = ticker.upper()
    
    # DEBUG: Print what we're about to store
    if history:
        print(f"DEBUG _store_history: ticker={ticker}, records={len(history)}, last_close={history[-1].get('close')}")
    
    with Session(engine) as session:
        # Get existing dates to avoid duplicates
        existing = session.execute(
            select(PriceHistory.date).where(PriceHistory.ticker == ticker)
        )
        existing_dates = {row[0] for row in existing}
        
        # Only insert new records
        new_records = []
        for h in history:
            if h['date'] not in existing_dates:
                new_records.append(PriceHistory(
                    ticker=ticker,
                    date=h['date'],
                    open=h.get('open'),
                    high=h.get('high'),
                    low=h.get('low'),
                    close=h.get('close'),
                    volume=h.get('volume')
                ))
        
        if new_records:
            session.add_all(new_records)
            session.commit()
            print(f"Stored {len(new_records)} new daily records for {ticker}, last_close={new_records[-1].close}")


# ============ Intraday Price History Functions ============

def _get_stored_intraday(ticker: str, interval: str, start_time: datetime, end_time: datetime) -> list[dict]:
    """Get intraday price history from database."""
    ticker = ticker.upper()
    with Session(engine) as session:
        result = session.execute(
            select(IntradayPriceHistory)
            .where(IntradayPriceHistory.ticker == ticker)
            .where(IntradayPriceHistory.interval == interval)
            .where(IntradayPriceHistory.timestamp >= start_time)
            .where(IntradayPriceHistory.timestamp <= end_time)
            .order_by(IntradayPriceHistory.timestamp)
        )
        rows = result.scalars().all()
        return [
            {
                "date": row.timestamp.strftime('%Y-%m-%d %H:%M'),
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume
            }
            for row in rows
        ]


def _store_intraday(ticker: str, interval: str, history: list[dict]):
    """Store intraday price history in database. Skips duplicates."""
    if not history:
        return
    
    ticker = ticker.upper()
    with Session(engine) as session:
        # Get existing timestamps to avoid duplicates
        existing = session.execute(
            select(IntradayPriceHistory.timestamp)
            .where(IntradayPriceHistory.ticker == ticker)
            .where(IntradayPriceHistory.interval == interval)
        )
        existing_timestamps = {row[0] for row in existing}
        
        # Only insert new records
        new_records = []
        for h in history:
            ts = h.get('timestamp')
            if ts and ts not in existing_timestamps:
                new_records.append(IntradayPriceHistory(
                    ticker=ticker,
                    timestamp=ts,
                    interval=interval,
                    open=h.get('open'),
                    high=h.get('high'),
                    low=h.get('low'),
                    close=h.get('close'),
                    volume=h.get('volume')
                ))
        
        if new_records:
            session.add_all(new_records)
            session.commit()
            print(f"Stored {len(new_records)} new intraday ({interval}) records for {ticker}")


def _filter_intraday_by_period(history: list[dict], period: str) -> list[dict]:
    """Filter intraday history to match the requested period (most recent trading days)."""
    if not history:
        return []
    
    # Map period to number of trading days we want
    trading_days_map = {"1d": 1, "3d": 3, "1w": 5}
    target_days = trading_days_map.get(period, 1)
    
    # Group by date and get the most recent N trading days
    from collections import defaultdict
    by_date = defaultdict(list)
    
    for h in history:
        ts = h.get('timestamp')
        if ts:
            date_key = ts.strftime('%Y-%m-%d')
            by_date[date_key].append(h)
    
    # Sort dates descending and take the most recent N
    sorted_dates = sorted(by_date.keys(), reverse=True)[:target_days]
    
    # Collect all data from those dates, sorted by timestamp
    result = []
    for date in sorted(sorted_dates):  # Sort ascending for output
        result.extend(sorted(by_date[date], key=lambda x: x['timestamp']))
    
    return result


def _dataframe_to_intraday_history(df: pd.DataFrame) -> list[dict]:
    """Convert a pandas DataFrame to intraday history dictionaries with full timestamps."""
    history = []
    df = _flatten_columns(df)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        ts = df.index[idx]
        
        # Convert to Python datetime
        if hasattr(ts, 'to_pydatetime'):
            timestamp = ts.to_pydatetime()
        else:
            timestamp = pd.Timestamp(ts).to_pydatetime()
        
        # Convert to US/Eastern timezone (market timezone) for display
        if timestamp.tzinfo is not None:
            # Convert to Eastern time, then remove timezone info for storage
            timestamp = timestamp.astimezone(MARKET_TZ).replace(tzinfo=None)
        
        history.append({
            "date": timestamp.strftime('%Y-%m-%d %H:%M'),
            "timestamp": timestamp,
            "open": round(_safe_float(row['Open']), 2),
            "high": round(_safe_float(row['High']), 2),
            "low": round(_safe_float(row['Low']), 2),
            "close": round(_safe_float(row['Close']), 2),
            "volume": _safe_int(row['Volume'])
        })
    
    return history


# ============ Ticker Validation ============

def _validate_ticker_sync(ticker: str) -> tuple[bool, float]:
    """Quick validation that a ticker exists. Returns (is_valid, current_price)."""
    ticker = ticker.upper()
    
    # Check cache first
    cached = _get_from_cache(f"stock:{ticker}")
    if cached and cached.get('current_price', 0) > 0:
        return True, cached['current_price']
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.fast_info
        price = getattr(info, 'last_price', None) or getattr(info, 'previous_close', None)
        
        if price and float(price) > 0:
            return True, float(price)
        return False, 0.0
    except Exception as e:
        print(f"Validation failed for {ticker}: {e}")
        return False, 0.0


async def validate_ticker(ticker: str) -> tuple[bool, float]:
    """Async wrapper for quick ticker validation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _validate_ticker_sync, ticker)


# ============ Stock Data Functions ============

def _get_stock_data_sync(ticker: str, retries: int = 2) -> dict:
    """Synchronous function to fetch stock data from yfinance."""
    ticker = ticker.upper()
    
    # Check cache first
    cached = _get_from_cache(f"stock:{ticker}")
    if cached:
        return cached
    
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            
            # Get price info - try to get after-hours price if available
            fast_info = stock.fast_info
            regular_price = getattr(fast_info, 'last_price', None) or getattr(fast_info, 'previous_close', 0)
            previous_close = getattr(fast_info, 'previous_close', regular_price) or regular_price
            
            # Try to get after-hours price from info dict (slower but more accurate)
            # Only do this if we already have a valid regular price
            current_price = regular_price
            if regular_price and regular_price > 0:
                try:
                    info = stock.info
                    # Prefer post-market price if available, otherwise pre-market, otherwise regular
                    post_market = info.get('postMarketPrice')
                    pre_market = info.get('preMarketPrice')
                    if post_market and post_market > 0:
                        current_price = post_market
                    elif pre_market and pre_market > 0:
                        current_price = pre_market
                except Exception:
                    pass  # Fall back to regular price
            
            if not current_price or current_price == 0:
                if attempt < retries - 1:
                    time.sleep(1)
                continue
            
            current_price = float(current_price)
            previous_close = float(previous_close)
            
            # Calculate change
            change = current_price - previous_close
            change_pct = (change / previous_close * 100) if previous_close else 0
            
            # Get historical data for SMA (use cached/stored if possible)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=250)
            
            # Try to get from database first
            history_data = _get_stored_history(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # If we don't have enough data, fetch from API
            if len(history_data) < 50:
                with _yfinance_lock:
                    hist = yf.download(
                        ticker, 
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        progress=False,
                        auto_adjust=True,
                        timeout=10
                    )
                hist = _flatten_columns(hist)
                
                if not hist.empty:
                    history_data = _dataframe_to_history(hist)
                    # Store for future use
                    _store_history(ticker, history_data)
            
            # Calculate SMA(200)
            sma_200 = None
            price_vs_sma = None
            
            if history_data:
                closes = [h['close'] for h in history_data if h.get('close')]
                if len(closes) >= 200:
                    sma_200 = sum(closes[-200:]) / 200
                elif len(closes) >= 50:
                    sma_200 = sum(closes) / len(closes)
                
                if sma_200:
                    price_vs_sma = ((current_price - sma_200) / sma_200) * 100
            
            # Calculate YTD return - compare current price to last close of previous year
            ytd_return = 0.0
            year_start = datetime(datetime.now().year, 1, 1)
            year_start_str = year_start.strftime('%Y-%m-%d')
            
            if history_data:
                # PREFERRED: Use last trading day of previous year (true YTD reference)
                prev_year_prices = [h for h in history_data if h['date'] < year_start_str]
                if prev_year_prices:
                    last_price_prev_year = prev_year_prices[-1]['close']
                    if last_price_prev_year and last_price_prev_year > 0:
                        ytd_return = ((current_price - last_price_prev_year) / last_price_prev_year) * 100
                
                # FALLBACK: If no previous year data, use first trading day of current year
                # (but not if it's today - that would give 0%)
                if ytd_return == 0.0:
                    ytd_prices = [h for h in history_data if h['date'] >= year_start_str]
                    if len(ytd_prices) > 1:  # Need at least 2 days of data
                        first_price = ytd_prices[0]['close']
                        if first_price and first_price > 0 and first_price != current_price:
                            ytd_return = ((current_price - first_price) / first_price) * 100
            
            # Get 52-week high/low
            high_52w = getattr(fast_info, 'year_high', None)
            low_52w = getattr(fast_info, 'year_low', None)
            
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
            
            _set_cache(f"stock:{ticker}", result)
            return result
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{retries} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
            continue
    
    return _empty_response(ticker)


def _dataframe_to_history(df: pd.DataFrame) -> list[dict]:
    """Convert a pandas DataFrame to a list of daily history dictionaries."""
    history = []
    df = _flatten_columns(df)
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        date_val = df.index[idx]
        
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y-%m-%d')
        else:
            date_str = str(date_val)[:10]
        
        history.append({
            "date": date_str,
            "open": round(_safe_float(row['Open']), 2),
            "high": round(_safe_float(row['High']), 2),
            "low": round(_safe_float(row['Low']), 2),
            "close": round(_safe_float(row['Close']), 2),
            "volume": _safe_int(row['Volume'])
        })
    
    return history


def _empty_response(ticker: str) -> dict:
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


async def get_stock_data(ticker: str) -> dict:
    """Async wrapper for fetching stock data."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _get_stock_data_sync, ticker)


async def get_multiple_stocks(tickers: list[str]) -> dict[str, dict]:
    """Fetch data for multiple stocks SEQUENTIALLY to avoid yfinance data corruption."""
    results = {}
    for ticker in tickers:
        try:
            result = await get_stock_data(ticker)
            results[ticker.upper()] = result
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            results[ticker.upper()] = _empty_response(ticker)
    return results


def _get_reference_close(ticker: str, on_or_before_date: datetime) -> Optional[float]:
    """Get the closing price from the last trading day on or before the given date.
    
    This is used to calculate the reference line - the close price at the START of the chart period.
    For example, for a 1D chart showing today, this returns yesterday's close.
    """
    ticker = ticker.upper()
    
    # Look back up to 10 days to find a trading day (handles weekends/holidays)
    start_date = on_or_before_date - timedelta(days=10)
    end_date = on_or_before_date + timedelta(days=1)  # Include the target date
    
    # Try stored data first
    stored = _get_stored_history(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        on_or_before_date.strftime('%Y-%m-%d')
    )
    
    if stored:
        # Get the most recent close on or before the target date
        return stored[-1]['close']
    
    # Fetch from API if not in database
    try:
        with _yfinance_lock:
            hist = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True,
                timeout=10
            )
        hist = _flatten_columns(hist)
        
        if not hist.empty:
            history = _dataframe_to_history(hist)
            # Filter to only dates on or before target
            target_str = on_or_before_date.strftime('%Y-%m-%d')
            history = [h for h in history if h['date'] <= target_str]
            _store_history(ticker, history)  # Cache for future use
            return history[-1]['close'] if history else None
    except Exception as e:
        print(f"Error fetching reference close for {ticker}: {e}")
    
    return None


def _get_stock_history_sync(ticker: str, period: str = "1y") -> dict:
    """Get historical data for a stock, using intraday for short periods.
    
    Returns: dict with:
        - 'history': list of price data points
        - 'reference_close': closing price before period starts
        - 'is_complete': whether data covers the full requested period
        - 'expected_start': the expected start date for the period
        - 'actual_start': the actual first date in the data
    """
    ticker = ticker.upper()
    
    # Determine if we need intraday data
    # yfinance interval availability:
    # - 1m: last 7 days only
    # - 5m: last 60 days
    # - 15m: last 60 days
    # - 1h: longer
    
    intraday_config = {
        "1d": ("1m", 2),    # 1 minute intervals, fetch 2 days to ensure we get last trading day
        "3d": ("5m", 5),    # 5 minute intervals, fetch extra days for safety
        "1w": ("15m", 10),  # 15 minute intervals, fetch extra days for safety
    }
    
    daily_periods = {
        "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
    }
    
    # For intraday, reference_close is calculated after we know which trading days we're showing
    # For daily periods, calculate based on calendar days
    daily_period_days = {
        "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
    }
    
    end_time = datetime.now()
    reference_close = None
    
    # Handle YTD (Year to Date) specially - from Jan 1st of current year
    if period == "ytd":
        year_start = datetime(end_time.year, 1, 1)
        days_since_year_start = (end_time - year_start).days
        daily_periods["ytd"] = days_since_year_start
        daily_period_days["ytd"] = days_since_year_start
    
    def check_data_completeness(history_data, expected_start_date: datetime, tolerance_days: int = 7) -> dict:
        """Check if the data covers the expected period.
        
        Returns dict with is_complete, expected_start, actual_start.
        """
        expected_start_str = expected_start_date.strftime('%Y-%m-%d')
        
        if not history_data:
            return {
                "is_complete": False,
                "expected_start": expected_start_str,
                "actual_start": None
            }
        
        sorted_data = sorted(history_data, key=lambda x: x['date'])
        actual_start = sorted_data[0]['date']
        
        # Parse actual start date
        if ' ' in actual_start:  # Intraday format "YYYY-MM-DD HH:MM"
            actual_start_date = datetime.strptime(actual_start.split(' ')[0], '%Y-%m-%d')
        else:
            actual_start_date = datetime.strptime(actual_start, '%Y-%m-%d')
        
        # Check if actual start is within tolerance of expected start
        days_difference = (actual_start_date - expected_start_date).days
        is_complete = days_difference <= tolerance_days
        
        return {
            "is_complete": is_complete,
            "expected_start": expected_start_str,
            "actual_start": actual_start.split(' ')[0] if ' ' in actual_start else actual_start
        }
    
    # Check if this is an intraday period
    if period in intraday_config:
        interval, fetch_days = intraday_config[period]
        
        # For intraday, use a wider range for database lookups to catch weekends/holidays
        db_start_time = end_time - timedelta(days=fetch_days + 5)  # Extra days for weekends
        
        # Try to get from database first
        stored = _get_stored_intraday(ticker, interval, db_start_time, end_time)
        
        # Check if stored data is recent AND covers enough of the trading day
        have_recent = False
        have_good_coverage = False
        if stored:
            latest = max(h['timestamp'] for h in stored)
            earliest = min(h['timestamp'] for h in stored)
            days_since_latest = (end_time - latest).days
            have_recent = days_since_latest < 4  # Allow for weekends and holidays
            
            # Check if data covers at least 4 hours of trading (not just 1 hour)
            # Group by date and check the time span for the most recent day
            from collections import defaultdict
            by_date = defaultdict(list)
            for h in stored:
                date_key = h['timestamp'].strftime('%Y-%m-%d')
                by_date[date_key].append(h['timestamp'])
            
            if by_date:
                most_recent_date = max(by_date.keys())
                times = sorted(by_date[most_recent_date])
                if len(times) >= 2:
                    time_span_hours = (times[-1] - times[0]).total_seconds() / 3600
                    have_good_coverage = time_span_hours >= 4  # At least 4 hours of data
                    print(f"DEBUG {ticker}: stored data spans {time_span_hours:.1f} hours on {most_recent_date}, {len(times)} points")
        
        # Calculate expected data points for the period
        # Extended hours: roughly 16 hours per day (4am-8pm) = 960 minutes
        intraday_trading_days = {"1d": 1, "3d": 3, "1w": 5}  # Trading days for each period
        minutes_per_interval = int(interval.replace('m', '')) if 'm' in interval else 60
        expected_points = int((intraday_trading_days.get(period, 1) * 600) / minutes_per_interval * 0.3)
        
        have_enough = len(stored) >= expected_points
        
        # Fetch fresh data if we don't have enough points OR data isn't recent OR coverage is poor
        need_fetch = not have_enough or not have_recent or not have_good_coverage
        
        if need_fetch:
            try:
                print(f"Fetching intraday ({interval}) data for {ticker}, {fetch_days} days with extended hours")
                with _yfinance_lock:
                    hist = yf.download(
                        ticker,
                        period=f"{fetch_days}d",
                        interval=interval,
                        progress=False,
                        auto_adjust=True,
                        prepost=True,  # Include pre-market and post-market data
                        timeout=15
                    )
                
                if not hist.empty:
                    new_history = _dataframe_to_intraday_history(hist)
                    
                    # Store new data
                    _store_intraday(ticker, interval, new_history)
                    
                    # Combine with stored data
                    all_history = {h['timestamp']: h for h in stored}
                    for h in new_history:
                        all_history[h['timestamp']] = h
                    
                    stored = list(all_history.values())
            except Exception as e:
                print(f"Error fetching intraday data for {ticker}: {e}")
        
        if stored:
            # Sort by timestamp
            stored = sorted(stored, key=lambda x: x['timestamp'])
            
            # Filter to the requested period
            result = _filter_intraday_by_period(stored, period)
            
            if result:
                # Get the trading days in the result to calculate reference close
                result_dates = sorted(set(h['timestamp'].strftime('%Y-%m-%d') for h in result if h.get('timestamp')))
                
                if result_dates:
                    # Reference close = close from the trading day BEFORE the first day in chart
                    first_chart_date = datetime.strptime(result_dates[0], '%Y-%m-%d')
                    reference_close = _get_reference_close(ticker, first_chart_date - timedelta(days=1))
                
                # Remove timestamp from response (frontend uses 'date' string)
                for h in result:
                    if 'timestamp' in h:
                        del h['timestamp']
                
                # For intraday, expected start depends on the period
                intraday_trading_days = {"1d": 1, "3d": 3, "1w": 5}
                trading_days_needed = intraday_trading_days.get(period, 1)
                # Estimate expected start (accounting for weekends)
                expected_start = end_time - timedelta(days=trading_days_needed + 2)
                completeness = check_data_completeness(result, expected_start, tolerance_days=3)
                
                return {
                    "history": result, 
                    "reference_close": reference_close,
                    **completeness
                }
        
        return {"history": [], "reference_close": None, "is_complete": False, "expected_start": None, "actual_start": None}
    
    # Daily data for longer periods
    days = daily_periods.get(period, 365)
    start_date = end_time - timedelta(days=days)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_time.strftime('%Y-%m-%d')
    
    # Get stored history from database
    stored_history = _get_stored_history(ticker, start_str, end_str)
    
    def calculate_daily_reference(history_data):
        """Calculate reference close from the first day of the chart data."""
        if history_data:
            sorted_history = sorted(history_data, key=lambda x: x['date'])
            first_date = sorted_history[0]['date']
            first_date_dt = datetime.strptime(first_date, '%Y-%m-%d')
            return _get_reference_close(ticker, first_date_dt - timedelta(days=1))
        return None
    
    def analyze_coverage(history_data, req_start: str, req_end: str):
        """Analyze what data we have and what gaps need to be filled."""
        if not history_data:
            return {
                "has_data": False,
                "actual_start": None,
                "actual_end": None,
                "missing_start_range": (req_start, req_end),  # Need everything
                "missing_end_range": None,
            }
        
        sorted_data = sorted(history_data, key=lambda x: x['date'])
        actual_start = sorted_data[0]['date']
        actual_end = sorted_data[-1]['date']
        
        # Check gap at the beginning
        start_gap_days = (datetime.strptime(actual_start, '%Y-%m-%d') - datetime.strptime(req_start, '%Y-%m-%d')).days
        missing_start_range = None
        if start_gap_days > 7:  # More than a week gap at start
            # Fetch from requested start to just before our actual start
            gap_end = (datetime.strptime(actual_start, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
            missing_start_range = (req_start, gap_end)
        
        # Check gap at the end (need recent data)
        end_gap_days = (datetime.strptime(req_end, '%Y-%m-%d') - datetime.strptime(actual_end, '%Y-%m-%d')).days
        missing_end_range = None
        if end_gap_days > 4:  # More than 4 days gap at end
            # Fetch from day after our actual end to requested end
            gap_start = (datetime.strptime(actual_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
            missing_end_range = (gap_start, req_end)
        
        return {
            "has_data": True,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "missing_start_range": missing_start_range,
            "missing_end_range": missing_end_range,
        }
    
    coverage = analyze_coverage(stored_history, start_str, end_str)
    
    # If we have complete coverage, just return what we have
    if coverage["has_data"] and not coverage["missing_start_range"] and not coverage["missing_end_range"]:
        print(f"Using {len(stored_history)} stored daily records for {ticker} (complete coverage)")
        reference_close = calculate_daily_reference(stored_history)
        return {
            "history": stored_history, 
            "reference_close": reference_close,
            "is_complete": True,
            "expected_start": start_str,
            "actual_start": coverage["actual_start"]
        }
    
    # Need to fetch missing data - only fetch the gaps, not data we already have
    all_history = {h['date']: h for h in stored_history}
    fetched_something = False
    fetch_error = False
    
    try:
        # Fetch missing start range (historical gap)
        if coverage["missing_start_range"]:
            gap_start, gap_end = coverage["missing_start_range"]
            print(f"Fetching historical gap for {ticker} from {gap_start} to {gap_end}")
            
            with _yfinance_lock:
                hist = yf.download(
                    ticker,
                    start=gap_start,
                    end=gap_end,
                    progress=False,
                    auto_adjust=True,
                    timeout=15
                )
            
            if not hist.empty:
                new_history = _dataframe_to_history(hist)
                _store_history(ticker, new_history)
                for h in new_history:
                    all_history[h['date']] = h
                fetched_something = True
        
        # Fetch missing end range (recent data)
        if coverage["missing_end_range"]:
            gap_start, gap_end = coverage["missing_end_range"]
            print(f"Fetching recent data for {ticker} from {gap_start} to {gap_end}")
            
            with _yfinance_lock:
                hist = yf.download(
                    ticker,
                    start=gap_start,
                    end=gap_end,
                    progress=False,
                    auto_adjust=True,
                    timeout=15
                )
            
            if not hist.empty:
                new_history = _dataframe_to_history(hist)
                _store_history(ticker, new_history)
                for h in new_history:
                    all_history[h['date']] = h
                fetched_something = True
                
    except Exception as e:
        print(f"Error fetching daily history for {ticker}: {e}")
        fetch_error = True
    
    # Sort and filter to requested range
    result = sorted(all_history.values(), key=lambda x: x['date'])
    result = [h for h in result if start_str <= h['date'] <= end_str]
    
    reference_close = calculate_daily_reference(result)
    result_start = result[0]['date'] if result else None
    
    # Determine completeness:
    # - If we successfully fetched (or attempted to fetch) from yfinance, mark complete
    #   (whatever yfinance returns IS all available data for this stock)
    # - Only mark incomplete if there was an error, so we retry later
    is_complete = not fetch_error
    
    return {
        "history": result, 
        "reference_close": reference_close,
        "is_complete": is_complete,
        "expected_start": start_str,
        "actual_start": result_start
    }


async def get_stock_history(ticker: str, period: str = "1y") -> dict:
    """Async wrapper for fetching stock history.
    
    Returns: dict with 'history' list and 'reference_close' (closing price before period).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _get_stock_history_sync, ticker, period)
