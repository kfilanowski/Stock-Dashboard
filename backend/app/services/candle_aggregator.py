"""
Candle Aggregation Service for real-time chart updates.

Tracks in-progress 1-minute candles and manages OHLC updates as
live prices arrive every 3 seconds. Persists completed candles to SQLite.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import threading

import pytz

from ..config import settings
from ..logging_config import get_logger
from .price_history import PriceHistoryService, get_price_history_service

logger = get_logger(__name__)

# US Eastern timezone for market hours
MARKET_TZ = pytz.timezone('US/Eastern')


@dataclass
class CandleState:
    """Represents an in-progress candle being built from live prices."""
    ticker: str
    interval_key: str  # "YYYY-MM-DD HH:MM" in Eastern time (rounded to interval)
    timestamp: datetime  # Full datetime with timezone
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    update_count: int = 1  # Number of price updates in this candle
    
    def update(self, price: float, volume: int = 0) -> None:
        """Update the candle with a new price tick."""
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.update_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "date": self.interval_key,
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": self.volume
        }
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "timestamp": self.timestamp,
            "open": round(self.open, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "close": round(self.close, 2),
            "volume": self.volume
        }


class CandleAggregator:
    """
    Service for aggregating live price updates into 5-minute candles.
    
    Tracks in-progress candles for each ticker in memory. When a new
    5-minute period starts, the previous candle is closed and persisted to SQLite.
    
    Thread-safe for concurrent updates from multiple API calls.
    """
    
    # Candle interval in minutes (matches 1d chart interval)
    CANDLE_INTERVAL_MINUTES = 5
    
    def __init__(
        self,
        price_history: Optional[PriceHistoryService] = None,
        interval: str = "5m"
    ):
        """
        Initialize the candle aggregator.
        
        Args:
            price_history: Price history service for persistence.
            interval: Candle interval (default "5m").
        """
        self._price_history = price_history or get_price_history_service()
        self._interval = interval
        
        # In-progress candles: ticker -> CandleState
        self._candles: Dict[str, CandleState] = {}
        
        # Lock for thread-safe access
        self._lock = threading.Lock()
    
    def _get_current_interval_key(self, now: Optional[datetime] = None) -> tuple[str, datetime]:
        """
        Get the current interval key in Eastern time.
        
        Rounds down to the nearest 5-minute interval.
        E.g., 10:03 -> 10:00, 10:07 -> 10:05, 10:12 -> 10:10
        
        Returns:
            Tuple of (interval_key string "YYYY-MM-DD HH:MM", full datetime with tz)
        """
        if now is None:
            now = datetime.now(MARKET_TZ)
        elif now.tzinfo is None:
            now = MARKET_TZ.localize(now)
        else:
            now = now.astimezone(MARKET_TZ)
        
        # Round down to nearest 5-minute interval
        interval_minutes = self.CANDLE_INTERVAL_MINUTES
        rounded_minute = (now.minute // interval_minutes) * interval_minutes
        interval_dt = now.replace(minute=rounded_minute, second=0, microsecond=0)
        interval_key = interval_dt.strftime('%Y-%m-%d %H:%M')
        
        return interval_key, interval_dt
    
    def _is_trading_hours(self, dt: datetime) -> bool:
        """
        Check if the given time is during extended trading hours.
        
        Extended hours: 4am - 8pm Eastern, weekdays only.
        """
        # Ensure we're working with Eastern time
        if dt.tzinfo is None:
            dt = MARKET_TZ.localize(dt)
        else:
            dt = dt.astimezone(MARKET_TZ)
        
        # Weekend - no trading
        if dt.weekday() >= 5:
            return False
        
        # Extended hours are 4am to 8pm Eastern
        return 4 <= dt.hour < 20
    
    def update_price(
        self,
        ticker: str,
        price: float,
        volume: int = 0,
        now: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update candle with a new price tick.
        
        If this is a new minute, closes the previous candle, persists it,
        and starts a new one.
        
        Only updates during trading hours (4am-8pm Eastern, weekdays).
        Returns None if outside trading hours.
        
        Args:
            ticker: Stock ticker symbol.
            price: Current price.
            volume: Volume for this tick (optional).
            now: Current time (optional, for testing).
            
        Returns:
            Current candle state as dict (for API response), or None if outside trading hours.
        """
        ticker = ticker.upper()
        interval_key, interval_dt = self._get_current_interval_key(now)
        
        # Don't build candles outside trading hours
        if not self._is_trading_hours(interval_dt):
            return None
        
        with self._lock:
            existing = self._candles.get(ticker)
            
            if existing is None:
                # First price for this ticker - create new candle
                candle = CandleState(
                    ticker=ticker,
                    interval_key=interval_key,
                    timestamp=interval_dt,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=volume
                )
                self._candles[ticker] = candle
                logger.debug(f"Started new candle for {ticker} at {interval_key}")
                return candle.to_dict()
            
            if existing.interval_key == interval_key:
                # Same interval - update existing candle
                existing.update(price, volume)
                return existing.to_dict()
            
            # New interval - close previous candle and start new one
            self._close_candle(existing)
            
            # Create new candle
            candle = CandleState(
                ticker=ticker,
                interval_key=interval_key,
                timestamp=interval_dt,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=volume
            )
            self._candles[ticker] = candle
            logger.debug(f"Closed candle for {ticker}, started new at {interval_key}")
            return candle.to_dict()
    
    def _close_candle(self, candle: CandleState) -> None:
        """
        Close a candle and persist it to SQLite.
        
        Args:
            candle: The candle to close and persist.
        """
        try:
            # Convert to storage format with timestamp
            storage_data = candle.to_storage_dict()
            
            # Store in database
            self._price_history.store_intraday_history(
                ticker=candle.ticker,
                interval=self._interval,
                history=[storage_data]
            )
            
            logger.debug(
                f"Persisted candle {candle.ticker} {candle.interval_key}: "
                f"O={candle.open} H={candle.high} L={candle.low} C={candle.close}"
            )
        except Exception as e:
            logger.error(f"Failed to persist candle for {candle.ticker}: {e}")
    
    def clear_ticker(self, ticker: str) -> bool:
        """
        Remove in-progress candle for a ticker without persisting.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            True if candle was removed, False if not found.
        """
        ticker = ticker.upper()
        
        with self._lock:
            if ticker in self._candles:
                del self._candles[ticker]
                return True
            return False


# Singleton instance
_candle_aggregator: Optional[CandleAggregator] = None


def get_candle_aggregator() -> CandleAggregator:
    """Get the singleton candle aggregator instance."""
    global _candle_aggregator
    if _candle_aggregator is None:
        _candle_aggregator = CandleAggregator()
    return _candle_aggregator

