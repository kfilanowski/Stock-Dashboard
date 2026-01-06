"""
Price history data storage and retrieval.

Handles both daily and intraday price data persistence.
"""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session
import pytz

from ..config import settings
from ..logging_config import get_logger
from ..models import PriceHistory, IntradayPriceHistory

logger = get_logger(__name__)

# US Eastern timezone for market hours
MARKET_TZ = pytz.timezone('US/Eastern')


class PriceHistoryService:
    """
    Service for storing and retrieving price history data.
    
    Uses synchronous SQLAlchemy for compatibility with yfinance's
    synchronous data fetching.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the price history service.
        
        Args:
            database_url: Database URL. Defaults to config value.
        """
        url = database_url or settings.database_url_sync
        self._engine = create_engine(url, connect_args={"check_same_thread": False})
    
    # ============ Daily Price History ============
    
    def get_daily_history(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Get daily price history from database for a date range.
        
        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            List of price data dictionaries.
        """
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
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
    
    def store_daily_history(self, ticker: str, history: List[Dict[str, Any]]) -> int:
        """
        Store daily price history in database. Skips duplicates.
        
        Args:
            ticker: Stock ticker symbol.
            history: List of price data dictionaries.
            
        Returns:
            Number of new records inserted.
        """
        if not history:
            return 0
        
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
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
                logger.info(
                    f"Stored {len(new_records)} new daily records for {ticker}, "
                    f"last_close={new_records[-1].close}"
                )
            
            return len(new_records)
    
    # ============ Intraday Price History ============
    
    def get_intraday_history(
        self, 
        ticker: str, 
        interval: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Get intraday price history from database.
        
        Args:
            ticker: Stock ticker symbol.
            interval: Time interval (e.g., '1m', '5m', '15m').
            start_time: Start datetime.
            end_time: End datetime.
            
        Returns:
            List of price data dictionaries with timestamps.
        """
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
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
    
    def store_intraday_history(
        self, 
        ticker: str, 
        interval: str, 
        history: List[Dict[str, Any]]
    ) -> int:
        """
        Store intraday price history in database. Skips duplicates.
        
        Args:
            ticker: Stock ticker symbol.
            interval: Time interval.
            history: List of price data dictionaries with timestamps.
            
        Returns:
            Number of new records inserted.
        """
        if not history:
            return 0
        
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
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
                logger.info(
                    f"Stored {len(new_records)} new intraday ({interval}) records for {ticker}"
                )
            
            return len(new_records)
    
    def clear_intraday_cache(self, ticker: str) -> int:
        """
        Clear all intraday cache data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol.
            
        Returns:
            Number of records deleted.
        """
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
            result = session.execute(
                delete(IntradayPriceHistory)
                .where(IntradayPriceHistory.ticker == ticker)
            )
            session.commit()
            deleted_count = result.rowcount
            logger.info(f"Cleared {deleted_count} intraday records for {ticker}")
            return deleted_count
    
    # ============ Incremental Fetch Support ============
    
    def get_latest_intraday_timestamp(
        self, 
        ticker: str, 
        interval: str
    ) -> Optional[datetime]:
        """
        Get the most recent timestamp we have stored for a ticker/interval.
        
        This enables incremental fetching - only request data after this point.
        
        Args:
            ticker: Stock ticker symbol.
            interval: Time interval (e.g., '1m', '5m', '15m').
            
        Returns:
            Most recent timestamp or None if no data exists.
        """
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
            from sqlalchemy import func
            result = session.execute(
                select(func.max(IntradayPriceHistory.timestamp))
                .where(IntradayPriceHistory.ticker == ticker)
                .where(IntradayPriceHistory.interval == interval)
            )
            row = result.scalar()
            return row
    
    # ============ Data Cleanup ============
    
    def cleanup_old_data(self) -> tuple[int, int]:
        """
        Delete price history data older than retention period.
        
        Returns:
            Tuple of (daily_deleted, intraday_deleted) counts.
        """
        cutoff_date = datetime.now() - timedelta(days=settings.data_retention_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        with Session(self._engine) as session:
            # Delete old daily data
            daily_result = session.execute(
                delete(PriceHistory).where(PriceHistory.date < cutoff_str)
            )
            
            # Delete old intraday data
            intraday_result = session.execute(
                delete(IntradayPriceHistory)
                .where(IntradayPriceHistory.timestamp < cutoff_date)
            )
            
            session.commit()
            
            daily_deleted = daily_result.rowcount
            intraday_deleted = intraday_result.rowcount
            
            if daily_deleted > 0 or intraday_deleted > 0:
                logger.info(
                    f"Cleaned up {daily_deleted} daily and {intraday_deleted} intraday "
                    f"records older than {settings.data_retention_days} days"
                )
            
            return daily_deleted, intraday_deleted
    
    # ============ Utility Methods ============
    
    def filter_intraday_by_period(
        self, 
        history: List[Dict[str, Any]], 
        period: str
    ) -> List[Dict[str, Any]]:
        """
        Filter intraday history to match the requested period.
        
        Args:
            history: Full intraday history.
            period: Period string ('1d', '3d', '1w').
            
        Returns:
            Filtered history containing most recent trading days.
        """
        if not history:
            return []
        
        # Map period to number of trading days
        trading_days_map = {"1d": 1, "3d": 3, "1w": 5, "1mo": 21}
        target_days = trading_days_map.get(period, 1)
        
        # Group by date
        by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for h in history:
            ts = h.get('timestamp')
            if ts:
                date_key = ts.strftime('%Y-%m-%d')
                by_date[date_key].append(h)
        
        # Sort dates descending and take the most recent N
        sorted_dates = sorted(by_date.keys(), reverse=True)[:target_days]
        
        # Collect all data from those dates, sorted by timestamp
        result = []
        for date in sorted(sorted_dates):
            result.extend(sorted(by_date[date], key=lambda x: x['timestamp']))
        
        return result
    
    def analyze_coverage(
        self, 
        history: List[Dict[str, Any]], 
        requested_start: str, 
        requested_end: str
    ) -> Dict[str, Any]:
        """
        Analyze what data coverage we have and identify gaps.
        
        Args:
            history: Current history data.
            requested_start: Requested start date (YYYY-MM-DD).
            requested_end: Requested end date (YYYY-MM-DD).
            
        Returns:
            Dict with coverage analysis including missing ranges.
        """
        if not history:
            return {
                "has_data": False,
                "actual_start": None,
                "actual_end": None,
                "missing_start_range": (requested_start, requested_end),
                "missing_end_range": None,
            }
        
        sorted_data = sorted(history, key=lambda x: x['date'])
        actual_start = sorted_data[0]['date']
        actual_end = sorted_data[-1]['date']
        
        # Check gap at beginning
        start_gap_days = (
            datetime.strptime(actual_start, '%Y-%m-%d') - 
            datetime.strptime(requested_start, '%Y-%m-%d')
        ).days
        
        missing_start_range = None
        if start_gap_days > 7:  # More than a week gap at start
            gap_end = (
                datetime.strptime(actual_start, '%Y-%m-%d') - timedelta(days=1)
            ).strftime('%Y-%m-%d')
            missing_start_range = (requested_start, gap_end)
        
        # Check gap at end
        end_gap_days = (
            datetime.strptime(requested_end, '%Y-%m-%d') - 
            datetime.strptime(actual_end, '%Y-%m-%d')
        ).days
        
        missing_end_range = None
        if end_gap_days > 4:  # More than 4 days gap at end
            gap_start = (
                datetime.strptime(actual_end, '%Y-%m-%d') + timedelta(days=1)
            ).strftime('%Y-%m-%d')
            missing_end_range = (gap_start, requested_end)
        
        return {
            "has_data": True,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "missing_start_range": missing_start_range,
            "missing_end_range": missing_end_range,
        }
    
    def analyze_intraday_coverage(
        self,
        ticker: str,
        interval: str,
        requested_start: datetime,
        requested_end: datetime
    ) -> Dict[str, Any]:
        """
        Analyze intraday data coverage and identify time gaps.
        
        This is the key method for smart gap-filling - it determines what
        data we have and what ranges need to be fetched from the API.
        
        Key logic:
        - If we're in trading hours and don't have today's data, fetch it
        - Don't flag overnight/weekend gaps as needing fetch (no data exists)
        - Flag internal gaps during trading hours
        
        Args:
            ticker: Stock ticker symbol.
            interval: Time interval (e.g., '1m', '5m', '15m').
            requested_start: Start of desired range.
            requested_end: End of desired range.
            
        Returns:
            Dict with coverage analysis:
            - has_data: bool
            - stored_count: int
            - actual_start: datetime or None
            - actual_end: datetime or None
            - missing_ranges: List of (start, end) tuples for gaps
            - needs_fetch: bool
        """
        ticker = ticker.upper()
        
        # Get stored data for the range
        stored = self.get_intraday_history(ticker, interval, requested_start, requested_end)
        
        if not stored:
            return {
                "has_data": False,
                "stored_count": 0,
                "actual_start": None,
                "actual_end": None,
                "missing_ranges": [(requested_start, requested_end)],
                "needs_fetch": True,
                "stored_data": []
            }
        
        # Sort by timestamp
        sorted_data = sorted(stored, key=lambda x: x['timestamp'])
        actual_start = sorted_data[0]['timestamp']
        actual_end = sorted_data[-1]['timestamp']
        
        # Determine interval duration in minutes
        interval_minutes = self._parse_interval_minutes(interval)
        
        # Find gaps in the data - but be smart about market hours
        missing_ranges = []
        
        # IMPORTANT: All times must be in Eastern timezone for proper comparison
        # Stored data is already in Eastern time (converted during storage)
        # We need to get current time in Eastern for accurate comparison
        now_eastern = datetime.now(MARKET_TZ).replace(tzinfo=None)
        today_eastern = now_eastern.date()
        actual_end_date = actual_end.date()
        
        logger.debug(
            f"Coverage check for {ticker}: now_eastern={now_eastern}, "
            f"today_eastern={today_eastern}, actual_end={actual_end}, "
            f"actual_end_date={actual_end_date}"
        )
        
        # Helper to check if a time is during potential trading hours
        # Extended hours: 4am - 8pm Eastern (inclusive of 8pm to catch last candle)
        def is_trading_hours(dt: datetime) -> bool:
            hour = dt.hour
            weekday = dt.weekday()
            # Weekend - no trading
            if weekday >= 5:
                return False
            # Extended hours are 4am to 8pm (use <= 20 to include 8pm candle)
            return 4 <= hour <= 20
        
        # Helper to check if we're currently in an active trading window
        def is_market_potentially_active(dt: datetime) -> bool:
            hour = dt.hour
            weekday = dt.weekday()
            if weekday >= 5:
                return False
            # Market could be active 4am - 8pm
            return 4 <= hour < 20
        
        # KEY FIX: Check if we need today's data
        # If we're in trading hours and our latest data is from a previous day,
        # we definitely need to fetch today's data
        if is_market_potentially_active(now_eastern) and actual_end_date < today_eastern:
            # We're in trading hours but have no data from today
            # This means we need to fetch today's trading data
            logger.info(
                f"Coverage gap: {ticker} latest data from {actual_end_date}, "
                f"need today's data ({today_eastern})"
            )
            # Fetch from market open today (or from actual_end, whichever is later)
            today_market_open = datetime.combine(today_eastern, datetime.min.time().replace(hour=4))
            fetch_start = max(actual_end + timedelta(minutes=interval_minutes), today_market_open)
            missing_ranges.append((fetch_start, requested_end))
        
        elif is_market_potentially_active(now_eastern) and actual_end < requested_end:
            # Same day - check if we're missing recent data
            gap_minutes = (requested_end - actual_end).total_seconds() / 60
            # Flag if gap is significant (more than a few intervals)
            if gap_minutes > interval_minutes * 5:
                missing_ranges.append((
                    actual_end + timedelta(minutes=interval_minutes), 
                    requested_end
                ))
        
        # Check for internal gaps during trading hours only
        prev_ts = sorted_data[0]['timestamp']
        for point in sorted_data[1:]:
            current_ts = point['timestamp']
            gap_minutes = (current_ts - prev_ts).total_seconds() / 60
            
            # Only consider large gaps (at least 1 hour or 15 intervals)
            gap_threshold = max(60, interval_minutes * 15)
            if gap_minutes > gap_threshold:
                # Skip overnight/weekend gaps
                prev_date = prev_ts.date()
                curr_date = current_ts.date()
                
                # If same day and both during trading hours, it's a real gap
                if prev_date == curr_date:
                    if is_trading_hours(prev_ts) and is_trading_hours(current_ts):
                        missing_ranges.append((
                            prev_ts + timedelta(minutes=interval_minutes),
                            current_ts - timedelta(minutes=interval_minutes)
                        ))
                # Different days - only flag if it's not a normal overnight gap
                # (e.g., missing data from Friday afternoon or Monday morning within same week)
                elif (curr_date - prev_date).days == 1:
                    # Consecutive days - check if gap spans expected close to open
                    # Normal overnight: 8pm to 4am = expected
                    # But if prev_ts is 2pm and curr_ts is 10am, that's abnormal
                    prev_after_close = prev_ts.hour >= 20
                    curr_before_open = current_ts.hour < 4
                    if not (prev_after_close or curr_before_open):
                        # Unusual gap - might be missing data
                        if is_trading_hours(prev_ts) and is_trading_hours(current_ts):
                            missing_ranges.append((
                                prev_ts + timedelta(minutes=interval_minutes),
                                current_ts - timedelta(minutes=interval_minutes)
                            ))
            
            prev_ts = current_ts
        
        return {
            "has_data": True,
            "stored_count": len(stored),
            "actual_start": actual_start,
            "actual_end": actual_end,
            "missing_ranges": missing_ranges,
            "needs_fetch": len(missing_ranges) > 0,
            "stored_data": stored
        }
    
    def _parse_interval_minutes(self, interval: str) -> int:
        """Parse interval string to minutes."""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        return 1  # Default to 1 minute


# Singleton instance
_price_history_service: Optional[PriceHistoryService] = None


def get_price_history_service() -> PriceHistoryService:
    """Get the singleton price history service instance."""
    global _price_history_service
    if _price_history_service is None:
        _price_history_service = PriceHistoryService()
    return _price_history_service

