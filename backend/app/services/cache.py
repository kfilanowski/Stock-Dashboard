"""
Cache management for stock data.

Provides in-memory caching with TTL to reduce API calls to yfinance.
"""
import time
from typing import Optional, Dict, Any, Tuple

from ..config import settings
from ..logging_config import get_logger

logger = get_logger(__name__)


class StockCache:
    """
    In-memory cache for stock data with time-to-live (TTL) support.
    
    Thread-safe for read operations. Write operations should be 
    synchronized externally if called from multiple threads.
    """
    
    def __init__(self, ttl_seconds: Optional[int] = None):
        """
        Initialize the cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries. Defaults to config value.
        """
        self._cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
        self._ttl = ttl_seconds or settings.cache_ttl_seconds
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if not expired.
        
        Args:
            key: Cache key.
            
        Returns:
            Cached data if exists and not expired, None otherwise.
        """
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                logger.debug(f"Cache hit for {key}")
                return data
            else:
                logger.debug(f"Cache expired for {key}")
                del self._cache[key]
        return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """
        Store data in cache.
        
        Args:
            key: Cache key.
            data: Data to cache.
        """
        self._cache[key] = (data, time.time())
        logger.debug(f"Cached data for {key}")
    
    def delete(self, key: str) -> bool:
        """
        Remove an entry from cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if entry was removed, False if not found.
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Deleted cache entry for {key}")
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed.
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self._ttl
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    @property
    def size(self) -> int:
        """Return the number of entries in cache."""
        return len(self._cache)


# Singleton instance
_stock_cache: Optional[StockCache] = None


def get_stock_cache() -> StockCache:
    """Get the singleton stock cache instance."""
    global _stock_cache
    if _stock_cache is None:
        _stock_cache = StockCache()
    return _stock_cache

