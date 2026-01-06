"""
Tests for cache module.
"""
import pytest
import time
from unittest.mock import patch

from app.services.cache import StockCache


class TestStockCache:
    """Tests for StockCache class."""
    
    def test_set_and_get(self):
        cache = StockCache(ttl_seconds=60)
        data = {"price": 100}
        
        cache.set("AAPL", data)
        result = cache.get("AAPL")
        
        assert result == data
    
    def test_get_nonexistent_key(self):
        cache = StockCache(ttl_seconds=60)
        result = cache.get("NONEXISTENT")
        
        assert result is None
    
    def test_expired_entry_returns_none(self):
        cache = StockCache(ttl_seconds=1)  # 1 second TTL
        cache.set("AAPL", {"price": 100})
        
        # Wait for expiration
        time.sleep(1.1)
        
        result = cache.get("AAPL")
        assert result is None
    
    def test_delete_existing_key(self):
        cache = StockCache(ttl_seconds=60)
        cache.set("AAPL", {"price": 100})
        
        deleted = cache.delete("AAPL")
        
        assert deleted is True
        assert cache.get("AAPL") is None
    
    def test_delete_nonexistent_key(self):
        cache = StockCache(ttl_seconds=60)
        
        deleted = cache.delete("NONEXISTENT")
        
        assert deleted is False
    
    def test_clear(self):
        cache = StockCache(ttl_seconds=60)
        cache.set("AAPL", {"price": 100})
        cache.set("GOOG", {"price": 200})
        
        count = cache.clear()
        
        assert count == 2
        assert cache.size == 0
    
    def test_cleanup_expired(self):
        cache = StockCache(ttl_seconds=1)
        cache.set("AAPL", {"price": 100})
        cache.set("GOOG", {"price": 200})
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Add fresh entry
        cache.set("MSFT", {"price": 300})
        
        # Cleanup expired
        removed = cache.cleanup_expired()
        
        assert removed == 2
        assert cache.get("MSFT") == {"price": 300}
    
    def test_size_property(self):
        cache = StockCache(ttl_seconds=60)
        
        assert cache.size == 0
        
        cache.set("AAPL", {"price": 100})
        assert cache.size == 1
        
        cache.set("GOOG", {"price": 200})
        assert cache.size == 2
    
    def test_uses_config_ttl_by_default(self):
        with patch('app.services.cache.settings') as mock_settings:
            mock_settings.cache_ttl_seconds = 120
            cache = StockCache()
            
            assert cache._ttl == 120

