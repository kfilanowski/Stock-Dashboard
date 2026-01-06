"""
Centralized configuration management for the Stock Dashboard API.

All configuration values should be defined here and imported elsewhere.
Supports environment variable overrides.
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "Stock Portfolio Dashboard API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./data/portfolio.db"
    database_url_sync: str = "sqlite:///./data/portfolio.db"
    
    # CORS
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173"
    ]
    
    # Cache settings
    cache_ttl_seconds: int = 60  # TTL for stock data cache (used for full stock data, not prices)
    
    # Data retention
    data_retention_days: int = 730  # 2 years
    
    # Stock service settings
    yfinance_max_workers: int = 10
    yfinance_timeout: int = 15
    yfinance_retries: int = 2
    
    # Market hours (Eastern Time, in minutes from midnight)
    market_open_minutes: int = 570   # 9:30 AM = 9*60 + 30
    market_close_minutes: int = 960  # 4:00 PM = 16*60
    
    # Intraday data settings
    intraday_min_coverage_hours: float = 4.0  # Minimum hours of trading data required
    intraday_expected_points_ratio: float = 0.3  # Expected data points ratio
    
    # Default portfolio settings
    default_portfolio_name: str = "My Portfolio"
    default_portfolio_value: float = 1000.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Allow environment variables to override
        # e.g., CACHE_TTL_SECONDS=120 overrides cache_ttl_seconds


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Get the settings instance. Useful for dependency injection."""
    return settings

