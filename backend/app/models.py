from sqlalchemy import Column, Integer, String, Float, DateTime, Date, ForeignKey, UniqueConstraint, Index, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, default="My Portfolio")
    total_value = Column(Float, default=10000.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # User preferences (persisted)
    chart_period = Column(String, default="1d")  # 1d, 3d, 1w, 1mo, 3mo, 6mo, ytd, 1y, 2y
    sort_field = Column(String, default="allocation")  # ticker, daily_change, allocation, equity, ytd
    sort_direction = Column(String, default="desc")  # asc, desc

    holdings = relationship("Holding", back_populates="portfolio", cascade="all, delete-orphan")
    option_holdings = relationship("OptionHolding", back_populates="portfolio", cascade="all, delete-orphan")


class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=False)
    shares = Column(Float, nullable=False, default=0)  # Number of shares owned
    avg_cost = Column(Float, nullable=True)  # Average cost per share
    added_at = Column(DateTime, default=datetime.utcnow)
    is_pinned = Column(Boolean, default=False, nullable=False)  # Pin to top of holdings list
    # Legacy column - kept for database compatibility, no longer used
    allocation_pct = Column(Float, nullable=True, default=0)

    portfolio = relationship("Portfolio", back_populates="holdings")


class PriceHistory(Base):
    """Stores historical price data (daily) to avoid repeated API calls."""
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False, index=True)  # YYYY-MM-DD format for daily data
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_ticker_date'),
    )


class IntradayPriceHistory(Base):
    """Stores intraday price data (1m, 5m, 15m intervals) for short-term charts."""
    __tablename__ = "intraday_price_history"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)  # Full datetime
    interval = Column(String, nullable=False)  # '1m', '5m', '15m', '1h'
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    __table_args__ = (
        UniqueConstraint('ticker', 'timestamp', 'interval', name='uix_ticker_timestamp_interval'),
        Index('ix_intraday_ticker_interval_time', 'ticker', 'interval', 'timestamp'),
    )


class StockAnalysisCache(Base):
    """
    Caches stock analysis data (fundamentals, options) with TTL.
    
    This data is semi-stale - refreshed every 15 minutes.
    Fundamentals rarely change, but we want reasonably fresh data.
    """
    __tablename__ = "stock_analysis_cache"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, unique=True, index=True)
    
    # Fundamentals
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    roic = Column(Float, nullable=True)  # Return on Invested Capital (%)
    roe = Column(Float, nullable=True)   # Return on Equity (%)
    roa = Column(Float, nullable=True)   # Return on Assets (%)
    profit_margin = Column(Float, nullable=True)
    operating_margin = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    forward_pe = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    
    # Options data
    call_open_interest = Column(Integer, nullable=True)
    put_open_interest = Column(Integer, nullable=True)
    call_volume = Column(Integer, nullable=True)
    put_volume = Column(Integer, nullable=True)
    call_put_ratio_oi = Column(Float, nullable=True)
    call_put_ratio_volume = Column(Float, nullable=True)
    avg_implied_volatility = Column(Float, nullable=True)
    iv_percentile = Column(Float, nullable=True)
    options_sentiment = Column(String, nullable=True)  # 'bullish', 'bearish', 'neutral'
    has_options = Column(Boolean, nullable=True, default=True)
    
    # Earnings date (for binary event risk)
    next_earnings_date = Column(DateTime, nullable=True)
    
    # Timestamp for TTL check
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class OptionHolding(Base):
    """
    Stores option contract positions (calls/puts, long/short).
    
    Options are fundamentally different from stock holdings:
    - They have expiration dates and strike prices
    - Each contract represents 100 shares of the underlying
    - Position can be long (bought) or short (sold/written)
    """
    __tablename__ = "option_holdings"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Option contract identifiers
    underlying_ticker = Column(String, nullable=False, index=True)  # e.g., "AAPL"
    option_type = Column(String, nullable=False)  # "call" or "put"
    position_type = Column(String, nullable=False)  # "long" or "short"
    strike_price = Column(Float, nullable=False)  # Strike price
    expiration_date = Column(Date, nullable=False, index=True)  # Expiration date
    
    # Position details
    contracts = Column(Integer, nullable=False, default=1)  # Number of contracts (each = 100 shares)
    premium_per_contract = Column(Float, nullable=True)  # Premium paid (long) or received (short) per contract
    
    # Metadata
    opened_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(String, nullable=True)  # Optional notes about the position
    
    portfolio = relationship("Portfolio", back_populates="option_holdings")
    
    __table_args__ = (
        Index('ix_option_underlying_exp', 'underlying_ticker', 'expiration_date'),
    )
