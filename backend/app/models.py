from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, Index
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


class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=False)
    allocation_pct = Column(Float, nullable=False)  # Percentage of portfolio (e.g., 5.0 for 5%)
    added_at = Column(DateTime, default=datetime.utcnow)
    investment_date = Column(DateTime, nullable=True)  # When user started tracking this stock
    investment_price = Column(Float, nullable=True)  # Stock price at investment_date for gain/loss calc

    portfolio = relationship("Portfolio", back_populates="holdings")


class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    total_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


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
