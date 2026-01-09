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


# ============================================================================
# Walk-Forward Optimization (WFO) Calibration Models
# ============================================================================

class CalibrationWeights(Base):
    """
    Stores calibrated indicator weights per-stock, per-indicator, per-horizon.
    
    These weights are learned through Walk-Forward Optimization and used
    to customize the scoring engine for each stock's "personality".
    """
    __tablename__ = "calibration_weights"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    indicator = Column(String, nullable=False)  # 'rsi', 'macd', 'bollinger', etc.
    action = Column(String, nullable=False)  # 'buyShares', 'sellShares', etc.
    horizon = Column(Integer, nullable=False)  # 3 (swing) or 15 (trend)
    
    # Calibrated values
    weight = Column(Float, nullable=False, default=1.0)  # 0.0 to 2.5
    sqn_score = Column(Float, nullable=True)  # SQN at this weight
    stability_passed = Column(Boolean, default=True)  # Neighbor validation passed
    
    # Window tracking
    window_end_date = Column(String, nullable=True)  # YYYY-MM-DD of training window end
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('ticker', 'indicator', 'action', 'horizon', name='uix_calibration_key'),
        Index('ix_calibration_ticker_horizon', 'ticker', 'horizon'),
    )


class CalibrationWindow(Base):
    """
    Records each rolling window in the Walk-Forward Optimization process.
    
    Tracks train/test periods, optimized weights, and performance metrics
    to enable analysis of weight drift and overfitting detection.
    """
    __tablename__ = "calibration_windows"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, nullable=False, index=True)
    horizon = Column(Integer, nullable=False)  # 3 or 15
    
    # Window boundaries
    train_start = Column(String, nullable=False)  # YYYY-MM-DD
    train_end = Column(String, nullable=False)
    test_start = Column(String, nullable=False)
    test_end = Column(String, nullable=False)
    window_days = Column(Integer, nullable=False)  # Actual training window size (adaptive)
    
    # Optimized weights (JSON string of {indicator: {action: weight}})
    weights_json = Column(String, nullable=False)
    
    # Performance metrics
    train_sqn = Column(Float, nullable=True)  # In-sample SQN
    test_sqn = Column(Float, nullable=True)  # Out-of-sample SQN
    expectancy = Column(Float, nullable=True)  # Average return per trade
    trades_count = Column(Integer, nullable=True)  # Number of trades in window
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to trades
    trades = relationship("CalibrationTrade", back_populates="window", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_window_ticker_horizon', 'ticker', 'horizon'),
        Index('ix_window_end_date', 'ticker', 'test_end'),
    )


class CalibrationTrade(Base):
    """
    Logs simulated trades from the WFO backtesting process.
    
    Each trade is tagged with market regime to enable regime-specific
    performance analysis (e.g., "RSI in BEAR_VOLATILE" queries).
    """
    __tablename__ = "calibration_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    window_id = Column(Integer, ForeignKey("calibration_windows.id"), nullable=False)
    ticker = Column(String, nullable=False, index=True)
    
    # Trade details
    entry_date = Column(String, nullable=False)  # YYYY-MM-DD
    exit_date = Column(String, nullable=False)
    horizon = Column(Integer, nullable=False)  # 3 or 15
    direction = Column(String, nullable=False)  # 'long' or 'short'
    
    # Prices and P&L
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    pnl_pct = Column(Float, nullable=False)  # Profit/loss percentage
    transaction_cost = Column(Float, nullable=False, default=0.001)  # 0.1% default
    
    # 6-State Market Regime at entry
    # Values: BULL_QUIET, BULL_VOLATILE, BEAR_QUIET, BEAR_VOLATILE, 
    #         NEUTRAL_CHOP, NEUTRAL_VOLATILE
    market_regime = Column(String, nullable=True)
    
    # Relationship back to window
    window = relationship("CalibrationWindow", back_populates="trades")
    
    __table_args__ = (
        Index('ix_trade_ticker_regime', 'ticker', 'market_regime'),
        Index('ix_trade_window', 'window_id'),
    )
