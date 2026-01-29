from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/portfolio.db")

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def migrate_db(conn):
    """Run simple migrations to add new columns to existing tables."""
    # Check holdings table columns
    result = await conn.execute(text("PRAGMA table_info(holdings)"))
    columns = [row[1] for row in result.fetchall()]
    
    # Legacy columns (kept for backwards compatibility during migration)
    if 'investment_date' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN investment_date DATETIME"))
        print("Added investment_date column to holdings table")
    
    if 'investment_price' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN investment_price FLOAT"))
        print("Added investment_price column to holdings table")
    
    # New position tracking columns
    if 'shares' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN shares FLOAT DEFAULT 0"))
        print("Added shares column to holdings table")
    
    if 'avg_cost' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN avg_cost FLOAT"))
        print("Added avg_cost column to holdings table")
    
    # Migration: Convert old allocation_pct based holdings to share-based
    # If a holding has allocation_pct but no shares, we can't auto-convert (need price data)
    # Just ensure shares defaults to 0 for existing holdings
    await conn.execute(text("UPDATE holdings SET shares = 0 WHERE shares IS NULL"))
    
    # Check and add user preference columns to portfolios table
    result = await conn.execute(text("PRAGMA table_info(portfolios)"))
    portfolio_columns = [row[1] for row in result.fetchall()]
    
    if 'chart_period' not in portfolio_columns:
        await conn.execute(text("ALTER TABLE portfolios ADD COLUMN chart_period VARCHAR DEFAULT '1d'"))
        await conn.execute(text("UPDATE portfolios SET chart_period = '1d' WHERE chart_period IS NULL"))
        print("Added chart_period column to portfolios table")
    
    if 'sort_field' not in portfolio_columns:
        await conn.execute(text("ALTER TABLE portfolios ADD COLUMN sort_field VARCHAR DEFAULT 'allocation'"))
        await conn.execute(text("UPDATE portfolios SET sort_field = 'allocation' WHERE sort_field IS NULL"))
        print("Added sort_field column to portfolios table")
    
    if 'sort_direction' not in portfolio_columns:
        await conn.execute(text("ALTER TABLE portfolios ADD COLUMN sort_direction VARCHAR DEFAULT 'desc'"))
        await conn.execute(text("UPDATE portfolios SET sort_direction = 'desc' WHERE sort_direction IS NULL"))
        print("Added sort_direction column to portfolios table")
    
    # Ensure defaults are set for any null values (in case migration ran before this fix)
    await conn.execute(text("UPDATE portfolios SET chart_period = '1d' WHERE chart_period IS NULL"))
    await conn.execute(text("UPDATE portfolios SET sort_field = 'allocation' WHERE sort_field IS NULL"))
    await conn.execute(text("UPDATE portfolios SET sort_direction = 'desc' WHERE sort_direction IS NULL"))
    
    # Create option_holdings table if it doesn't exist
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS option_holdings (
            id INTEGER PRIMARY KEY,
            portfolio_id INTEGER NOT NULL REFERENCES portfolios(id),
            underlying_ticker VARCHAR NOT NULL,
            option_type VARCHAR NOT NULL,
            position_type VARCHAR NOT NULL,
            strike_price FLOAT NOT NULL,
            expiration_date DATE NOT NULL,
            contracts INTEGER NOT NULL DEFAULT 1,
            premium_per_contract FLOAT,
            opened_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes VARCHAR
        )
    """))
    
    # Create indexes for option_holdings if they don't exist
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_option_holdings_underlying_ticker 
        ON option_holdings(underlying_ticker)
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_option_holdings_expiration_date 
        ON option_holdings(expiration_date)
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_option_underlying_exp 
        ON option_holdings(underlying_ticker, expiration_date)
    """))
    
    # Migrate stock_analysis_cache table - add next_earnings_date column
    result = await conn.execute(text("PRAGMA table_info(stock_analysis_cache)"))
    cache_columns = [row[1] for row in result.fetchall()]
    
    if cache_columns:  # Table exists
        if 'next_earnings_date' not in cache_columns:
            await conn.execute(text("ALTER TABLE stock_analysis_cache ADD COLUMN next_earnings_date DATETIME"))
            print("Added next_earnings_date column to stock_analysis_cache table")
    
    # ========================================================================
    # WFO Calibration Tables
    # ========================================================================
    
    # Create calibration_weights table
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS calibration_weights (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            indicator VARCHAR NOT NULL,
            action VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            weight FLOAT NOT NULL DEFAULT 1.0,
            sqn_score FLOAT,
            stability_passed BOOLEAN DEFAULT 1,
            window_end_date VARCHAR,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, indicator, action, horizon)
        )
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_calibration_weights_ticker
        ON calibration_weights(ticker)
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_calibration_ticker_horizon
        ON calibration_weights(ticker, horizon)
    """))

    # Migration: Add new columns to calibration_weights
    result = await conn.execute(text("PRAGMA table_info(calibration_weights)"))
    calibration_columns = [row[1] for row in result.fetchall()]

    if 'strategy_class' not in calibration_columns:
        await conn.execute(text("ALTER TABLE calibration_weights ADD COLUMN strategy_class VARCHAR DEFAULT 'all'"))
        print("Added strategy_class column to calibration_weights table")

    if 'overfit_warning' not in calibration_columns:
        await conn.execute(text("ALTER TABLE calibration_weights ADD COLUMN overfit_warning BOOLEAN DEFAULT 0"))
        print("Added overfit_warning column to calibration_weights table")

    if 'avg_train_sqn' not in calibration_columns:
        await conn.execute(text("ALTER TABLE calibration_weights ADD COLUMN avg_train_sqn FLOAT"))
        print("Added avg_train_sqn column to calibration_weights table")

    if 'avg_test_sqn' not in calibration_columns:
        await conn.execute(text("ALTER TABLE calibration_weights ADD COLUMN avg_test_sqn FLOAT"))
        print("Added avg_test_sqn column to calibration_weights table")

    if 'avg_gross_sqn' not in calibration_columns:
        await conn.execute(text("ALTER TABLE calibration_weights ADD COLUMN avg_gross_sqn FLOAT"))
        print("Added avg_gross_sqn column to calibration_weights table")

    # Create calibration_windows table
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS calibration_windows (
            id INTEGER PRIMARY KEY,
            ticker VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            train_start VARCHAR NOT NULL,
            train_end VARCHAR NOT NULL,
            test_start VARCHAR NOT NULL,
            test_end VARCHAR NOT NULL,
            window_days INTEGER NOT NULL,
            weights_json VARCHAR NOT NULL,
            train_sqn FLOAT,
            test_sqn FLOAT,
            expectancy FLOAT,
            trades_count INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_window_ticker_horizon 
        ON calibration_windows(ticker, horizon)
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_window_end_date 
        ON calibration_windows(ticker, test_end)
    """))
    
    # Create calibration_trades table
    await conn.execute(text("""
        CREATE TABLE IF NOT EXISTS calibration_trades (
            id INTEGER PRIMARY KEY,
            window_id INTEGER NOT NULL REFERENCES calibration_windows(id),
            ticker VARCHAR NOT NULL,
            entry_date VARCHAR NOT NULL,
            exit_date VARCHAR NOT NULL,
            horizon INTEGER NOT NULL,
            direction VARCHAR NOT NULL,
            entry_price FLOAT NOT NULL,
            exit_price FLOAT NOT NULL,
            pnl_pct FLOAT NOT NULL,
            transaction_cost FLOAT NOT NULL DEFAULT 0.001,
            market_regime VARCHAR
        )
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_trade_ticker_regime 
        ON calibration_trades(ticker, market_regime)
    """))
    await conn.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_trade_window 
        ON calibration_trades(window_id)
    """))
    
    print("WFO calibration tables created/verified")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Run migrations for existing tables
        await migrate_db(conn)

