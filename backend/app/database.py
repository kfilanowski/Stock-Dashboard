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


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Run migrations for existing tables
        await migrate_db(conn)

