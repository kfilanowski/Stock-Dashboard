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
    # Check and add investment_date column to holdings table
    result = await conn.execute(text("PRAGMA table_info(holdings)"))
    columns = [row[1] for row in result.fetchall()]
    
    if 'investment_date' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN investment_date DATETIME"))
        print("Added investment_date column to holdings table")
    
    if 'investment_price' not in columns:
        await conn.execute(text("ALTER TABLE holdings ADD COLUMN investment_price FLOAT"))
        print("Added investment_price column to holdings table")
    
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


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Run migrations for existing tables
        await migrate_db(conn)

