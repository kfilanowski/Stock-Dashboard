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


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Run migrations for existing tables
        await migrate_db(conn)

