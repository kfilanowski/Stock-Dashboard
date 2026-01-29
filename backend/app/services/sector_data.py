"""
Sector Data Service

Provides sector ETF mappings and data access for relative strength calculations.
Stocks are compared against their sector benchmark (XLK, XLF, etc.) to determine
relative performance.

Sector ETFs used:
- XLK: Technology
- XLF: Financials
- XLV: Health Care
- XLE: Energy
- XLI: Industrials
- XLY: Consumer Discretionary
- XLP: Consumer Staples
- XLU: Utilities
- XLB: Materials
- XLRE: Real Estate
- XLC: Communication Services
- SPY: S&P 500 (default/broad market)
"""

import logging
from typing import Optional, Dict
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models import PriceHistory, StockAnalysisCache

logger = logging.getLogger(__name__)


# ============================================================================
# Sector ETF Mapping
# ============================================================================

# Map sector names to their corresponding Select Sector SPDR ETFs
SECTOR_TO_ETF = {
    'Technology': 'XLK',
    'Information Technology': 'XLK',
    'Financial Services': 'XLF',
    'Financials': 'XLF',
    'Financial': 'XLF',
    'Healthcare': 'XLV',
    'Health Care': 'XLV',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Industrial': 'XLI',
    'Consumer Cyclical': 'XLY',
    'Consumer Discretionary': 'XLY',
    'Consumer Defensive': 'XLP',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Basic Materials': 'XLB',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC',
    'Telecommunications': 'XLC',
}

# All sector ETFs that should be fetched for calibration
ALL_SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC', 'SPY']

# Default benchmark when sector is unknown
DEFAULT_BENCHMARK = 'SPY'


# ============================================================================
# Sector Lookup
# ============================================================================

async def get_sector_for_ticker(ticker: str, db: AsyncSession) -> Optional[str]:
    """
    Get the sector for a ticker from the StockAnalysisCache.

    Args:
        ticker: Stock symbol
        db: Database session

    Returns:
        Sector name or None if not found
    """
    result = await db.execute(
        select(StockAnalysisCache).where(StockAnalysisCache.ticker == ticker)
    )
    cache = result.scalar_one_or_none()

    if cache and cache.sector:
        return cache.sector
    return None


def get_sector_etf_from_sector(sector: Optional[str]) -> str:
    """
    Get the sector ETF symbol for a given sector name.

    Args:
        sector: Sector name (e.g., 'Technology', 'Health Care')

    Returns:
        Sector ETF symbol (e.g., 'XLK' for tech stocks)
    """
    if sector and sector in SECTOR_TO_ETF:
        return SECTOR_TO_ETF[sector]

    logger.debug(f"[SECTOR] No sector mapping for {sector}, using {DEFAULT_BENCHMARK}")
    return DEFAULT_BENCHMARK


async def get_sector_etf_for_ticker(ticker: str, db: AsyncSession) -> str:
    """
    Get the sector ETF symbol for a given stock ticker.

    Args:
        ticker: Stock symbol
        db: Database session

    Returns:
        Sector ETF symbol (e.g., 'XLK' for tech stocks)
    """
    sector = await get_sector_for_ticker(ticker, db)
    return get_sector_etf_from_sector(sector)


# ============================================================================
# Price History Access
# ============================================================================

async def get_sector_etf_history(
    sector_etf: str,
    db: AsyncSession,
    start_date: str = None,
    end_date: str = None
) -> Optional[pd.DataFrame]:
    """
    Get price history for a sector ETF from the database.

    Args:
        sector_etf: ETF symbol (e.g., 'XLK')
        db: Database session
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data or None if not found
    """
    query = select(PriceHistory).where(PriceHistory.ticker == sector_etf)

    if start_date:
        query = query.where(PriceHistory.date >= start_date)
    if end_date:
        query = query.where(PriceHistory.date <= end_date)

    query = query.order_by(PriceHistory.date.asc())
    result = await db.execute(query)
    records = result.scalars().all()

    if not records:
        logger.warning(f"[SECTOR] No price history for {sector_etf}")
        return None

    df = pd.DataFrame([{
        'date': r.date,
        'open': r.open,
        'high': r.high,
        'low': r.low,
        'close': r.close,
        'volume': r.volume
    } for r in records])

    logger.debug(f"[SECTOR] Loaded {len(df)} days for {sector_etf}")
    return df


async def get_benchmark_for_stock(
    ticker: str,
    db: AsyncSession,
    start_date: str = None,
    end_date: str = None
) -> Optional[pd.DataFrame]:
    """
    Get the appropriate benchmark (sector ETF) price history for a stock.

    This is the main entry point for relative strength calculations.

    Args:
        ticker: Stock symbol
        db: Database session
        start_date: Optional start date
        end_date: Optional end date

    Returns:
        DataFrame with benchmark OHLCV data
    """
    sector_etf = await get_sector_etf_for_ticker(ticker, db)
    return await get_sector_etf_history(sector_etf, db, start_date, end_date)


async def check_sector_etf_availability(db: AsyncSession) -> Dict[str, bool]:
    """
    Check which sector ETFs have price data in the database.

    Returns:
        Dict of {etf_symbol: has_data}
    """
    from sqlalchemy import func

    availability = {}
    for etf in ALL_SECTOR_ETFS:
        result = await db.execute(
            select(func.count()).where(PriceHistory.ticker == etf)
        )
        count = result.scalar()
        availability[etf] = count > 0
        if count > 0:
            logger.debug(f"[SECTOR] {etf}: {count} days of data")
        else:
            logger.warning(f"[SECTOR] {etf}: No data - run calibration fetch")

    return availability


async def ensure_sector_etf_data(
    ticker: str,
    db: AsyncSession,
    min_days: int = 750
) -> Optional[pd.DataFrame]:
    """
    Ensure sector ETF data is available for a stock's relative strength calculation.

    If the sector ETF data doesn't exist or is insufficient, fetches it from Yahoo Finance.

    Args:
        ticker: Stock symbol to get sector ETF for
        db: Database session
        min_days: Minimum days of data required

    Returns:
        DataFrame with sector ETF data, or None if unavailable
    """
    from sqlalchemy import func

    # Get the sector ETF for this stock
    sector_etf = await get_sector_etf_for_ticker(ticker, db)

    # Check if we have sufficient data
    result = await db.execute(
        select(func.count()).where(PriceHistory.ticker == sector_etf)
    )
    count = result.scalar()

    if count >= min_days:
        # Data exists, just return it
        return await get_sector_etf_history(sector_etf, db)

    # Need to fetch sector ETF data
    logger.info(f"[SECTOR] Fetching {sector_etf} data for {ticker} RS calculation ({count} days exist, need {min_days})")

    try:
        from . import get_stock_fetcher
        fetcher = get_stock_fetcher()

        # Fetch comprehensive history for the sector ETF
        await fetcher.fetch_comprehensive_history(sector_etf)

        # Verify data was stored
        result = await db.execute(
            select(func.count()).where(PriceHistory.ticker == sector_etf)
        )
        new_count = result.scalar()
        logger.info(f"[SECTOR] Fetched {sector_etf}: {new_count} days now available")

        if new_count >= min_days:
            return await get_sector_etf_history(sector_etf, db)
        else:
            logger.warning(f"[SECTOR] {sector_etf} still has insufficient data after fetch")
            return None

    except Exception as e:
        logger.error(f"[SECTOR] Failed to fetch {sector_etf}: {e}")
        return None
