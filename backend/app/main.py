"""
Stock Portfolio Dashboard API

Main FastAPI application with versioned API endpoints.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
import asyncio
import os

from .config import settings
from .logging_config import setup_logging, get_logger
from .database import get_db, init_db, engine
from .models import Portfolio
from .schemas import (
    PortfolioUpdate, PortfolioResponse, PortfolioWithData,
    HoldingCreate, HoldingUpdate, HoldingResponse,
    StockData
)
from .services import (
    PortfolioService, get_portfolio_service,
    StockFetcher, get_stock_fetcher,
    StockAnalysisService, get_stock_analysis_service
)

# Setup logging
setup_logging()
logger = get_logger(__name__)


async def periodic_cleanup(stock_fetcher: StockFetcher):
    """Run data cleanup every 24 hours."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            daily, intraday = stock_fetcher.cleanup_old_data()
            logger.info(f"Periodic cleanup complete: {daily} daily, {intraday} intraday")
        except Exception as e:
            logger.error(f"Error during periodic cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Run cleanup on startup
    stock_fetcher = get_stock_fetcher()
    try:
        daily, intraday = stock_fetcher.cleanup_old_data()
        if daily or intraday:
            logger.info(f"Startup cleanup: {daily} daily, {intraday} intraday records")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {e}")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup(stock_fetcher))
    
    yield
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup database connection
    await engine.dispose()
    logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    description="API for managing a mock stock portfolio with live market data",
    version=settings.app_version,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API v1 Endpoints
# ============================================================================

# ============ Portfolio Endpoints ============

@app.get("/api/v1/portfolio", response_model=PortfolioWithData)
@app.get("/api/portfolio", response_model=PortfolioWithData, include_in_schema=False)  # Legacy
async def get_portfolio(
    db: AsyncSession = Depends(get_db),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    lite: bool = False
):
    """
    Get the portfolio with holdings and calculated values.
    
    Use `lite=true` for fast load without live stock data.
    """
    return await portfolio_service.get_portfolio(db, lite)


@app.put("/api/v1/portfolio", response_model=PortfolioResponse)
@app.put("/api/portfolio", response_model=PortfolioResponse, include_in_schema=False)  # Legacy
async def update_portfolio(
    portfolio_update: PortfolioUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update portfolio settings (name, total value, chart period, sort options)."""
    from sqlalchemy.orm import selectinload
    
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.holdings)).limit(1)
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        portfolio = Portfolio()
        db.add(portfolio)
    
    if portfolio_update.name is not None:
        portfolio.name = portfolio_update.name
    if portfolio_update.total_value is not None:
        portfolio.total_value = portfolio_update.total_value
    if portfolio_update.chart_period is not None:
        portfolio.chart_period = portfolio_update.chart_period
    if portfolio_update.sort_field is not None:
        portfolio.sort_field = portfolio_update.sort_field
    if portfolio_update.sort_direction is not None:
        portfolio.sort_direction = portfolio_update.sort_direction
    
    await db.commit()
    await db.refresh(portfolio)
    # Reload with holdings relationship for response serialization
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.holdings)).where(Portfolio.id == portfolio.id)
    )
    portfolio = result.scalar_one()
    
    logger.info(f"Updated portfolio: {portfolio.name}, ${portfolio.total_value:,.2f}")
    return portfolio


# ============ Holdings Endpoints ============

@app.post("/api/v1/holdings", response_model=HoldingResponse)
@app.post("/api/holdings", response_model=HoldingResponse, include_in_schema=False)  # Legacy
async def add_holding(
    holding: HoldingCreate,
    db: AsyncSession = Depends(get_db),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Add a stock holding to the portfolio."""
    try:
        return await portfolio_service.add_holding(db, holding)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/v1/holdings/{holding_id}", response_model=HoldingResponse)
@app.put("/api/holdings/{holding_id}", response_model=HoldingResponse, include_in_schema=False)  # Legacy
async def update_holding(
    holding_id: int,
    holding_update: HoldingUpdate,
    db: AsyncSession = Depends(get_db),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Update a holding's allocation or investment info."""
    try:
        return await portfolio_service.update_holding(db, holding_id, holding_update)
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/v1/holdings/{holding_id}")
@app.delete("/api/holdings/{holding_id}", include_in_schema=False)  # Legacy
async def delete_holding(
    holding_id: int,
    db: AsyncSession = Depends(get_db),
    portfolio_service: PortfolioService = Depends(get_portfolio_service)
):
    """Remove a holding from the portfolio."""
    deleted = await portfolio_service.delete_holding(db, holding_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Holding not found")
    return {"message": "Holding deleted successfully"}


# ============ Stock Data Endpoints ============

@app.get("/api/v1/stock/{ticker}", response_model=StockData)
@app.get("/api/stock/{ticker}", response_model=StockData, include_in_schema=False)  # Legacy
async def get_stock(
    ticker: str,
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """Get detailed stock data including price, YTD, SMA(200), and history."""
    data = await stock_fetcher.get_stock_data(ticker)
    
    if data['current_price'] == 0:
        raise HTTPException(status_code=404, detail=f"Stock not found: {ticker}")
    
    return StockData(**data)


@app.get("/api/v1/stock/{ticker}/quote")
@app.get("/api/stock/{ticker}/quote", include_in_schema=False)  # Legacy
async def get_stock_quote(
    ticker: str,
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """Get just the current price data for a single stock (lightweight refresh)."""
    data = await stock_fetcher.get_stock_data(ticker)
    
    if data['current_price'] == 0:
        raise HTTPException(status_code=404, detail=f"Stock not found: {ticker}")
    
    return {
        "ticker": ticker.upper(),
        "current_price": data['current_price'],
        "previous_close": data['previous_close'],
        "change": data['change'],
        "change_pct": data['change_pct'],
        "ytd_return": data['ytd_return'],
        "sma_200": data.get('sma_200'),
        "price_vs_sma": data.get('price_vs_sma'),
    }


@app.get("/api/v1/stock/{ticker}/history")
@app.get("/api/stock/{ticker}/history", include_in_schema=False)  # Legacy
async def get_stock_history(
    ticker: str,
    period: str = "1y",
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Get extended historical data for a stock.
    
    Returns history data, reference_close, and data completeness info.
    """
    result = await stock_fetcher.get_stock_history(ticker, period)
    
    if not result.get("history"):
        raise HTTPException(status_code=404, detail=f"No history found for: {ticker}")
    
    return {
        "ticker": ticker.upper(),
        "history": result["history"],
        "reference_close": result.get("reference_close"),
        "is_complete": result.get("is_complete", True),
        "expected_start": result.get("expected_start"),
        "actual_start": result.get("actual_start")
    }


# ============ Batch Prices Endpoint ============

@app.get("/api/v1/prices")
async def get_batch_prices(
    tickers: str,
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Get current prices for multiple tickers in one efficient batch call.
    
    This is optimized for frequent polling - only fetches price data,
    not historical data or other metadata.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOG")
    
    Returns:
        Dict of ticker -> price data with current_price, change, change_pct, market_state
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    
    if not ticker_list:
        return {}
    
    return await stock_fetcher.get_batch_prices(ticker_list)


# ============ Cache Management ============

@app.delete("/api/v1/stock/{ticker}/cache")
@app.delete("/api/stock/{ticker}/cache", include_in_schema=False)  # Legacy
async def clear_stock_cache(
    ticker: str,
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """Clear the intraday cache for a specific stock."""
    deleted_count = stock_fetcher.clear_ticker_cache(ticker)
    return {
        "message": f"Cleared intraday cache for {ticker.upper()}",
        "deleted_records": deleted_count
    }


# ============ Stock Analysis Endpoints ============

@app.get("/api/v1/stock/{ticker}/analysis")
async def get_stock_analysis(
    ticker: str,
    analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """
    Get comprehensive analysis data for a stock.
    
    Returns fundamentals (ROIC, sector, margins) and options data (call/put ratio, IV).
    """
    return await analysis_service.get_analysis_data(ticker)


@app.get("/api/v1/stock/{ticker}/fundamentals")
async def get_stock_fundamentals(
    ticker: str,
    analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """
    Get fundamental data for a stock.
    
    Returns ROIC, ROE, ROA, sector, industry, profit margins, beta, etc.
    """
    return await analysis_service.get_fundamentals(ticker)


@app.get("/api/v1/stock/{ticker}/options")
async def get_stock_options(
    ticker: str,
    analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """
    Get options data for a stock.
    
    Returns call/put ratio, open interest, implied volatility, options sentiment.
    """
    return await analysis_service.get_options_data(ticker)


@app.get("/api/v1/stock/{ticker}/sector-correlation")
async def get_stock_sector_correlation(
    ticker: str,
    days: int = 60,
    analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """
    Get sector correlation data for a stock.
    
    Returns correlation with sector ETF, beta to sector, and sector/industry info.
    """
    return await analysis_service.get_sector_correlation(ticker, days)


# ============ Health Check ============

@app.get("/api/v1/health")
@app.get("/api/health", include_in_schema=False)  # Legacy
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version
    }
