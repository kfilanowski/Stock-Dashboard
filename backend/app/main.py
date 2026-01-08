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
    StockData,
    OptionHoldingCreate, OptionHoldingUpdate, OptionHoldingResponse, OptionHoldingWithData
)
from .models import OptionHolding
from .services import (
    PortfolioService, get_portfolio_service,
    StockFetcher, get_stock_fetcher,
    StockAnalysisService, get_stock_analysis_service,
    OptionPricingService, get_option_pricing_service,
    OptionAnalyticsService, get_option_analytics_service
)
from .services.option_analytics import OptionPosition

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
    
    # Auto-cleanup intraday data with unfillable gaps
    # This prevents showing charts with holes from previous downtime
    try:
        from .services.price_history import get_price_history_service
        price_history = get_price_history_service()
        cleared = price_history.auto_cleanup_gapped_intraday()
        if cleared:
            logger.info(f"Auto-cleared gapped intraday data for: {list(cleared.keys())}")
    except Exception as e:
        logger.error(f"Error during intraday gap cleanup: {e}")
    
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


# ============ Batch History Endpoint ============

@app.get("/api/v1/history/batch")
async def get_batch_history(
    tickers: str,
    period: str = "1d",
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Get historical data for multiple tickers in one efficient batch call.
    
    This endpoint checks SQLite cache first and only fetches missing data
    from the API. Uses yf.download() for intraday and yahooquery for daily
    periods to batch requests efficiently.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOG")
        period: Time period (1d, 3d, 1w, 1mo, 3mo, 6mo, ytd, 1y, 2y, 5y)
    
    Returns:
        Dict of ticker -> history data with history array, reference_close, is_complete
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    
    if not ticker_list:
        return {}
    
    return await stock_fetcher.batch_get_history(ticker_list, period)


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
    """Clear ALL price history for a specific stock (forces full refresh)."""
    daily_deleted, intraday_deleted = stock_fetcher.clear_ticker_history(ticker)
    return {
        "message": f"Cleared all price history for {ticker.upper()}",
        "daily_deleted": daily_deleted,
        "intraday_deleted": intraday_deleted
    }


@app.delete("/api/v1/cache/intraday")
async def clear_all_intraday_cache(
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Clear ALL intraday chart data for ALL tickers.
    
    Use this when there's a systemic data issue (like a gap affecting all holdings).
    The next chart request for each ticker will fetch fresh data.
    """
    deleted_count = stock_fetcher.clear_all_intraday_cache()
    return {
        "message": "Cleared all intraday cache data",
        "records_deleted": deleted_count
    }


@app.post("/api/v1/cache/cleanup-gaps")
async def cleanup_gapped_data(
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Automatically detect and clear intraday data with unfillable gaps.
    
    This identifies tickers where historical data has gaps that are too old
    to fill from the API (yfinance only keeps ~60 days of 5-minute data).
    Clearing this data allows fresh, gap-free data to be fetched.
    
    This runs automatically on server startup but can be triggered manually.
    """
    cleared = stock_fetcher.auto_cleanup_gapped_data()
    return {
        "message": f"Cleaned up gapped data for {len(cleared)} tickers",
        "cleared_tickers": cleared
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


@app.get("/api/v1/analysis/batch")
async def get_batch_analysis(
    tickers: str,
    analysis_service: StockAnalysisService = Depends(get_stock_analysis_service)
):
    """
    Get analysis data for multiple stocks in one batch call.
    
    Efficiently fetches fundamentals and options data for multiple tickers,
    using yahooquery's batch capabilities and SQLite caching.
    
    Args:
        tickers: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOG")
    
    Returns:
        Dict of ticker -> analysis data (fundamentals + options).
    """
    ticker_list = [t.strip().upper() for t in tickers.split(',') if t.strip()]
    
    if not ticker_list:
        return {}
    
    return await analysis_service.get_batch_analysis(ticker_list)


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


# ============ Option Holdings Endpoints ============

@app.get("/api/v1/options", response_model=list[OptionHoldingWithData])
async def get_option_holdings(
    db: AsyncSession = Depends(get_db),
    option_pricing: OptionPricingService = Depends(get_option_pricing_service),
    option_analytics: OptionAnalyticsService = Depends(get_option_analytics_service)
):
    """
    Get all option holdings with live pricing and analytics.
    
    Returns option positions with current prices, Greeks, breakeven, and P/L.
    """
    from sqlalchemy.orm import selectinload
    
    # Get portfolio with option holdings
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.option_holdings)).limit(1)
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio or not portfolio.option_holdings:
        return []
    
    # Build position list for batch pricing
    positions = [
        {
            "id": oh.id,
            "underlying_ticker": oh.underlying_ticker,
            "expiration_date": oh.expiration_date,
            "option_type": oh.option_type,
            "strike_price": oh.strike_price
        }
        for oh in portfolio.option_holdings
    ]
    
    # Fetch prices in batch
    price_data = await option_pricing.get_batch_option_prices(positions)
    
    # Build response with analytics
    result_list = []
    for oh in portfolio.option_holdings:
        prices = price_data.get(oh.id, {})
        
        # Calculate analytics
        position = OptionPosition(
            underlying_ticker=oh.underlying_ticker,
            option_type=oh.option_type,
            position_type=oh.position_type,
            strike_price=oh.strike_price,
            expiration_date=oh.expiration_date,
            contracts=oh.contracts,
            premium_per_contract=oh.premium_per_contract,
            underlying_price=prices.get("underlying_price"),
            current_option_price=prices.get("current_price"),
            implied_volatility=prices.get("implied_volatility")
        )
        analytics = option_analytics.calculate_position_analytics(position)
        
        # Prefer Greeks from pricing service, fall back to analytics calculation
        pricing_greeks = prices.get("greeks") or {}
        analytics_greeks = analytics.get("greeks") or {}
        
        # Merge Greeks - use pricing service values if available, otherwise analytics
        merged_greeks = {
            "delta": pricing_greeks.get("delta") if pricing_greeks.get("delta") is not None else analytics_greeks.get("delta"),
            "gamma": pricing_greeks.get("gamma") if pricing_greeks.get("gamma") is not None else analytics_greeks.get("gamma"),
            "theta": pricing_greeks.get("theta") if pricing_greeks.get("theta") is not None else analytics_greeks.get("theta"),
            "vega": pricing_greeks.get("vega") if pricing_greeks.get("vega") is not None else analytics_greeks.get("vega"),
            "rho": pricing_greeks.get("rho") if pricing_greeks.get("rho") is not None else analytics_greeks.get("rho"),
        }
        
        result_list.append(OptionHoldingWithData(
            id=oh.id,
            portfolio_id=oh.portfolio_id,
            underlying_ticker=oh.underlying_ticker,
            option_type=oh.option_type,
            position_type=oh.position_type,
            strike_price=oh.strike_price,
            expiration_date=oh.expiration_date,
            contracts=oh.contracts,
            premium_per_contract=oh.premium_per_contract,
            opened_at=oh.opened_at,
            notes=oh.notes,
            # Market data
            underlying_price=prices.get("underlying_price"),
            current_price=prices.get("current_price"),
            bid=prices.get("bid"),
            ask=prices.get("ask"),
            implied_volatility=prices.get("implied_volatility"),
            open_interest=prices.get("open_interest"),
            volume=prices.get("volume"),
            # Position values
            position_value=analytics.get("position_value"),
            cost_basis=analytics.get("cost_basis"),
            gain_loss=analytics.get("gain_loss"),
            gain_loss_pct=analytics.get("gain_loss_pct"),
            # Greeks - merged from both sources
            greeks=merged_greeks,
            # Analytics
            analytics={
                "breakeven_price": analytics.get("breakeven_price"),
                "max_profit": analytics.get("max_profit"),
                "max_loss": analytics.get("max_loss"),
                "profit_probability": analytics.get("profit_probability"),
                "days_to_expiration": analytics.get("days_to_expiration"),
                "is_itm": analytics.get("is_itm"),
                "is_expired": analytics.get("is_expired"),
                "intrinsic_value": analytics.get("intrinsic_value"),
                "time_value": analytics.get("time_value"),
            }
        ))
    
    return result_list


@app.post("/api/v1/options", response_model=OptionHoldingResponse)
async def add_option_holding(
    option: OptionHoldingCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Add an option position to the portfolio.
    
    Provide underlying ticker, strike, expiration, type (call/put),
    position (long/short), contracts, and premium paid/received.
    """
    from sqlalchemy.orm import selectinload
    
    # Get or create portfolio
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.option_holdings)).limit(1)
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        portfolio = Portfolio()
        db.add(portfolio)
        await db.flush()
    
    # Create the option holding
    new_option = OptionHolding(
        portfolio_id=portfolio.id,
        underlying_ticker=option.underlying_ticker.upper(),
        option_type=option.option_type,
        position_type=option.position_type,
        strike_price=option.strike_price,
        expiration_date=option.expiration_date,
        contracts=option.contracts,
        premium_per_contract=option.premium_per_contract,
        notes=option.notes
    )
    
    db.add(new_option)
    await db.commit()
    await db.refresh(new_option)
    
    logger.info(
        f"Added option: {new_option.underlying_ticker} "
        f"{new_option.option_type} ${new_option.strike_price} "
        f"exp {new_option.expiration_date}"
    )
    
    return new_option


@app.put("/api/v1/options/{option_id}", response_model=OptionHoldingResponse)
async def update_option_holding(
    option_id: int,
    option_update: OptionHoldingUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update an option holding (contracts, premium, notes)."""
    result = await db.execute(
        select(OptionHolding).where(OptionHolding.id == option_id)
    )
    option = result.scalar_one_or_none()
    
    if not option:
        raise HTTPException(status_code=404, detail="Option holding not found")
    
    if option_update.contracts is not None:
        option.contracts = option_update.contracts
    if option_update.premium_per_contract is not None:
        option.premium_per_contract = option_update.premium_per_contract
    if option_update.notes is not None:
        option.notes = option_update.notes
    
    await db.commit()
    await db.refresh(option)
    
    logger.info(f"Updated option holding {option_id}")
    return option


@app.delete("/api/v1/options/{option_id}")
async def delete_option_holding(
    option_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Remove an option holding from the portfolio."""
    result = await db.execute(
        select(OptionHolding).where(OptionHolding.id == option_id)
    )
    option = result.scalar_one_or_none()
    
    if not option:
        raise HTTPException(status_code=404, detail="Option holding not found")
    
    await db.delete(option)
    await db.commit()
    
    logger.info(f"Deleted option holding {option_id}")
    return {"message": "Option holding deleted successfully"}


@app.get("/api/v1/options/{option_id}", response_model=OptionHoldingWithData)
async def get_option_holding(
    option_id: int,
    db: AsyncSession = Depends(get_db),
    option_pricing: OptionPricingService = Depends(get_option_pricing_service),
    option_analytics: OptionAnalyticsService = Depends(get_option_analytics_service)
):
    """Get a single option holding with live pricing and analytics."""
    result = await db.execute(
        select(OptionHolding).where(OptionHolding.id == option_id)
    )
    oh = result.scalar_one_or_none()
    
    if not oh:
        raise HTTPException(status_code=404, detail="Option holding not found")
    
    # Fetch live price
    prices = await option_pricing.get_option_price(
        oh.underlying_ticker,
        oh.expiration_date,
        oh.option_type,
        oh.strike_price
    )
    
    # Calculate analytics
    position = OptionPosition(
        underlying_ticker=oh.underlying_ticker,
        option_type=oh.option_type,
        position_type=oh.position_type,
        strike_price=oh.strike_price,
        expiration_date=oh.expiration_date,
        contracts=oh.contracts,
        premium_per_contract=oh.premium_per_contract,
        underlying_price=prices.get("underlying_price"),
        current_option_price=prices.get("current_price"),
        implied_volatility=prices.get("implied_volatility")
    )
    analytics = option_analytics.calculate_position_analytics(position)
    
    # Merge Greeks from both sources
    pricing_greeks = prices.get("greeks") or {}
    analytics_greeks = analytics.get("greeks") or {}
    merged_greeks = {
        "delta": pricing_greeks.get("delta") if pricing_greeks.get("delta") is not None else analytics_greeks.get("delta"),
        "gamma": pricing_greeks.get("gamma") if pricing_greeks.get("gamma") is not None else analytics_greeks.get("gamma"),
        "theta": pricing_greeks.get("theta") if pricing_greeks.get("theta") is not None else analytics_greeks.get("theta"),
        "vega": pricing_greeks.get("vega") if pricing_greeks.get("vega") is not None else analytics_greeks.get("vega"),
        "rho": pricing_greeks.get("rho") if pricing_greeks.get("rho") is not None else analytics_greeks.get("rho"),
    }
    
    return OptionHoldingWithData(
        id=oh.id,
        portfolio_id=oh.portfolio_id,
        underlying_ticker=oh.underlying_ticker,
        option_type=oh.option_type,
        position_type=oh.position_type,
        strike_price=oh.strike_price,
        expiration_date=oh.expiration_date,
        contracts=oh.contracts,
        premium_per_contract=oh.premium_per_contract,
        opened_at=oh.opened_at,
        notes=oh.notes,
        # Market data
        underlying_price=prices.get("underlying_price"),
        current_price=prices.get("current_price"),
        bid=prices.get("bid"),
        ask=prices.get("ask"),
        implied_volatility=prices.get("implied_volatility"),
        open_interest=prices.get("open_interest"),
        volume=prices.get("volume"),
        # Position values
        position_value=analytics.get("position_value"),
        cost_basis=analytics.get("cost_basis"),
        gain_loss=analytics.get("gain_loss"),
        gain_loss_pct=analytics.get("gain_loss_pct"),
        # Greeks - merged from both sources
        greeks=merged_greeks,
        # Analytics
        analytics={
            "breakeven_price": analytics.get("breakeven_price"),
            "max_profit": analytics.get("max_profit"),
            "max_loss": analytics.get("max_loss"),
            "profit_probability": analytics.get("profit_probability"),
            "days_to_expiration": analytics.get("days_to_expiration"),
            "is_itm": analytics.get("is_itm"),
            "is_expired": analytics.get("is_expired"),
            "intrinsic_value": analytics.get("intrinsic_value"),
            "time_value": analytics.get("time_value"),
        }
    )


# ============ Option Chain Discovery Endpoints ============

@app.get("/api/v1/options/chain/{ticker}/expirations")
async def get_option_expirations(
    ticker: str,
    option_pricing: OptionPricingService = Depends(get_option_pricing_service)
):
    """
    Get available expiration dates for a ticker's options.
    
    Use this to populate the expiration date picker when adding an option.
    """
    expirations = await option_pricing.get_available_expirations(ticker)
    return {"ticker": ticker.upper(), "expirations": expirations}


@app.get("/api/v1/options/chain/{ticker}/strikes")
async def get_option_strikes(
    ticker: str,
    expiration: str,
    option_type: str = None,
    option_pricing: OptionPricingService = Depends(get_option_pricing_service)
):
    """
    Get available strike prices for a ticker and expiration.
    
    Use this to populate the strike price picker when adding an option.
    """
    strikes = await option_pricing.get_strikes_for_expiration(ticker, expiration, option_type)
    return {
        "ticker": ticker.upper(),
        "expiration": expiration,
        "strikes": strikes
    }


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
