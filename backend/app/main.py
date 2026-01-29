"""
Stock Portfolio Dashboard API

Main FastAPI application with versioned API endpoints.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
import math
import os

from .config import settings
from .logging_config import setup_logging, get_logger
from .database import get_db, init_db, engine
from .models import Portfolio, CalibrationWeights, CalibrationWindow, CalibrationTrade, StockAnalysisCache, PriceHistory, IntradayPriceHistory
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
from .services.calibration_service import (
    calibrate_ticker,
    calibrate_ticker_streaming,
    load_calibrated_weights,
    save_calibration_result,
    CalibrationProgress,
    CalibrationErrorResult
)
from .services.wfo_optimizer import DEFAULT_WEIGHTS, InsufficientVolatilityError, OptimizerType, DEFAULT_OPTIMIZER
from .services.option_analytics import OptionPosition

# Setup logging
setup_logging()
logger = get_logger(__name__)


def sanitize_float(value: float) -> float | None:
    """Convert nan/inf to None for JSON serialization."""
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def sanitize_weights(weights: dict) -> dict:
    """Sanitize all weight values for JSON serialization."""
    if weights is None:
        return None
    return {k: sanitize_float(v) if isinstance(v, float) else v for k, v in weights.items()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
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
    
    yield
    
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


# ============ Portfolio Export/Import ============

@app.get("/api/v1/portfolio/export")
async def export_portfolio(db: AsyncSession = Depends(get_db)):
    """
    Export portfolio holdings (stocks and options) to JSON for backup.

    Use this before database migrations or schema changes.
    Returns a downloadable JSON file.
    """
    from sqlalchemy.orm import selectinload
    from fastapi.responses import JSONResponse

    result = await db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.holdings))
        .options(selectinload(Portfolio.option_holdings))
        .limit(1)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        return JSONResponse(
            content={"error": "No portfolio found"},
            status_code=404
        )

    export_data = {
        "exported_at": datetime.utcnow().isoformat(),
        "version": "1.0",
        "portfolio": {
            "name": portfolio.name,
            "total_value": portfolio.total_value,
            "chart_period": portfolio.chart_period,
            "sort_field": portfolio.sort_field,
            "sort_direction": portfolio.sort_direction,
        },
        "holdings": [
            {
                "ticker": h.ticker,
                "shares": h.shares,
                "avg_cost": h.avg_cost,
                "is_pinned": h.is_pinned,
            }
            for h in portfolio.holdings
        ],
        "option_holdings": [
            {
                "underlying_ticker": o.underlying_ticker,
                "option_type": o.option_type,
                "position_type": o.position_type,
                "strike_price": o.strike_price,
                "expiration_date": o.expiration_date.isoformat() if o.expiration_date else None,
                "contracts": o.contracts,
                "premium_per_contract": o.premium_per_contract,
                "notes": o.notes,
            }
            for o in portfolio.option_holdings
        ],
    }

    logger.info(f"Exported portfolio: {len(export_data['holdings'])} holdings, {len(export_data['option_holdings'])} options")

    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f"attachment; filename=portfolio_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        }
    )


@app.post("/api/v1/portfolio/import")
async def import_portfolio(
    import_data: dict,
    db: AsyncSession = Depends(get_db),
    clear_existing: bool = True
):
    """
    Import portfolio holdings from a previously exported JSON backup.

    Args:
        import_data: The exported JSON data
        clear_existing: If True, clears existing holdings before import (default: True)

    Returns:
        Summary of imported holdings
    """
    from sqlalchemy.orm import selectinload
    from datetime import date

    # Validate import data
    if "holdings" not in import_data and "option_holdings" not in import_data:
        raise HTTPException(status_code=400, detail="Invalid import data: missing holdings")

    # Get or create portfolio
    result = await db.execute(
        select(Portfolio)
        .options(selectinload(Portfolio.holdings))
        .options(selectinload(Portfolio.option_holdings))
        .limit(1)
    )
    portfolio = result.scalar_one_or_none()

    if not portfolio:
        portfolio = Portfolio()
        db.add(portfolio)
        await db.flush()

    # Optionally restore portfolio settings
    if "portfolio" in import_data:
        p_data = import_data["portfolio"]
        if "name" in p_data:
            portfolio.name = p_data["name"]
        if "total_value" in p_data:
            portfolio.total_value = p_data["total_value"]
        if "chart_period" in p_data:
            portfolio.chart_period = p_data["chart_period"]
        if "sort_field" in p_data:
            portfolio.sort_field = p_data["sort_field"]
        if "sort_direction" in p_data:
            portfolio.sort_direction = p_data["sort_direction"]

    # Clear existing holdings if requested
    if clear_existing:
        for h in list(portfolio.holdings):
            await db.delete(h)
        for o in list(portfolio.option_holdings):
            await db.delete(o)
        await db.flush()

    # Import stock holdings
    holdings_imported = 0
    for h_data in import_data.get("holdings", []):
        holding = Holding(
            portfolio_id=portfolio.id,
            ticker=h_data["ticker"].upper(),
            shares=h_data.get("shares", 0),
            avg_cost=h_data.get("avg_cost"),
            is_pinned=h_data.get("is_pinned", False),
        )
        db.add(holding)
        holdings_imported += 1

    # Import option holdings
    options_imported = 0
    for o_data in import_data.get("option_holdings", []):
        exp_date = o_data.get("expiration_date")
        if isinstance(exp_date, str):
            exp_date = date.fromisoformat(exp_date)

        option = OptionHolding(
            portfolio_id=portfolio.id,
            underlying_ticker=o_data["underlying_ticker"].upper(),
            option_type=o_data["option_type"],
            position_type=o_data["position_type"],
            strike_price=o_data["strike_price"],
            expiration_date=exp_date,
            contracts=o_data.get("contracts", 1),
            premium_per_contract=o_data.get("premium_per_contract"),
            notes=o_data.get("notes"),
        )
        db.add(option)
        options_imported += 1

    await db.commit()

    logger.info(f"Imported portfolio: {holdings_imported} holdings, {options_imported} options")

    return {
        "status": "success",
        "holdings_imported": holdings_imported,
        "options_imported": options_imported,
        "cleared_existing": clear_existing,
    }


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


# ============ Calibration Endpoints (Walk-Forward Optimization) ============

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional


class CalibrationRequest(BaseModel):
    """Request to start calibration for a ticker."""
    ticker: str
    horizons: Optional[List[int]] = None  # If None and auto_discover=True, uses discovered horizons
    strategy_classes: Optional[List[str]] = None  # If None, uses all: ['directional', 'premium_sell', 'premium_buy']
    auto_discover_horizons: bool = True   # Auto-discover optimal horizons via IC analysis
    min_horizon: int = 2                  # Min horizon for discovery
    max_horizon: int = 30                 # Max horizon for discovery
    optimizer: str = 'coordinate_descent' # 'coordinate_descent', 'differential_evolution', or 'hybrid'


class WeightsResponse(BaseModel):
    """Response containing calibrated weights."""
    ticker: str
    horizon: int
    weights: dict
    calibrated_at: Optional[str] = None
    is_default: bool = False


@app.post("/api/v1/calibration/start")
async def start_calibration(
    request: CalibrationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start Walk-Forward Optimization calibration for a ticker.

    If auto_discover_horizons=True (default), first runs resonance analysis to find
    the stock's optimal trading horizons based on Information Coefficient.

    Calibrates for each strategy class:
    - directional: For buyShares, sellShares
    - premium_sell: For openCSP, openCC
    - premium_buy: For buyCall, buyPut

    This is an isolated operation that doesn't affect the main dashboard.
    Returns immediately with a job ID; use the SSE stream to monitor progress.
    """
    from .services.horizon_resonance import analyze_resonance
    from .services.calibration_service import load_price_history, DEFAULT_STRATEGY_CLASSES

    ticker = request.ticker.upper()

    # Determine which strategy classes to use
    strategy_classes = request.strategy_classes or DEFAULT_STRATEGY_CLASSES

    # Determine which horizons to use
    horizons = request.horizons
    resonance_result = None

    if request.auto_discover_horizons and horizons is None:
        # Auto-discover optimal horizons using IC analysis
        logger.info(f"[WFO] Auto-discovering optimal horizons for {ticker}...")
        try:
            df = await load_price_history(db, ticker, min_days=252)
            resonance_result = analyze_resonance(
                df=df,
                ticker=ticker,
                min_horizon=request.min_horizon,
                max_horizon=request.max_horizon
            )

            # Use recommended horizons from resonance analysis
            # Pick best from short and medium/long ranges for diversity
            discovered = []
            if resonance_result.recommended_short:
                discovered.append(resonance_result.recommended_short)
            if resonance_result.recommended_medium:
                discovered.append(resonance_result.recommended_medium)
            elif resonance_result.recommended_long:
                discovered.append(resonance_result.recommended_long)

            # Fallback to top horizons if recommendations are empty
            if not discovered:
                discovered = resonance_result.top_horizons[:2]

            # Final fallback to defaults
            horizons = discovered if discovered else [5, 21]

            logger.info(f"[WFO] Discovered optimal horizons for {ticker}: {horizons}")
            logger.info(f"[WFO] Resonance: top={resonance_result.top_horizons}, "
                       f"short={resonance_result.recommended_short}, "
                       f"medium={resonance_result.recommended_medium}, "
                       f"long={resonance_result.recommended_long}")

        except Exception as e:
            logger.warning(f"[WFO] Resonance discovery failed for {ticker}: {e}, using defaults")
            horizons = [5, 21]  # Fallback defaults (better than 3, 15)
    elif horizons is None:
        # No auto-discover, no explicit horizons - use improved defaults
        horizons = [5, 21]

    # Parse optimizer type
    try:
        optimizer_type = OptimizerType(request.optimizer)
    except ValueError:
        optimizer_type = DEFAULT_OPTIMIZER
        logger.warning(f"[WFO] Unknown optimizer '{request.optimizer}', using default")

    logger.info(f"[WFO] Starting calibration for {ticker}, horizons={horizons}, strategies={strategy_classes}, optimizer={optimizer_type.value}")

    try:
        # Run calibration with discovered/specified horizons and strategy classes
        # New return structure: {strategy_class: {horizon: result}}
        results = await calibrate_ticker(
            db=db,
            ticker=ticker,
            horizons=horizons,
            strategy_classes=strategy_classes,
            optimizer=optimizer_type
        )

        # Save results to database
        saved_count = 0
        response_results = {}

        for strategy_class, horizon_results in results.items():
            response_results[strategy_class] = {}

            for horizon, result in horizon_results.items():
                # Check if result is a successful optimization or an error
                if isinstance(result, CalibrationErrorResult):
                    # Calibration failed with detailed error info
                    logger.warning(f"[WFO] {ticker} {strategy_class} horizon={horizon}: {result.error_type} - {result.trades_found} trades in {result.window_days} days")
                    response_results[strategy_class][horizon] = {
                        "sqn": None,
                        "trades": result.trades_found,
                        "weights": None,
                        "reduced_confidence": False,
                        "error": f"Insufficient trades: Only {result.trades_found} signals generated over {result.window_days} days (need 20+). This stock may be too stable for WFO calibration - it doesn't trigger enough buy/sell signals with the current thresholds."
                    }
                elif result is not None:
                    # Successful optimization
                    logger.info(f"[WFO] {ticker} {strategy_class} horizon={horizon}: SQN={result.train_sqn:.3f}, trades={result.total_trades}")
                    logger.info(f"[WFO] {ticker} {strategy_class} horizon={horizon} weights: {result.weights}")

                    # Save to database (including window results and trades if available)
                    save_error = None
                    try:
                        window_results = getattr(result, 'window_results', None)
                        await save_calibration_result(
                            db, ticker, result,
                            window_results=window_results,
                            strategy_class=strategy_class
                        )
                        saved_count += 1
                        windows_saved = len(window_results) if window_results else 0
                        logger.info(f"[WFO] Saved calibration for {ticker} {strategy_class} horizon={horizon} to database ({windows_saved} windows)")
                    except Exception as save_err:
                        save_error = str(save_err)
                        logger.error(f"[WFO] Failed to save {ticker} {strategy_class} horizon={horizon}: {save_err}")

                    response_results[strategy_class][horizon] = {
                        "sqn": sanitize_float(result.train_sqn),
                        "gross_sqn": sanitize_float(result.avg_gross_sqn) if result.avg_gross_sqn else None,
                        "trades": result.total_trades,
                        "weights": sanitize_weights(result.weights),
                        "reduced_confidence": result.reduced_confidence,
                        "error": None,
                        "save_error": save_error  # Surface database save failures to frontend
                    }
                else:
                    # Fallback for None results (shouldn't happen anymore)
                    logger.warning(f"[WFO] {ticker} {strategy_class} horizon={horizon}: No result (unknown error)")
                    response_results[strategy_class][horizon] = {
                        "sqn": None,
                        "trades": 0,
                        "weights": None,
                        "reduced_confidence": False,
                        "error": "Calibration failed for unknown reason"
                    }

        total_results = sum(len(hr) for hr in results.values())
        logger.info(f"[WFO] Calibration complete for {ticker}: saved {saved_count}/{total_results} strategy+horizon combinations")

        # Check for save failures
        save_warnings = []
        if saved_count < total_results:
            save_warnings.append(f"Only {saved_count}/{total_results} results saved to database")

        # Build response with resonance info if available
        response = {
            "status": "complete",
            "ticker": ticker,
            "horizons": horizons,  # Actual horizons used (may be discovered)
            "strategy_classes": strategy_classes,
            "horizons_auto_discovered": resonance_result is not None,
            "saved_count": saved_count,
            "total_results": total_results,
            "save_warnings": save_warnings if save_warnings else None,
            "results": response_results
        }

        # Include resonance analysis details if we discovered horizons
        if resonance_result:
            response["resonance"] = {
                "top_horizons": resonance_result.top_horizons,
                "recommended_short": resonance_result.recommended_short,
                "recommended_medium": resonance_result.recommended_medium,
                "recommended_long": resonance_result.recommended_long,
                "best_ic": float(resonance_result.heatmap[resonance_result.top_horizons[0]].ic)
                    if resonance_result.top_horizons else None
            }

        return response
    except InsufficientVolatilityError as e:
        logger.warning(f"[WFO] {ticker}: Insufficient volatility - {e}")
        return {
            "status": "error",
            "ticker": ticker,
            "error": str(e),
            "error_code": "INSUFFICIENT_VOLATILITY"
        }
    except ValueError as e:
        logger.warning(f"[WFO] {ticker}: Value error - {e}")
        return {
            "status": "error",
            "ticker": ticker,
            "error": str(e),
            "error_code": "INSUFFICIENT_DATA"
        }
    except Exception as e:
        logger.error(f"[WFO] Calibration failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/calibration/stream/{ticker}")
async def stream_calibration(
    ticker: str,
    horizons: Optional[str] = None,  # Comma-separated list of horizons, e.g., "6,7,8"
    db: AsyncSession = Depends(get_db)
):
    """
    Stream calibration progress via Server-Sent Events (SSE).

    Events are formatted as:
    data: {"ticker": "AAPL", "stage": "optimizing", "progress": 50, ...}

    Stages: loading, optimizing, testing, saving, complete, error

    Args:
        ticker: Stock ticker
        horizons: Comma-separated list of horizons (e.g., "6,7,8"). If not provided, uses defaults.
    """
    ticker = ticker.upper()

    # Parse horizons from query string
    horizon_list = None
    if horizons:
        try:
            horizon_list = [int(h.strip()) for h in horizons.split(',') if h.strip()]
            logger.info(f"[SSE] Streaming calibration for {ticker} with horizons: {horizon_list}")
        except ValueError:
            logger.warning(f"[SSE] Invalid horizons format: {horizons}, using defaults")

    async def event_generator():
        try:
            async for event in calibrate_ticker_streaming(db, ticker, horizons=horizon_list):
                yield event
        except Exception as e:
            error_event = CalibrationProgress(
                ticker=ticker,
                horizon=0,
                stage="error",
                progress=0,
                message=str(e)
            )
            yield error_event.to_sse()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@app.get("/api/v1/calibration/weights/batch")
async def get_calibration_weights_batch(
    tickers: str,  # Comma-separated list
    horizon: int = 3,
    strategy_class: str = 'all',  # 'all', 'directional', 'premium_sell', 'premium_buy'
    db: AsyncSession = Depends(get_db)
):
    """
    Get calibrated weights for multiple tickers.

    Useful for loading weights for entire portfolio at once.

    Args:
        tickers: Comma-separated list of ticker symbols
        horizon: Trading horizon (3 or 15)
        strategy_class: Strategy class to load weights for ('all', 'directional', 'premium_sell', 'premium_buy')

    NOTE: This endpoint must be defined BEFORE /{ticker} to avoid path matching issues.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    results = {}
    for ticker in ticker_list:
        weights = await load_calibrated_weights(db, ticker, horizon, strategy_class)

        # Get metadata directly from CalibrationWeights (not CalibrationWindow which may be empty)
        weights_meta = await db.execute(
            select(
                func.avg(CalibrationWeights.sqn_score),
                func.max(CalibrationWeights.updated_at)
            )
            .where(CalibrationWeights.ticker == ticker)
            .where(CalibrationWeights.horizon == horizon)
            .where(CalibrationWeights.strategy_class == strategy_class)
        )
        meta = weights_meta.one()
        avg_sqn, last_updated = meta

        # Handle -inf/inf/nan values which are not JSON compliant
        sqn_value = None
        if avg_sqn is not None and weights:
            if math.isfinite(avg_sqn):
                sqn_value = float(avg_sqn)
            # else: leave as None for -inf/inf/nan

        results[ticker] = {
            "weights": weights or DEFAULT_WEIGHTS,
            "is_default": weights is None,
            "sqn": sqn_value,
            "updated_at": last_updated.isoformat() if last_updated and weights else None
        }

    return {
        "horizon": horizon,
        "strategy_class": strategy_class,
        "tickers": results
    }


@app.get("/api/v1/calibration/weights/{ticker}")
async def get_calibration_weights(
    ticker: str,
    horizon: int = 3,
    strategy_class: str = 'directional',  # Default to 'directional' for backward compat
    db: AsyncSession = Depends(get_db)
):
    """
    Get calibrated weights for a ticker, horizon, and strategy class.

    Strategy classes:
    - 'directional': For buyShares, sellShares (default)
    - 'premium_sell': For openCSP, openCC
    - 'premium_buy': For buyCall, buyPut

    Returns default weights if no calibration exists.
    Falls back to 'all' strategy if specific strategy not found.
    This is what the frontend should call to get weights for scoring.
    """
    ticker = ticker.upper()

    weights = await load_calibrated_weights(db, ticker, horizon, strategy_class)

    if weights:
        return WeightsResponse(
            ticker=ticker,
            horizon=horizon,
            weights=weights,
            is_default=False
        )
    else:
        return WeightsResponse(
            ticker=ticker,
            horizon=horizon,
            weights=DEFAULT_WEIGHTS,
            is_default=True
        )


@app.get("/api/v1/calibration/defaults")
async def get_default_weights():
    """
    Get default indicator weights.
    
    These are used when no calibration exists for a ticker.
    """
    return {
        "weights": DEFAULT_WEIGHTS,
        "description": "Default weights (all indicators weighted 1.0)"
    }


@app.get("/api/v1/calibration/tickers")
async def get_calibration_tickers(
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of portfolio tickers for the calibration picker.
    """
    from sqlalchemy.orm import selectinload
    
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.holdings)).limit(1)
    )
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        return {"tickers": []}
    
    tickers = [h.ticker.upper() for h in portfolio.holdings]
    return {"tickers": sorted(tickers)}


@app.get("/api/v1/calibration/verify/{ticker}")
async def verify_calibration(
    ticker: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Verify calibration data exists in database for a ticker.
    
    Returns detailed info about what's stored.
    """
    from sqlalchemy import func
    
    ticker = ticker.upper()
    
    # Count weights
    weights_result = await db.execute(
        select(func.count(CalibrationWeights.id))
        .where(CalibrationWeights.ticker == ticker)
    )
    weights_count = weights_result.scalar() or 0
    
    # Get weight details
    weights_data = await db.execute(
        select(CalibrationWeights)
        .where(CalibrationWeights.ticker == ticker)
        .order_by(CalibrationWeights.horizon, CalibrationWeights.strategy_class, CalibrationWeights.indicator)
    )
    
    def safe_sqn(val):
        """Convert -inf/inf/nan to JSON-safe values"""
        if val is None:
            return None
        import math
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    
    weights_list = [
        {
            "indicator": w.indicator,
            "horizon": w.horizon,
            "strategy_class": w.strategy_class or "all",
            "weight": w.weight,
            "sqn_score": safe_sqn(w.sqn_score),
            "gross_sqn": safe_sqn(w.avg_gross_sqn),
            "stability_passed": w.stability_passed,
            "updated_at": str(w.updated_at) if w.updated_at else None
        }
        for w in weights_data.scalars().all()
    ]
    
    # Count windows
    windows_result = await db.execute(
        select(func.count(CalibrationWindow.id))
        .where(CalibrationWindow.ticker == ticker)
    )
    windows_count = windows_result.scalar() or 0
    
    # Count trades
    trades_result = await db.execute(
        select(func.count(CalibrationTrade.id))
        .where(CalibrationTrade.ticker == ticker)
    )
    trades_count = trades_result.scalar() or 0
    
    return {
        "ticker": ticker,
        "verified": weights_count > 0,
        "weights_count": weights_count,
        "windows_count": windows_count,
        "trades_count": trades_count,
        "weights": weights_list,
        "message": (
            f"Found {weights_count} weights, {windows_count} windows, {trades_count} trades"
            if weights_count > 0
            else "No calibration data found"
        )
    }


@app.get("/api/v1/calibration/diagnostic/{ticker}")
async def run_calibration_diagnostic(
    ticker: str,
    horizon: int = 5,
    db: AsyncSession = Depends(get_db)
):
    """
    Run comprehensive diagnostic to establish ground truth.

    This helps answer: "Are the signals predictive, or are costs killing us?"

    Returns:
    - Buy-and-hold baseline (what did the market do?)
    - Gross vs Net SQN (signals before/after costs)
    - Random baseline (are signals better than random?)
    - Sample trades (for manual inspection)
    - Verdict and recommendations
    """
    from .services.calibration_diagnostic import run_diagnostic, diagnostic_to_dict

    try:
        report = await run_diagnostic(db, ticker, horizon)
        return diagnostic_to_dict(report)
    except ValueError as e:
        return {"error": str(e), "ticker": ticker.upper()}
    except Exception as e:
        logger.error(f"[DIAGNOSTIC] Failed for {ticker}: {e}", exc_info=True)
        return {"error": str(e), "ticker": ticker.upper()}


@app.get("/api/v1/calibration/data-status/{ticker}")
async def get_calibration_data_status(
    ticker: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Check how much historical data is available for a ticker.

    WFO requires ~750 trading days (3 years) for swing trading calibration.
    More data = more trades = better statistical significance.
    """
    from sqlalchemy import func
    from .models import PriceHistory

    ticker = ticker.upper()

    # Count available daily records
    result = await db.execute(
        select(
            func.count(PriceHistory.id),
            func.min(PriceHistory.date),
            func.max(PriceHistory.date)
        ).where(PriceHistory.ticker == ticker)
    )
    row = result.one()
    count, min_date, max_date = row

    # WFO optimally needs ~1260 days (5 years), but we accept 750 (3 years)
    # as a minimum to allow newer stocks to be calibrated.
    required_days = 750
    has_sufficient_data = count >= required_days

    return {
        "ticker": ticker,
        "available_days": count,
        "required_days": required_days,
        "has_sufficient_data": has_sufficient_data,
        "earliest_date": str(min_date) if min_date else None,
        "latest_date": str(max_date) if max_date else None,
        "message": (
            f"Ready for calibration ({count} days available)"
            if has_sufficient_data
            else f"Need {required_days - count} more days of data"
        )
    }


@app.get("/api/v1/calibration/resonance/{ticker}")
async def analyze_horizon_resonance(
    ticker: str,
    min_horizon: int = 2,
    max_horizon: int = 30,
    db: AsyncSession = Depends(get_db)
):
    """
    Discover optimal trading horizons for a stock using Information Coefficient analysis.

    Instead of hard-coding horizons (3d, 15d), this finds the "resonant frequency"
    for each stock by testing multiple horizons and measuring signal-return correlation.

    Different stocks have different natural trading cycles:
    - TSLA might resonate at 5-day volatility cycles
    - KO might resonate at 20-day trend cycles

    Returns:
        - heatmap: IC values for each horizon (2-30 days)
        - top_horizons: Best horizons sorted by predictive power
        - recommended: Best horizon in each range (short/medium/long)
    """
    from .services.horizon_resonance import analyze_resonance, format_heatmap_for_display
    from .services.calibration_service import load_price_history

    ticker = ticker.upper()
    logger.info(f"[RESONANCE] Starting horizon analysis for {ticker}")

    try:
        # Load price history
        df = await load_price_history(db, ticker, min_days=252)  # Need at least 1 year

        # Run resonance analysis
        result = analyze_resonance(
            df=df,
            ticker=ticker,
            min_horizon=min_horizon,
            max_horizon=max_horizon
        )

        # Format for response
        heatmap_data = format_heatmap_for_display(result)

        return {
            "status": "success",
            "ticker": ticker,
            "horizons_tested": result.horizons_tested,
            "top_horizons": result.top_horizons,
            "recommended": {
                "short": result.recommended_short,
                "medium": result.recommended_medium,
                "long": result.recommended_long
            },
            "heatmap": heatmap_data,
            "interpretation": _interpret_resonance(result)
        }

    except ValueError as e:
        return {
            "status": "error",
            "ticker": ticker,
            "error": str(e),
            "message": "Insufficient data for resonance analysis"
        }
    except Exception as e:
        logger.error(f"[RESONANCE] Analysis failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_resonance(result) -> str:
    """Generate human-readable interpretation of resonance results."""
    from .services.horizon_resonance import ResonanceResult

    significant_count = sum(1 for r in result.heatmap.values() if r.is_significant)

    if significant_count == 0:
        return (
            f"{result.ticker} shows weak signal-return correlation across all horizons. "
            "Technical indicators may have limited predictive power for this stock. "
            "Consider using default horizons (5d, 21d) or fundamental analysis."
        )

    best_horizon = result.top_horizons[0] if result.top_horizons else None
    best_ic = result.heatmap[best_horizon].ic if best_horizon else 0

    if best_horizon and best_horizon <= 7:
        style = "swing trading (mean reversion)"
    elif best_horizon and best_horizon <= 15:
        style = "short-term trend following"
    else:
        style = "position trading (longer trends)"

    return (
        f"{result.ticker} resonates best at {best_horizon}-day horizon (IC={best_ic:.3f}). "
        f"This suggests {style} may be most effective. "
        f"Found {significant_count} statistically significant horizons."
    )


@app.post("/api/v1/calibration/fetch-data/{ticker}")
async def fetch_calibration_data(
    ticker: str,
    stock_fetcher: StockFetcher = Depends(get_stock_fetcher)
):
    """
    Fetch and cache comprehensive price history for WFO calibration.
    
    This performs a deep fetch of:
    - 5 years of Daily data (no downsampling)
    - 2 years of Hourly data
    - 60 days of 5-minute data
    - 7 days of 1-minute data
    
    The data is stored in SQLite for future use.
    Returns the daily history to the frontend.
    """
    ticker = ticker.upper()
    
    try:
        # Fetch comprehensive data
        result = await stock_fetcher.fetch_comprehensive_history(ticker)
        
        history = result.get('history', [])
        
        if not history:
            return {
                "status": "error",
                "ticker": ticker,
                "message": "No data returned from Yahoo Finance",
                "days_fetched": 0
            }
        
        return {
            "status": "success",
            "ticker": ticker,
            "days_fetched": len(history),
            "earliest_date": history[0].get('date') if history else None,
            "latest_date": history[-1].get('date') if history else None,
            "message": f"Successfully cached {len(history)} days of history"
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch calibration data for {ticker}: {e}")
        return {
            "status": "error",
            "ticker": ticker,
            "message": str(e),
            "days_fetched": 0
        }


# ============================================================================
# Admin / Cache Management Endpoints
# ============================================================================

@app.delete("/api/v1/admin/clear/analysis-cache")
async def clear_analysis_cache(
    ticker: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear the stock analysis cache (fundamentals + options data).
    If ticker is provided, only clear that ticker. Otherwise clear all.
    """
    try:
        if ticker:
            ticker = ticker.upper()
            result = await db.execute(
                delete(StockAnalysisCache).where(StockAnalysisCache.ticker == ticker)
            )
            count = result.rowcount
            await db.commit()
            return {"status": "success", "message": f"Cleared analysis cache for {ticker}", "rows_deleted": count}
        else:
            result = await db.execute(delete(StockAnalysisCache))
            count = result.rowcount
            await db.commit()
            return {"status": "success", "message": "Cleared all analysis cache", "rows_deleted": count}
    except Exception as e:
        logger.error(f"Failed to clear analysis cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/admin/clear/calibration-weights")
async def clear_calibration_weights(
    ticker: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Clear WFO calibration weights.
    If ticker is provided, only clear that ticker. Otherwise clear all.
    Also clears related CalibrationWindow and CalibrationTrade records.
    """
    try:
        if ticker:
            ticker = ticker.upper()
            # Clear weights
            r1 = await db.execute(
                delete(CalibrationWeights).where(CalibrationWeights.ticker == ticker)
            )
            # Clear windows
            r2 = await db.execute(
                delete(CalibrationWindow).where(CalibrationWindow.ticker == ticker)
            )
            # Clear trades
            r3 = await db.execute(
                delete(CalibrationTrade).where(CalibrationTrade.ticker == ticker)
            )
            await db.commit()
            total = r1.rowcount + r2.rowcount + r3.rowcount
            return {"status": "success", "message": f"Cleared calibration data for {ticker}", "rows_deleted": total}
        else:
            r1 = await db.execute(delete(CalibrationWeights))
            r2 = await db.execute(delete(CalibrationWindow))
            r3 = await db.execute(delete(CalibrationTrade))
            await db.commit()
            total = r1.rowcount + r2.rowcount + r3.rowcount
            return {"status": "success", "message": "Cleared all calibration data", "rows_deleted": total}
    except Exception as e:
        logger.error(f"Failed to clear calibration weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/admin/clear/price-history")
async def clear_price_history(
    ticker: Optional[str] = None,
    history_type: str = "all",  # "daily", "intraday", or "all"
    db: AsyncSession = Depends(get_db)
):
    """
    Clear price history cache.
    history_type: "daily", "intraday", or "all"
    If ticker is provided, only clear that ticker.
    """
    try:
        total = 0
        ticker_upper = ticker.upper() if ticker else None

        if history_type in ("daily", "all"):
            if ticker_upper:
                r = await db.execute(
                    delete(PriceHistory).where(PriceHistory.ticker == ticker_upper)
                )
            else:
                r = await db.execute(delete(PriceHistory))
            total += r.rowcount

        if history_type in ("intraday", "all"):
            if ticker_upper:
                r = await db.execute(
                    delete(IntradayPriceHistory).where(IntradayPriceHistory.ticker == ticker_upper)
                )
            else:
                r = await db.execute(delete(IntradayPriceHistory))
            total += r.rowcount

        await db.commit()
        scope = f"for {ticker_upper}" if ticker_upper else "all"
        return {"status": "success", "message": f"Cleared {history_type} price history {scope}", "rows_deleted": total}
    except Exception as e:
        logger.error(f"Failed to clear price history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/cache-stats")
async def get_cache_stats(db: AsyncSession = Depends(get_db)):
    """
    Get statistics about cached data.
    """
    try:
        from sqlalchemy import func

        # Count records in each table
        analysis_count = (await db.execute(select(func.count(StockAnalysisCache.id)))).scalar() or 0
        weights_count = (await db.execute(select(func.count(CalibrationWeights.id)))).scalar() or 0
        windows_count = (await db.execute(select(func.count(CalibrationWindow.id)))).scalar() or 0
        daily_count = (await db.execute(select(func.count(PriceHistory.id)))).scalar() or 0
        intraday_count = (await db.execute(select(func.count(IntradayPriceHistory.id)))).scalar() or 0

        # Get unique tickers with calibration
        calibrated_tickers = (await db.execute(
            select(func.count(func.distinct(CalibrationWeights.ticker)))
        )).scalar() or 0

        return {
            "analysis_cache": analysis_count,
            "calibration_weights": weights_count,
            "calibration_windows": windows_count,
            "calibrated_tickers": calibrated_tickers,
            "daily_price_history": daily_count,
            "intraday_price_history": intraday_count
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
