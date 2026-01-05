from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from contextlib import asynccontextmanager
from typing import Optional
import os

from .database import get_db, init_db, engine
from .models import Portfolio, Holding, PortfolioSnapshot
from .schemas import (
    PortfolioCreate, PortfolioUpdate, PortfolioResponse, PortfolioWithData,
    HoldingCreate, HoldingUpdate, HoldingResponse, HoldingWithData,
    StockData, PortfolioHistory, PortfolioHistoryPoint
)
from .stock_service import get_stock_data, get_multiple_stocks, get_stock_history, validate_ticker, cleanup_old_data, clear_intraday_cache
import asyncio


async def periodic_cleanup():
    """Run data cleanup every 24 hours."""
    while True:
        await asyncio.sleep(86400)  # 24 hours
        try:
            cleanup_old_data()
        except Exception as e:
            print(f"Error during periodic cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    # Initialize database
    await init_db()
    
    # Run cleanup on startup
    try:
        cleanup_old_data()
    except Exception as e:
        print(f"Error during startup cleanup: {e}")
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Cleanup
    await engine.dispose()


app = FastAPI(
    title="Stock Portfolio Dashboard API",
    description="API for managing a mock stock portfolio with live market data",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Portfolio Endpoints ============

@app.get("/api/portfolio", response_model=PortfolioWithData)
async def get_portfolio(db: AsyncSession = Depends(get_db), lite: bool = False):
    """Get the portfolio. Use lite=true for fast load without live stock data."""
    result = await db.execute(
        select(Portfolio).options(selectinload(Portfolio.holdings)).limit(1)
    )
    portfolio = result.scalar_one_or_none()
    
    # Create default portfolio if none exists
    if not portfolio:
        portfolio = Portfolio(name="My Portfolio", total_value=10000.0)
        db.add(portfolio)
        await db.commit()
        await db.refresh(portfolio)
    
    # For lite mode, return portfolio structure without live stock data (instant)
    if lite:
        holdings_with_data = [
            HoldingWithData(
                id=h.id,
                portfolio_id=h.portfolio_id,
                ticker=h.ticker,
                allocation_pct=h.allocation_pct,
                added_at=h.added_at,
                investment_date=h.investment_date,
                investment_price=h.investment_price,
                current_price=None,
                current_value=None,
                ytd_return=None,
                sma_200=None,
                price_vs_sma=None,
                gain_loss=None,
                gain_loss_pct=None
            )
            for h in portfolio.holdings
        ]
        return PortfolioWithData(
            id=portfolio.id,
            name=portfolio.name,
            total_value=portfolio.total_value,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
            holdings=holdings_with_data,
            current_total_value=None,
            total_gain_loss=None,
            total_gain_loss_pct=None
        )
    
    # Full mode: Fetch live data for all holdings (slower)
    holdings_with_data = []
    current_total_value = 0
    total_portfolio_gain_loss = 0
    
    if portfolio.holdings:
        tickers = [h.ticker for h in portfolio.holdings]
        stock_data = await get_multiple_stocks(tickers)
        
        for holding in portfolio.holdings:
            data = stock_data.get(holding.ticker.upper(), {})
            allocated_value = portfolio.total_value * (holding.allocation_pct / 100)
            current_price = data.get('current_price', 0)
            
            # Calculate current value based on price movement (using YTD for display)
            ytd_return = data.get('ytd_return', 0)
            current_value = allocated_value * (1 + ytd_return / 100)
            current_total_value += current_value
            
            # Calculate gain/loss since investment date (if we have investment_price)
            gain_loss = None
            gain_loss_pct = None
            if holding.investment_price and holding.investment_price > 0 and current_price > 0:
                # How many shares would we have bought at investment_price?
                shares = allocated_value / holding.investment_price
                # What's the current value of those shares?
                value_now = shares * current_price
                gain_loss = value_now - allocated_value
                gain_loss_pct = ((current_price - holding.investment_price) / holding.investment_price) * 100
                total_portfolio_gain_loss += gain_loss
            
            holdings_with_data.append(HoldingWithData(
                id=holding.id,
                portfolio_id=holding.portfolio_id,
                ticker=holding.ticker,
                allocation_pct=holding.allocation_pct,
                added_at=holding.added_at,
                investment_date=holding.investment_date,
                investment_price=holding.investment_price,
                current_price=current_price,
                current_value=round(current_value, 2),
                ytd_return=ytd_return,
                sma_200=data.get('sma_200'),
                price_vs_sma=data.get('price_vs_sma'),
                gain_loss=round(gain_loss, 2) if gain_loss is not None else None,
                gain_loss_pct=round(gain_loss_pct, 2) if gain_loss_pct is not None else None
            ))
    else:
        current_total_value = portfolio.total_value
    
    # Calculate total gain/loss based on investment dates
    total_allocated = sum(h.allocation_pct for h in portfolio.holdings) if portfolio.holdings else 0
    base_invested = portfolio.total_value * (total_allocated / 100)
    
    # Only show portfolio gain/loss if we have investment prices for all holdings
    all_have_investment_price = all(h.investment_price for h in portfolio.holdings) if portfolio.holdings else False
    
    if all_have_investment_price and base_invested > 0:
        total_gain_loss = total_portfolio_gain_loss
        total_gain_loss_pct = (total_gain_loss / base_invested * 100)
    else:
        total_gain_loss = None
        total_gain_loss_pct = None
    
    return PortfolioWithData(
        id=portfolio.id,
        name=portfolio.name,
        total_value=portfolio.total_value,
        created_at=portfolio.created_at,
        updated_at=portfolio.updated_at,
        holdings=holdings_with_data,
        current_total_value=round(current_total_value, 2),
        total_gain_loss=round(total_gain_loss, 2),
        total_gain_loss_pct=round(total_gain_loss_pct, 2)
    )


@app.put("/api/portfolio", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_update: PortfolioUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update portfolio settings (name, total value)."""
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
    
    await db.commit()
    await db.refresh(portfolio)
    
    return portfolio


# ============ Holdings Endpoints ============

@app.post("/api/holdings", response_model=HoldingResponse)
async def add_holding(
    holding: HoldingCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a stock holding to the portfolio."""
    # Get or create portfolio
    result = await db.execute(select(Portfolio).limit(1))
    portfolio = result.scalar_one_or_none()
    
    if not portfolio:
        portfolio = Portfolio()
        db.add(portfolio)
        await db.commit()
        await db.refresh(portfolio)
    
    # Check if ticker already exists
    existing = await db.execute(
        select(Holding).where(
            Holding.portfolio_id == portfolio.id,
            Holding.ticker == holding.ticker.upper()
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail=f"Holding for {holding.ticker} already exists")
    
    # Check total allocation doesn't exceed 100%
    result = await db.execute(
        select(Holding).where(Holding.portfolio_id == portfolio.id)
    )
    current_holdings = result.scalars().all()
    total_allocated = sum(h.allocation_pct for h in current_holdings)
    
    if total_allocated + holding.allocation_pct > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Total allocation would exceed 100%. Currently allocated: {total_allocated}%"
        )
    
    # Validate ticker exists (fast validation)
    is_valid, _ = await validate_ticker(holding.ticker)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid ticker: {holding.ticker}")
    
    new_holding = Holding(
        portfolio_id=portfolio.id,
        ticker=holding.ticker.upper(),
        allocation_pct=holding.allocation_pct,
        investment_date=holding.investment_date,
        investment_price=holding.investment_price
    )
    db.add(new_holding)
    await db.commit()
    await db.refresh(new_holding)
    
    return new_holding


@app.delete("/api/holdings/{holding_id}")
async def delete_holding(holding_id: int, db: AsyncSession = Depends(get_db)):
    """Remove a holding from the portfolio."""
    result = await db.execute(select(Holding).where(Holding.id == holding_id))
    holding = result.scalar_one_or_none()
    
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    await db.delete(holding)
    await db.commit()
    
    return {"message": "Holding deleted successfully"}


@app.put("/api/holdings/{holding_id}", response_model=HoldingResponse)
async def update_holding(
    holding_id: int,
    holding_update: HoldingUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update a holding's allocation percentage, investment date, or investment price."""
    result = await db.execute(select(Holding).where(Holding.id == holding_id))
    holding = result.scalar_one_or_none()
    
    if not holding:
        raise HTTPException(status_code=404, detail="Holding not found")
    
    # Check total allocation if allocation is being updated
    if holding_update.allocation_pct is not None:
        result = await db.execute(
            select(Holding).where(
                Holding.portfolio_id == holding.portfolio_id,
                Holding.id != holding_id
            )
        )
        other_holdings = result.scalars().all()
        total_allocated = sum(h.allocation_pct for h in other_holdings)
        
        if total_allocated + holding_update.allocation_pct > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Total allocation would exceed 100%. Other holdings: {total_allocated}%"
            )
        
        holding.allocation_pct = holding_update.allocation_pct
    
    # Update investment date if provided
    if holding_update.investment_date is not None:
        holding.investment_date = holding_update.investment_date
    
    # Update investment price if provided
    if holding_update.investment_price is not None:
        holding.investment_price = holding_update.investment_price
    
    await db.commit()
    await db.refresh(holding)
    
    return holding


# ============ Stock Data Endpoints ============

@app.get("/api/stock/{ticker}", response_model=StockData)
async def get_stock(ticker: str):
    """Get detailed stock data including price, YTD, SMA(200), and history."""
    data = await get_stock_data(ticker)
    
    if data['current_price'] == 0:
        raise HTTPException(status_code=404, detail=f"Stock not found: {ticker}")
    
    return StockData(**data)


@app.get("/api/stock/{ticker}/quote")
async def get_stock_quote(ticker: str):
    """Get just the current price data for a single stock (lightweight refresh)."""
    data = await get_stock_data(ticker)
    
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


@app.get("/api/stock/{ticker}/history")
async def get_stock_history_endpoint(ticker: str, period: str = "1y"):
    """Get extended historical data for a stock.
    
    Returns history data, reference_close, and data completeness info.
    """
    result = await get_stock_history(ticker, period)
    
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


@app.post("/api/stocks/history")
async def get_multiple_stock_histories(tickers: list[str], period: str = "1y"):
    """Get historical data for multiple stocks at once.
    
    More efficient than calling individual endpoints - processes sequentially
    to avoid yfinance data corruption but returns all results in one response.
    """
    results = {}
    for ticker in tickers:
        try:
            result = await get_stock_history(ticker, period)
            results[ticker.upper()] = {
                "history": result.get("history", []),
                "reference_close": result.get("reference_close"),
                "is_complete": result.get("is_complete", True),
                "expected_start": result.get("expected_start"),
                "actual_start": result.get("actual_start")
            }
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
            results[ticker.upper()] = {
                "history": [],
                "reference_close": None,
                "is_complete": False,
                "expected_start": None,
                "actual_start": None
            }
    return results


# ============ Portfolio History ============

@app.get("/api/portfolio/history", response_model=PortfolioHistory)
async def get_portfolio_history(db: AsyncSession = Depends(get_db)):
    """Get portfolio value history over time."""
    result = await db.execute(
        select(PortfolioSnapshot).order_by(PortfolioSnapshot.timestamp)
    )
    snapshots = result.scalars().all()
    
    return PortfolioHistory(
        history=[
            PortfolioHistoryPoint(timestamp=s.timestamp, total_value=s.total_value)
            for s in snapshots
        ]
    )


@app.post("/api/portfolio/snapshot")
async def create_portfolio_snapshot(db: AsyncSession = Depends(get_db)):
    """Create a snapshot of the current portfolio value."""
    # Get portfolio with live data
    portfolio_data = await get_portfolio(db)
    
    result = await db.execute(select(Portfolio).limit(1))
    portfolio = result.scalar_one_or_none()
    
    if portfolio:
        snapshot = PortfolioSnapshot(
            portfolio_id=portfolio.id,
            total_value=portfolio_data.current_total_value or portfolio.total_value
        )
        db.add(snapshot)
        await db.commit()
        
        return {"message": "Snapshot created", "value": snapshot.total_value}
    
    return {"message": "No portfolio found"}


# ============ Cache Management ============

@app.delete("/api/stock/{ticker}/cache")
async def clear_stock_cache(ticker: str):
    """Clear the intraday cache for a specific stock. Forces fresh data fetch on next request."""
    deleted_count = clear_intraday_cache(ticker)
    return {
        "message": f"Cleared intraday cache for {ticker.upper()}",
        "deleted_records": deleted_count
    }


# ============ Health Check ============

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stock-dashboard-api"}

