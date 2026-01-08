"""
Portfolio business logic service.

Handles all portfolio-related operations, keeping the API layer thin.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ..config import settings
from ..logging_config import get_logger
from ..models import Portfolio, Holding
from ..schemas import (
    PortfolioWithData, HoldingWithData, HoldingCreate, HoldingUpdate
)
from .stock_fetcher import StockFetcher, get_stock_fetcher
from .calculations import StockCalculations

logger = get_logger(__name__)


class PortfolioService:
    """
    Service for portfolio operations.
    
    Handles portfolio retrieval, updates, and calculations.
    """
    
    def __init__(self, stock_fetcher: Optional[StockFetcher] = None):
        """
        Initialize the portfolio service.
        
        Args:
            stock_fetcher: Stock fetcher instance for live data.
        """
        self._stock_fetcher = stock_fetcher or get_stock_fetcher()
        self._calc = StockCalculations()
    
    async def get_portfolio(
        self, 
        db: AsyncSession, 
        lite: bool = False
    ) -> PortfolioWithData:
        """
        Get the portfolio with optional live data.
        
        Args:
            db: Database session.
            lite: If True, return structure without live data (fast).
            
        Returns:
            Portfolio with holdings and calculated values.
        """
        # Get or create portfolio
        portfolio = await self._get_or_create_portfolio(db)
        
        # Lite mode: return structure without live data
        if lite:
            return self._build_lite_response(portfolio)
        
        # Full mode: fetch live data
        return await self._build_full_response(portfolio)
    
    async def _get_or_create_portfolio(self, db: AsyncSession) -> Portfolio:
        """Get existing portfolio or create default one."""
        result = await db.execute(
            select(Portfolio)
            .options(selectinload(Portfolio.holdings))
            .limit(1)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            portfolio = Portfolio(
                name=settings.default_portfolio_name,
                total_value=settings.default_portfolio_value
            )
            db.add(portfolio)
            await db.commit()
            await db.refresh(portfolio)
            logger.info("Created default portfolio")
        
        return portfolio
    
    def _build_lite_response(self, portfolio: Portfolio) -> PortfolioWithData:
        """Build response without live stock data."""
        holdings_with_data = [
            HoldingWithData(
                id=h.id,
                portfolio_id=h.portfolio_id,
                ticker=h.ticker,
                shares=h.shares,
                avg_cost=h.avg_cost,
                added_at=h.added_at,
                is_pinned=h.is_pinned,
                current_price=None,
                market_value=None,
                cost_basis=h.shares * h.avg_cost if h.shares and h.avg_cost else None,
                allocation_pct=None,
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
            chart_period=portfolio.chart_period,
            sort_field=portfolio.sort_field,
            sort_direction=portfolio.sort_direction,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
            holdings=holdings_with_data,
            total_market_value=None,
            total_cost_basis=None,
            total_gain_loss=None,
            total_gain_loss_pct=None
        )
    
    async def _build_full_response(self, portfolio: Portfolio) -> PortfolioWithData:
        """Build response with live stock data."""
        holdings_with_data = []
        total_market_value = 0.0
        total_cost_basis = 0.0
        
        if portfolio.holdings:
            # Fetch live data for all holdings
            tickers = [h.ticker for h in portfolio.holdings]
            stock_data = await self._stock_fetcher.get_multiple_stocks(tickers)
            
            # First pass: calculate market values for allocation percentages
            market_values = {}
            for holding in portfolio.holdings:
                data = stock_data.get(holding.ticker.upper(), {})
                current_price = data.get('current_price', 0)
                market_value = holding.shares * current_price if holding.shares and current_price else 0
                market_values[holding.id] = market_value
                total_market_value += market_value
                
                if holding.shares and holding.avg_cost:
                    total_cost_basis += holding.shares * holding.avg_cost
            
            # Second pass: build holding data with allocation percentages
            for holding in portfolio.holdings:
                holding_data = self._calculate_holding_data(
                    holding, 
                    stock_data.get(holding.ticker.upper(), {}),
                    total_market_value
                )
                holdings_with_data.append(holding_data)
        
        # Calculate total gain/loss
        total_gain_loss = total_market_value - total_cost_basis if total_cost_basis > 0 else None
        total_gain_loss_pct = (total_gain_loss / total_cost_basis * 100) if total_cost_basis > 0 and total_gain_loss is not None else None
        
        return PortfolioWithData(
            id=portfolio.id,
            name=portfolio.name,
            total_value=portfolio.total_value,
            chart_period=portfolio.chart_period,
            sort_field=portfolio.sort_field,
            sort_direction=portfolio.sort_direction,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
            holdings=holdings_with_data,
            total_market_value=round(total_market_value, 2),
            total_cost_basis=round(total_cost_basis, 2) if total_cost_basis > 0 else None,
            total_gain_loss=round(total_gain_loss, 2) if total_gain_loss is not None else None,
            total_gain_loss_pct=round(total_gain_loss_pct, 2) if total_gain_loss_pct is not None else None
        )
    
    def _calculate_holding_data(
        self, 
        holding: Holding, 
        stock_data: Dict[str, Any],
        total_market_value: float
    ) -> HoldingWithData:
        """Calculate derived values for a single holding."""
        current_price = stock_data.get('current_price', 0)
        ytd_return = stock_data.get('ytd_return', 0)
        
        # Calculate market value and cost basis
        market_value = holding.shares * current_price if holding.shares and current_price else None
        cost_basis = holding.shares * holding.avg_cost if holding.shares and holding.avg_cost else None
        
        # Calculate allocation percentage
        allocation_pct = (market_value / total_market_value * 100) if market_value and total_market_value > 0 else None
        
        # Calculate gain/loss
        gain_loss = None
        gain_loss_pct = None
        if market_value is not None and cost_basis is not None:
            gain_loss = round(market_value - cost_basis, 2)
            if holding.avg_cost and holding.avg_cost > 0:
                gain_loss_pct = round((current_price - holding.avg_cost) / holding.avg_cost * 100, 2)
        
        return HoldingWithData(
            id=holding.id,
            portfolio_id=holding.portfolio_id,
            ticker=holding.ticker,
            shares=holding.shares,
            avg_cost=holding.avg_cost,
            added_at=holding.added_at,
            is_pinned=holding.is_pinned,
            current_price=current_price,
            market_value=round(market_value, 2) if market_value is not None else None,
            cost_basis=round(cost_basis, 2) if cost_basis is not None else None,
            allocation_pct=round(allocation_pct, 2) if allocation_pct is not None else None,
            ytd_return=ytd_return,
            sma_200=stock_data.get('sma_200'),
            price_vs_sma=stock_data.get('price_vs_sma'),
            gain_loss=gain_loss,
            gain_loss_pct=gain_loss_pct
        )
    
    # ============ Holding Operations ============
    
    async def add_holding(
        self, 
        db: AsyncSession, 
        holding_data: HoldingCreate
    ) -> Holding:
        """
        Add a new holding to the portfolio.
        
        Args:
            db: Database session.
            holding_data: Holding creation data.
            
        Returns:
            Created holding.
            
        Raises:
            ValueError: If ticker already exists.
        """
        portfolio = await self._get_or_create_portfolio(db)
        
        # Check for existing ticker
        existing = await db.execute(
            select(Holding).where(
                Holding.portfolio_id == portfolio.id,
                Holding.ticker == holding_data.ticker.upper()
            )
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Holding for {holding_data.ticker} already exists")
        
        # Validate ticker
        is_valid, _ = await self._stock_fetcher.validate_ticker(holding_data.ticker)
        if not is_valid:
            raise ValueError(f"Invalid ticker: {holding_data.ticker}")
        
        # Create holding
        new_holding = Holding(
            portfolio_id=portfolio.id,
            ticker=holding_data.ticker.upper(),
            shares=holding_data.shares,
            avg_cost=holding_data.avg_cost,
            allocation_pct=0  # Legacy column - kept for DB compatibility
        )
        db.add(new_holding)
        await db.commit()
        await db.refresh(new_holding)
        
        logger.info(f"Added holding: {new_holding.ticker} with {new_holding.shares} shares")
        return new_holding
    
    async def update_holding(
        self, 
        db: AsyncSession, 
        holding_id: int, 
        update_data: HoldingUpdate
    ) -> Holding:
        """
        Update a holding's shares or average cost.
        
        Args:
            db: Database session.
            holding_id: ID of holding to update.
            update_data: Update data.
            
        Returns:
            Updated holding.
            
        Raises:
            ValueError: If holding not found.
        """
        result = await db.execute(
            select(Holding).where(Holding.id == holding_id)
        )
        holding = result.scalar_one_or_none()
        
        if not holding:
            raise ValueError("Holding not found")
        
        if update_data.shares is not None:
            holding.shares = update_data.shares
        
        if update_data.avg_cost is not None:
            holding.avg_cost = update_data.avg_cost
        
        if update_data.is_pinned is not None:
            holding.is_pinned = update_data.is_pinned
        
        await db.commit()
        await db.refresh(holding)
        
        logger.info(f"Updated holding {holding_id}: {holding.ticker} - {holding.shares} shares @ ${holding.avg_cost}")
        return holding
    
    async def delete_holding(self, db: AsyncSession, holding_id: int) -> bool:
        """
        Delete a holding from the portfolio.
        
        Args:
            db: Database session.
            holding_id: ID of holding to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        result = await db.execute(
            select(Holding).where(Holding.id == holding_id)
        )
        holding = result.scalar_one_or_none()
        
        if not holding:
            return False
        
        ticker = holding.ticker
        await db.delete(holding)
        await db.commit()
        
        logger.info(f"Deleted holding: {ticker}")
        return True


# Singleton instance
_portfolio_service: Optional[PortfolioService] = None


def get_portfolio_service() -> PortfolioService:
    """Get the singleton portfolio service instance."""
    global _portfolio_service
    if _portfolio_service is None:
        _portfolio_service = PortfolioService()
    return _portfolio_service

