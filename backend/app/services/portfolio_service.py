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
            chart_period=portfolio.chart_period,
            sort_field=portfolio.sort_field,
            sort_direction=portfolio.sort_direction,
            created_at=portfolio.created_at,
            updated_at=portfolio.updated_at,
            holdings=holdings_with_data,
            current_total_value=None,
            total_gain_loss=None,
            total_gain_loss_pct=None
        )
    
    async def _build_full_response(self, portfolio: Portfolio) -> PortfolioWithData:
        """Build response with live stock data."""
        holdings_with_data = []
        current_total_value = 0.0
        total_portfolio_gain_loss = 0.0
        
        if portfolio.holdings:
            # Fetch live data for all holdings
            tickers = [h.ticker for h in portfolio.holdings]
            stock_data = await self._stock_fetcher.get_multiple_stocks(tickers)
            
            for holding in portfolio.holdings:
                holding_data = self._calculate_holding_data(
                    holding, 
                    stock_data.get(holding.ticker.upper(), {}),
                    portfolio.total_value
                )
                holdings_with_data.append(holding_data)
                current_total_value += holding_data.current_value or 0
                
                if holding_data.gain_loss is not None:
                    total_portfolio_gain_loss += holding_data.gain_loss
        else:
            current_total_value = portfolio.total_value
        
        # Calculate total gain/loss
        total_gain_loss, total_gain_loss_pct = self._calculate_portfolio_totals(
            portfolio, holdings_with_data, total_portfolio_gain_loss
        )
        
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
            current_total_value=round(current_total_value, 2),
            total_gain_loss=round(total_gain_loss, 2) if total_gain_loss else None,
            total_gain_loss_pct=round(total_gain_loss_pct, 2) if total_gain_loss_pct else None
        )
    
    def _calculate_holding_data(
        self, 
        holding: Holding, 
        stock_data: Dict[str, Any],
        portfolio_total_value: float
    ) -> HoldingWithData:
        """Calculate derived values for a single holding."""
        allocated_value = portfolio_total_value * (holding.allocation_pct / 100)
        current_price = stock_data.get('current_price', 0)
        ytd_return = stock_data.get('ytd_return', 0)
        
        # Calculate current value based on YTD movement
        current_value = allocated_value * (1 + ytd_return / 100)
        
        # Calculate gain/loss since investment
        gain_loss, gain_loss_pct = self._calc.calculate_gain_loss(
            holding.investment_price,
            current_price,
            allocated_value
        )
        
        return HoldingWithData(
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
            sma_200=stock_data.get('sma_200'),
            price_vs_sma=stock_data.get('price_vs_sma'),
            gain_loss=gain_loss,
            gain_loss_pct=gain_loss_pct
        )
    
    def _calculate_portfolio_totals(
        self,
        portfolio: Portfolio,
        holdings: List[HoldingWithData],
        total_gain_loss: float
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate total portfolio gain/loss."""
        if not portfolio.holdings:
            return None, None
        
        total_allocated = sum(h.allocation_pct for h in portfolio.holdings)
        base_invested = portfolio.total_value * (total_allocated / 100)
        
        # Only show if all holdings have investment prices
        all_have_investment_price = all(
            h.investment_price for h in portfolio.holdings
        )
        
        if all_have_investment_price and base_invested > 0:
            return total_gain_loss, (total_gain_loss / base_invested * 100)
        
        return None, None
    
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
            ValueError: If ticker already exists or allocation exceeds 100%.
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
        
        # Check allocation limit
        result = await db.execute(
            select(Holding).where(Holding.portfolio_id == portfolio.id)
        )
        current_holdings = result.scalars().all()
        total_allocated = sum(h.allocation_pct for h in current_holdings)
        
        if total_allocated + holding_data.allocation_pct > 100:
            raise ValueError(
                f"Total allocation would exceed 100%. "
                f"Currently allocated: {total_allocated}%"
            )
        
        # Validate ticker
        is_valid, _ = await self._stock_fetcher.validate_ticker(holding_data.ticker)
        if not is_valid:
            raise ValueError(f"Invalid ticker: {holding_data.ticker}")
        
        # Create holding
        new_holding = Holding(
            portfolio_id=portfolio.id,
            ticker=holding_data.ticker.upper(),
            allocation_pct=holding_data.allocation_pct,
            investment_date=holding_data.investment_date,
            investment_price=holding_data.investment_price
        )
        db.add(new_holding)
        await db.commit()
        await db.refresh(new_holding)
        
        logger.info(f"Added holding: {new_holding.ticker} at {new_holding.allocation_pct}%")
        return new_holding
    
    async def update_holding(
        self, 
        db: AsyncSession, 
        holding_id: int, 
        update_data: HoldingUpdate
    ) -> Holding:
        """
        Update a holding's allocation or investment info.
        
        Args:
            db: Database session.
            holding_id: ID of holding to update.
            update_data: Update data.
            
        Returns:
            Updated holding.
            
        Raises:
            ValueError: If holding not found or allocation exceeds 100%.
        """
        result = await db.execute(
            select(Holding).where(Holding.id == holding_id)
        )
        holding = result.scalar_one_or_none()
        
        if not holding:
            raise ValueError("Holding not found")
        
        # Check allocation if being updated
        if update_data.allocation_pct is not None:
            result = await db.execute(
                select(Holding).where(
                    Holding.portfolio_id == holding.portfolio_id,
                    Holding.id != holding_id
                )
            )
            other_holdings = result.scalars().all()
            total_allocated = sum(h.allocation_pct for h in other_holdings)
            
            if total_allocated + update_data.allocation_pct > 100:
                raise ValueError(
                    f"Total allocation would exceed 100%. "
                    f"Other holdings: {total_allocated}%"
                )
            
            holding.allocation_pct = update_data.allocation_pct
        
        if update_data.investment_date is not None:
            holding.investment_date = update_data.investment_date
        
        if update_data.investment_price is not None:
            holding.investment_price = update_data.investment_price
        
        await db.commit()
        await db.refresh(holding)
        
        logger.info(f"Updated holding {holding_id}: {holding.ticker}")
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

