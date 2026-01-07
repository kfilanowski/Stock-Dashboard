from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# Portfolio Schemas
class PortfolioBase(BaseModel):
    name: str = "My Portfolio"
    total_value: float = 10000.0
    chart_period: Optional[str] = "1d"
    sort_field: Optional[str] = "allocation"
    sort_direction: Optional[str] = "desc"


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    total_value: Optional[float] = None
    chart_period: Optional[str] = None
    sort_field: Optional[str] = None
    sort_direction: Optional[str] = None


class HoldingBase(BaseModel):
    ticker: str
    shares: float = Field(default=0, ge=0)  # Number of shares


class HoldingCreate(HoldingBase):
    shares: float = Field(default=0, ge=0)  # Number of shares
    avg_cost: Optional[float] = Field(None, ge=0)  # Average cost per share


class HoldingUpdate(BaseModel):
    shares: Optional[float] = Field(None, ge=0)
    avg_cost: Optional[float] = Field(None, ge=0)


class HoldingResponse(HoldingBase):
    id: int
    portfolio_id: int
    added_at: datetime
    avg_cost: Optional[float] = None

    class Config:
        from_attributes = True


class HoldingWithData(HoldingResponse):
    current_price: Optional[float] = None
    market_value: Optional[float] = None  # shares * current_price
    cost_basis: Optional[float] = None  # shares * avg_cost
    allocation_pct: Optional[float] = None  # market_value / total_portfolio_value * 100
    ytd_return: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma: Optional[float] = None  # Percentage above/below SMA
    gain_loss: Optional[float] = None  # market_value - cost_basis
    gain_loss_pct: Optional[float] = None  # (current_price - avg_cost) / avg_cost * 100


class PortfolioResponse(PortfolioBase):
    id: int
    created_at: datetime
    updated_at: datetime
    holdings: list[HoldingResponse] = []

    class Config:
        from_attributes = True


class PortfolioWithData(PortfolioResponse):
    holdings: list[HoldingWithData] = []
    total_market_value: Optional[float] = None  # Sum of all holdings' market values
    total_cost_basis: Optional[float] = None  # Sum of all holdings' cost basis
    total_gain_loss: Optional[float] = None  # total_market_value - total_cost_basis
    total_gain_loss_pct: Optional[float] = None  # Percentage gain/loss


# Stock Schemas
class StockData(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    change: float
    change_pct: float
    ytd_return: float
    sma_200: Optional[float]
    price_vs_sma: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    history: list[dict]  # List of {date, open, high, low, close, volume}



