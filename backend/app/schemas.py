from pydantic import BaseModel, Field, field_validator
from datetime import datetime, date
from typing import Optional, Literal


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


# ============================================================================
# Option Holding Schemas
# ============================================================================

class OptionHoldingBase(BaseModel):
    """Base schema for option holdings."""
    underlying_ticker: str
    option_type: Literal["call", "put"]
    position_type: Literal["long", "short"]
    strike_price: float = Field(gt=0)
    expiration_date: date
    contracts: int = Field(default=1, ge=1)
    premium_per_contract: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None
    
    @field_validator('underlying_ticker')
    @classmethod
    def uppercase_ticker(cls, v: str) -> str:
        return v.upper().strip()


class OptionHoldingCreate(OptionHoldingBase):
    """Schema for creating a new option holding."""
    pass


class OptionHoldingUpdate(BaseModel):
    """Schema for updating an option holding."""
    contracts: Optional[int] = Field(None, ge=1)
    premium_per_contract: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None


class OptionHoldingResponse(OptionHoldingBase):
    """Schema for option holding response (without live data)."""
    id: int
    portfolio_id: int
    opened_at: datetime

    class Config:
        from_attributes = True


class OptionGreeks(BaseModel):
    """Greeks for an option position."""
    delta: Optional[float] = None  # Rate of change vs underlying price
    gamma: Optional[float] = None  # Rate of change of delta
    theta: Optional[float] = None  # Time decay (per day)
    vega: Optional[float] = None   # Sensitivity to IV changes
    rho: Optional[float] = None    # Sensitivity to interest rate


class OptionAnalytics(BaseModel):
    """Calculated analytics for an option position."""
    breakeven_price: Optional[float] = None  # Price at which position breaks even
    max_profit: Optional[float] = None       # Maximum possible profit
    max_loss: Optional[float] = None         # Maximum possible loss
    profit_probability: Optional[float] = None  # Estimated probability of profit (0-100)
    days_to_expiration: int
    is_itm: bool  # In the money
    is_expired: bool
    intrinsic_value: Optional[float] = None  # Current intrinsic value per contract
    time_value: Optional[float] = None       # Current time/extrinsic value per contract


class OptionHoldingWithData(OptionHoldingResponse):
    """Schema for option holding with live market data and analytics."""
    # Current market data
    underlying_price: Optional[float] = None
    current_price: Optional[float] = None      # Current option price per share
    bid: Optional[float] = None
    ask: Optional[float] = None
    implied_volatility: Optional[float] = None  # IV as percentage
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    
    # Position values
    position_value: Optional[float] = None     # contracts * 100 * current_price
    cost_basis: Optional[float] = None         # contracts * 100 * premium_per_contract
    gain_loss: Optional[float] = None          # position_value - cost_basis (or inverted for short)
    gain_loss_pct: Optional[float] = None      # Percentage gain/loss
    
    # Greeks
    greeks: Optional[OptionGreeks] = None
    
    # Analytics
    analytics: Optional[OptionAnalytics] = None

