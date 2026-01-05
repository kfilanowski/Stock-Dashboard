from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


# Portfolio Schemas
class PortfolioBase(BaseModel):
    name: str = "My Portfolio"
    total_value: float = 10000.0


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    total_value: Optional[float] = None


class HoldingBase(BaseModel):
    ticker: str
    allocation_pct: float = Field(..., gt=0, le=100)


class HoldingCreate(HoldingBase):
    pass


class HoldingUpdate(BaseModel):
    allocation_pct: float = Field(..., gt=0, le=100)


class HoldingResponse(HoldingBase):
    id: int
    portfolio_id: int
    added_at: datetime

    class Config:
        from_attributes = True


class HoldingWithData(HoldingResponse):
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    ytd_return: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma: Optional[float] = None  # Percentage above/below SMA


class PortfolioResponse(PortfolioBase):
    id: int
    created_at: datetime
    updated_at: datetime
    holdings: list[HoldingResponse] = []

    class Config:
        from_attributes = True


class PortfolioWithData(PortfolioResponse):
    holdings: list[HoldingWithData] = []
    current_total_value: Optional[float] = None
    total_gain_loss: Optional[float] = None
    total_gain_loss_pct: Optional[float] = None


# Stock Schemas
class StockPrice(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    change: float
    change_pct: float


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


class PortfolioHistoryPoint(BaseModel):
    timestamp: datetime
    total_value: float


class PortfolioHistory(BaseModel):
    history: list[PortfolioHistoryPoint]

