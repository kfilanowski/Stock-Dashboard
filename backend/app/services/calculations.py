"""
Financial calculations for stock data.

Provides utility functions for YTD returns, SMA, gain/loss calculations.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from ..logging_config import get_logger

logger = get_logger(__name__)


class StockCalculations:
    """Static methods for stock-related calculations."""
    
    @staticmethod
    def safe_float(value) -> float:
        """
        Safely convert a value to float, handling Series and other types.
        
        Args:
            value: Value to convert (can be float, Series, or other numeric type).
            
        Returns:
            Float value, or 0.0 if conversion fails.
        """
        if pd.isna(value):
            return 0.0
        if hasattr(value, 'iloc'):
            return float(value.iloc[0]) if len(value) > 0 else 0.0
        if hasattr(value, 'item'):
            return float(value.item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    
    @staticmethod
    def safe_int(value) -> int:
        """
        Safely convert a value to int, handling Series and other types.
        
        Args:
            value: Value to convert.
            
        Returns:
            Int value, or 0 if conversion fails.
        """
        if pd.isna(value):
            return 0
        if hasattr(value, 'iloc'):
            return int(value.iloc[0]) if len(value) > 0 else 0
        if hasattr(value, 'item'):
            return int(value.item())
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int = 200) -> Optional[float]:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: List of closing prices.
            period: SMA period (default 200).
            
        Returns:
            SMA value or None if insufficient data.
        """
        if len(prices) < period:
            # Use available data if we have at least 50 data points
            if len(prices) >= 50:
                return sum(prices) / len(prices)
            return None
        
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_price_vs_sma(current_price: float, sma: float) -> float:
        """
        Calculate percentage difference from SMA.
        
        Args:
            current_price: Current stock price.
            sma: Simple moving average value.
            
        Returns:
            Percentage above/below SMA.
        """
        if sma <= 0:
            return 0.0
        return ((current_price - sma) / sma) * 100
    
    @staticmethod
    def calculate_ytd_return(
        current_price: float, 
        history: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Year-to-Date return.
        
        Uses the last trading day of the previous year as reference.
        Falls back to first trading day of current year if no previous year data.
        
        Args:
            current_price: Current stock price.
            history: List of historical price data with 'date' and 'close' keys.
            
        Returns:
            YTD return as percentage.
        """
        if not history or current_price <= 0:
            return 0.0
        
        year_start = datetime(datetime.now().year, 1, 1)
        year_start_str = year_start.strftime('%Y-%m-%d')
        
        # Preferred: Use last trading day of previous year
        prev_year_prices = [h for h in history if h['date'] < year_start_str]
        if prev_year_prices:
            reference_price = prev_year_prices[-1]['close']
            if reference_price and reference_price > 0:
                return ((current_price - reference_price) / reference_price) * 100
        
        # Fallback: Use first trading day of current year
        ytd_prices = [h for h in history if h['date'] >= year_start_str]
        if len(ytd_prices) > 1:
            first_price = ytd_prices[0]['close']
            if first_price and first_price > 0 and first_price != current_price:
                return ((current_price - first_price) / first_price) * 100
        
        return 0.0
    
    @staticmethod
    def calculate_gain_loss(
        investment_price: float,
        current_price: float,
        allocated_value: float
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate gain/loss since investment.
        
        Args:
            investment_price: Price at time of investment.
            current_price: Current stock price.
            allocated_value: Dollar amount allocated to this holding.
            
        Returns:
            Tuple of (gain_loss_dollars, gain_loss_percent) or (None, None).
        """
        if not investment_price or investment_price <= 0 or current_price <= 0:
            return None, None
        
        # Calculate shares that would have been bought
        shares = allocated_value / investment_price
        # Current value of those shares
        current_value = shares * current_price
        
        gain_loss = current_value - allocated_value
        gain_loss_pct = ((current_price - investment_price) / investment_price) * 100
        
        return round(gain_loss, 2), round(gain_loss_pct, 2)
    
    @staticmethod
    def calculate_period_gain(
        history: List[Dict[str, Any]], 
        reference_close: Optional[float]
    ) -> Optional[float]:
        """
        Calculate gain for a period relative to reference close.
        
        Args:
            history: Price history for the period.
            reference_close: Closing price before the period started.
            
        Returns:
            Period gain as percentage, or None if cannot be calculated.
        """
        if not history or reference_close is None or reference_close <= 0:
            return None
        
        latest_close = history[-1].get('close', 0)
        if latest_close <= 0:
            return None
        
        return ((latest_close - reference_close) / reference_close) * 100

