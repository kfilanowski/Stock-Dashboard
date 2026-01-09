"""
Option Analytics Service - Greeks, breakeven, profit probability calculations.

Implements Black-Scholes calculations for option Greeks and analytics.
"""
import math
from datetime import datetime, date
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..logging_config import get_logger

logger = get_logger(__name__)

# Risk-free rate (approximate 10-year Treasury yield)
RISK_FREE_RATE = 0.045  # 4.5%


@dataclass
class OptionPosition:
    """Represents an option position for analytics calculations."""
    underlying_ticker: str
    option_type: str  # "call" or "put"
    position_type: str  # "long" or "short"
    strike_price: float
    expiration_date: date
    contracts: int
    premium_per_contract: Optional[float]
    underlying_price: Optional[float]
    current_option_price: Optional[float]
    implied_volatility: Optional[float]  # As decimal (e.g., 0.30 for 30%)


def norm_cdf(x: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.
    Uses the Abramowitz and Stegun approximation.
    """
    # Constants for approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2)
    
    return 0.5 * (1.0 + sign * y)


def norm_pdf(x: float) -> float:
    """Probability density function for standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


class OptionAnalyticsService:
    """
    Service for calculating option analytics.
    
    Provides:
    - Black-Scholes Greeks calculation
    - Breakeven price calculation
    - Max profit/loss calculation
    - Profit probability estimation
    """
    
    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
    
    def calculate_days_to_expiration(self, expiration_date: date) -> int:
        """
        Calculate days until expiration using US Eastern time (market time).
        
        Options expire at 4:00 PM ET on the expiration date, so we use Eastern
        timezone to ensure consistent calculations regardless of server timezone.
        """
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        
        # Get current date in US Eastern time (market time)
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)
        today_eastern = now_eastern.date()
        
        delta = expiration_date - today_eastern
        return max(0, delta.days)
    
    def calculate_time_to_expiration(self, expiration_date: date) -> float:
        """Calculate time to expiration in years."""
        days = self.calculate_days_to_expiration(expiration_date)
        return days / 365.0
    
    def is_expired(self, expiration_date: date) -> bool:
        """
        Check if option has expired.
        
        Options expire at 4:00 PM ET on the expiration date. An option is
        considered expired only after market close on expiration day.
        """
        from datetime import datetime, time
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        
        eastern = ZoneInfo("America/New_York")
        now_eastern = datetime.now(eastern)
        today_eastern = now_eastern.date()
        current_time = now_eastern.time()
        
        # Market closes at 4:00 PM ET
        market_close = time(16, 0)
        
        # Expired if: past expiration date, OR on expiration date after 4 PM ET
        if today_eastern > expiration_date:
            return True
        if today_eastern == expiration_date and current_time >= market_close:
            return True
        return False
    
    def is_itm(
        self, 
        option_type: str, 
        strike_price: float, 
        underlying_price: float
    ) -> bool:
        """Check if option is in the money."""
        if option_type.lower() == "call":
            return underlying_price > strike_price
        else:  # put
            return underlying_price < strike_price
    
    def calculate_intrinsic_value(
        self,
        option_type: str,
        strike_price: float,
        underlying_price: float
    ) -> float:
        """Calculate intrinsic value per share."""
        if option_type.lower() == "call":
            return max(0, underlying_price - strike_price)
        else:  # put
            return max(0, strike_price - underlying_price)
    
    def calculate_greeks(
        self,
        option_type: str,
        strike_price: float,
        underlying_price: float,
        time_to_expiration: float,
        implied_volatility: float,
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate option Greeks using Black-Scholes model.
        
        Args:
            option_type: "call" or "put"
            strike_price: Strike price
            underlying_price: Current underlying price
            time_to_expiration: Time to expiration in years
            implied_volatility: IV as decimal (e.g., 0.30 for 30%)
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Dict with delta, gamma, theta, vega, rho
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Handle edge cases
        if time_to_expiration <= 0 or implied_volatility <= 0:
            # Expired or invalid IV - return intrinsic-based delta only
            is_call = option_type.lower() == "call"
            if time_to_expiration <= 0:
                # Expired option
                itm = self.is_itm(option_type, strike_price, underlying_price)
                delta = 1.0 if (is_call and itm) else (-1.0 if (not is_call and itm) else 0.0)
            else:
                delta = None
            
            return {
                "delta": delta,
                "gamma": 0.0 if time_to_expiration <= 0 else None,
                "theta": 0.0 if time_to_expiration <= 0 else None,
                "vega": 0.0 if time_to_expiration <= 0 else None,
                "rho": 0.0 if time_to_expiration <= 0 else None,
            }
        
        try:
            S = underlying_price
            K = strike_price
            T = time_to_expiration
            r = risk_free_rate
            sigma = implied_volatility
            
            # Calculate d1 and d2
            sqrt_T = math.sqrt(T)
            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            is_call = option_type.lower() == "call"
            
            # Delta
            if is_call:
                delta = norm_cdf(d1)
            else:
                delta = norm_cdf(d1) - 1
            
            # Gamma (same for calls and puts)
            gamma = norm_pdf(d1) / (S * sigma * sqrt_T)
            
            # Theta (per day)
            first_term = -(S * norm_pdf(d1) * sigma) / (2 * sqrt_T)
            if is_call:
                second_term = -r * K * math.exp(-r * T) * norm_cdf(d2)
            else:
                second_term = r * K * math.exp(-r * T) * norm_cdf(-d2)
            theta = (first_term + second_term) / 365  # Per day
            
            # Vega (per 1% change in IV)
            vega = S * sqrt_T * norm_pdf(d1) / 100
            
            # Rho (per 1% change in interest rate)
            if is_call:
                rho = K * T * math.exp(-r * T) * norm_cdf(d2) / 100
            else:
                rho = -K * T * math.exp(-r * T) * norm_cdf(-d2) / 100
            
            return {
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "rho": round(rho, 4),
            }
            
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {e}")
            return {
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
            }
    
    def calculate_breakeven(
        self,
        option_type: str,
        position_type: str,
        strike_price: float,
        premium_per_contract: float
    ) -> float:
        """
        Calculate breakeven price for an option position.
        
        Args:
            option_type: "call" or "put"
            position_type: "long" or "short"
            strike_price: Strike price
            premium_per_contract: Premium paid/received per contract (per share)
            
        Returns:
            Breakeven price for the underlying
        """
        # Premium is per share in standard option pricing
        premium = premium_per_contract
        
        if option_type.lower() == "call":
            # Long call: breakeven = strike + premium paid
            # Short call: breakeven = strike + premium received
            return strike_price + premium
        else:  # put
            # Long put: breakeven = strike - premium paid
            # Short put: breakeven = strike - premium received
            return strike_price - premium
    
    def calculate_max_profit_loss(
        self,
        option_type: str,
        position_type: str,
        strike_price: float,
        premium_per_contract: float,
        contracts: int,
        underlying_price: Optional[float] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate maximum profit and loss for a position.
        
        Returns:
            Dict with max_profit, max_loss (both as positive numbers representing amounts)
        """
        premium_per_share = premium_per_contract
        total_premium = premium_per_share * 100 * contracts
        
        is_call = option_type.lower() == "call"
        is_long = position_type.lower() == "long"
        
        if is_call:
            if is_long:
                # Long call: max loss = premium paid, max profit = unlimited
                return {
                    "max_loss": total_premium,
                    "max_profit": None,  # Unlimited
                }
            else:
                # Short call: max profit = premium received, max loss = unlimited
                return {
                    "max_profit": total_premium,
                    "max_loss": None,  # Unlimited (naked) or capped (covered)
                }
        else:  # put
            if is_long:
                # Long put: max loss = premium paid, max profit = strike - premium (if stock goes to 0)
                max_profit = (strike_price - premium_per_share) * 100 * contracts
                return {
                    "max_loss": total_premium,
                    "max_profit": max(0, max_profit),
                }
            else:
                # Short put: max profit = premium received, max loss = strike - premium (if stock goes to 0)
                max_loss = (strike_price - premium_per_share) * 100 * contracts
                return {
                    "max_profit": total_premium,
                    "max_loss": max(0, max_loss),
                }
    
    def calculate_profit_probability(
        self,
        option_type: str,
        position_type: str,
        strike_price: float,
        underlying_price: float,
        implied_volatility: float,
        time_to_expiration: float
    ) -> Optional[float]:
        """
        Estimate probability of profit at expiration.
        
        Uses log-normal distribution assumption to estimate probability
        that the underlying will be above/below the breakeven at expiration.
        
        Returns:
            Probability of profit as percentage (0-100), or None if can't calculate
        """
        if time_to_expiration <= 0 or implied_volatility <= 0:
            return None
        
        try:
            S = underlying_price
            K = strike_price
            T = time_to_expiration
            sigma = implied_volatility
            r = self.risk_free_rate
            
            is_call = option_type.lower() == "call"
            is_long = position_type.lower() == "long"
            
            # Calculate probability of being ITM at expiration
            # Using d2 from Black-Scholes
            sqrt_T = math.sqrt(T)
            d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            
            prob_itm = norm_cdf(d2) if is_call else norm_cdf(-d2)
            
            # For long positions, profit = ITM probability
            # For short positions, profit = OTM probability
            if is_long:
                profit_prob = prob_itm * 100
            else:
                profit_prob = (1 - prob_itm) * 100
            
            return round(profit_prob, 1)
            
        except Exception as e:
            logger.warning(f"Error calculating profit probability: {e}")
            return None
    
    def calculate_position_analytics(
        self,
        position: OptionPosition
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive analytics for an option position.
        
        Args:
            position: OptionPosition with all relevant data
            
        Returns:
            Dict with all analytics including Greeks, breakeven, P/L, etc.
        """
        days_to_exp = self.calculate_days_to_expiration(position.expiration_date)
        time_to_exp = days_to_exp / 365.0
        is_expired = self.is_expired(position.expiration_date)
        
        # Basic checks
        has_underlying_price = position.underlying_price is not None
        has_option_price = position.current_option_price is not None
        has_premium = position.premium_per_contract is not None
        has_iv = position.implied_volatility is not None
        
        result = {
            "days_to_expiration": days_to_exp,
            "is_expired": is_expired,
            "is_itm": None,
            "intrinsic_value": None,
            "time_value": None,
            "breakeven_price": None,
            "max_profit": None,
            "max_loss": None,
            "profit_probability": None,
            "greeks": {
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
            },
            "position_value": None,
            "cost_basis": None,
            "gain_loss": None,
            "gain_loss_pct": None,
        }
        
        if has_underlying_price:
            result["is_itm"] = self.is_itm(
                position.option_type,
                position.strike_price,
                position.underlying_price
            )
            result["intrinsic_value"] = self.calculate_intrinsic_value(
                position.option_type,
                position.strike_price,
                position.underlying_price
            )
        
        if has_option_price and result["intrinsic_value"] is not None:
            result["time_value"] = max(0, position.current_option_price - result["intrinsic_value"])
        
        if has_premium:
            result["breakeven_price"] = self.calculate_breakeven(
                position.option_type,
                position.position_type,
                position.strike_price,
                position.premium_per_contract
            )
            
            profit_loss = self.calculate_max_profit_loss(
                position.option_type,
                position.position_type,
                position.strike_price,
                position.premium_per_contract,
                position.contracts
            )
            result["max_profit"] = profit_loss["max_profit"]
            result["max_loss"] = profit_loss["max_loss"]
            
            # Calculate cost basis and P/L
            result["cost_basis"] = position.premium_per_contract * 100 * position.contracts
        
        if has_underlying_price and has_iv and time_to_exp > 0:
            result["greeks"] = self.calculate_greeks(
                position.option_type,
                position.strike_price,
                position.underlying_price,
                time_to_exp,
                position.implied_volatility / 100  # Convert from percentage to decimal
            )
            
            result["profit_probability"] = self.calculate_profit_probability(
                position.option_type,
                position.position_type,
                position.strike_price,
                position.underlying_price,
                position.implied_volatility / 100,
                time_to_exp
            )
        
        if has_option_price:
            # Position value
            current_value = position.current_option_price * 100 * position.contracts
            result["position_value"] = round(current_value, 2)
            
            # P/L calculation depends on long vs short
            if has_premium:
                if position.position_type.lower() == "long":
                    # Long: profit = current value - cost
                    result["gain_loss"] = round(current_value - result["cost_basis"], 2)
                else:
                    # Short: profit = premium received - current liability
                    result["gain_loss"] = round(result["cost_basis"] - current_value, 2)
                
                if result["cost_basis"] > 0:
                    result["gain_loss_pct"] = round(
                        (result["gain_loss"] / result["cost_basis"]) * 100, 2
                    )
        
        return result
    
    def calculate_portfolio_greeks(
        self,
        positions: list[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for a portfolio of options.
        
        Args:
            positions: List of position dicts with greeks and contracts
            
        Returns:
            Dict with aggregate delta, gamma, theta, vega
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for pos in positions:
            greeks = pos.get("greeks", {})
            contracts = pos.get("contracts", 1)
            position_type = pos.get("position_type", "long")
            
            # Multiplier for short positions (Greeks are inverted)
            multiplier = 1 if position_type.lower() == "long" else -1
            
            if greeks.get("delta"):
                total_delta += greeks["delta"] * 100 * contracts * multiplier
            if greeks.get("gamma"):
                total_gamma += greeks["gamma"] * 100 * contracts * multiplier
            if greeks.get("theta"):
                total_theta += greeks["theta"] * 100 * contracts * multiplier
            if greeks.get("vega"):
                total_vega += greeks["vega"] * 100 * contracts * multiplier
        
        return {
            "total_delta": round(total_delta, 2),  # Equivalent share exposure
            "total_gamma": round(total_gamma, 4),
            "total_theta": round(total_theta, 2),  # Daily time decay in $
            "total_vega": round(total_vega, 2),    # $ change per 1% IV move
        }


# Singleton instance
_analytics_service: Optional[OptionAnalyticsService] = None


def get_option_analytics_service() -> OptionAnalyticsService:
    """Get the singleton option analytics service."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = OptionAnalyticsService()
    return _analytics_service

