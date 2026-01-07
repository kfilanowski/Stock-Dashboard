"""
Option Pricing Service - Fetches real-time option prices from Yahoo Finance.

Uses yahooquery to fetch option chain data and match user positions
to current market prices.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import Dict, Any, Optional, List

from yahooquery import Ticker as YQTicker

from ..logging_config import get_logger
from .option_analytics import get_option_analytics_service

logger = get_logger(__name__)


class OptionPricingService:
    """
    Service for fetching real-time option prices.
    
    Uses yahooquery to fetch option chains and extract pricing data
    for specific contracts based on strike/expiration/type.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers or 4)
    
    @staticmethod
    def _get_current_price(quote_data: Dict[str, Any]) -> Optional[float]:
        """
        Get the most current stock price, including after-hours/pre-market.
        
        Priority:
        1. Post-market price (if in after-hours and available)
        2. Pre-market price (if in pre-market and available)
        3. Regular market price (fallback)
        
        Args:
            quote_data: Price data dict from yahooquery
            
        Returns:
            Most current price available
        """
        if not quote_data or isinstance(quote_data, str):
            return None
        
        market_state = quote_data.get('marketState', '').upper()
        
        # After hours - use post-market price if available
        if market_state in ('POST', 'POSTPOST', 'CLOSED'):
            post_price = quote_data.get('postMarketPrice')
            if post_price and post_price > 0:
                return post_price
        
        # Pre-market - use pre-market price if available
        if market_state in ('PRE', 'PREPRE'):
            pre_price = quote_data.get('preMarketPrice')
            if pre_price and pre_price > 0:
                return pre_price
        
        # Regular market hours or fallback
        return quote_data.get('regularMarketPrice')
    
    def _build_occ_symbol(
        self,
        ticker: str,
        expiration: date,
        option_type: str,
        strike: float
    ) -> str:
        """
        Build OCC option symbol.
        
        Format: AAPL250321C00200000
        - AAPL: underlying ticker (padded to 6 chars)
        - 250321: expiration date YYMMDD
        - C/P: call or put
        - 00200000: strike price * 1000, padded to 8 digits
        """
        ticker_part = ticker.upper()
        date_part = expiration.strftime("%y%m%d")
        type_part = "C" if option_type.lower() == "call" else "P"
        strike_part = f"{int(strike * 1000):08d}"
        
        return f"{ticker_part}{date_part}{type_part}{strike_part}"
    
    def _get_option_price_sync(
        self,
        underlying_ticker: str,
        expiration_date: date,
        option_type: str,
        strike_price: float
    ) -> Dict[str, Any]:
        """
        Fetch current price data for a specific option contract.
        
        Args:
            underlying_ticker: The underlying stock ticker (e.g., "AAPL")
            expiration_date: Option expiration date
            option_type: "call" or "put"
            strike_price: Strike price
            
        Returns:
            Dict with current option price data, Greeks, etc.
        """
        ticker = underlying_ticker.upper()
        
        try:
            t = YQTicker(ticker)
            
            # Get the underlying stock price (including after-hours if available)
            quote_data = t.price.get(ticker, {})
            if isinstance(quote_data, str):
                quote_data = {}
            underlying_price = self._get_current_price(quote_data)
            
            # Get option chain
            option_chain = t.option_chain
            
            if isinstance(option_chain, str) or option_chain is None:
                logger.warning(f"No options chain available for {ticker}")
                return self._empty_price_response(ticker, underlying_price)
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return self._empty_price_response(ticker, underlying_price)
            
            # Reset index for easier filtering
            if hasattr(option_chain, 'reset_index'):
                option_chain = option_chain.reset_index()
            
            # Filter to our specific contract
            # Match expiration date
            if 'expiration' in option_chain.columns:
                option_chain['exp_date'] = option_chain['expiration'].apply(
                    lambda x: x.date() if isinstance(x, datetime) else 
                              datetime.strptime(str(x)[:10], '%Y-%m-%d').date()
                )
                filtered = option_chain[option_chain['exp_date'] == expiration_date]
            else:
                filtered = option_chain
            
            # Match option type
            if 'optionType' in filtered.columns:
                type_filter = 'calls' if option_type.lower() == 'call' else 'puts'
                filtered = filtered[filtered['optionType'] == type_filter]
            
            # Match strike price (with small tolerance for floating point)
            if 'strike' in filtered.columns:
                filtered = filtered[abs(filtered['strike'] - strike_price) < 0.01]
            
            if filtered.empty:
                logger.debug(
                    f"No matching contract for {ticker} {option_type} "
                    f"${strike_price} exp {expiration_date}"
                )
                return self._empty_price_response(ticker, underlying_price)
            
            # Get the first matching row
            contract = filtered.iloc[0]
            
            # Extract price data
            last_price = contract.get('lastPrice')
            bid = contract.get('bid')
            ask = contract.get('ask')
            
            # Calculate mid price if available
            if bid and ask and bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
            else:
                mid_price = last_price
            
            # Extract Greeks (if available from Yahoo)
            delta = contract.get('delta')
            gamma = contract.get('gamma')
            theta = contract.get('theta')
            vega = contract.get('vega')
            rho = contract.get('rho')
            
            # Extract other data
            iv = contract.get('impliedVolatility')
            open_interest = contract.get('openInterest')
            volume = contract.get('volume')
            
            # Calculate Greeks ourselves if Yahoo didn't provide them
            greeks = {
                "delta": round(delta, 4) if delta else None,
                "gamma": round(gamma, 4) if gamma else None,
                "theta": round(theta, 4) if theta else None,
                "vega": round(vega, 4) if vega else None,
                "rho": round(rho, 4) if rho else None,
            }
            
            # If Greeks are missing but we have IV and underlying price, calculate them
            if (not delta or not gamma or not theta) and iv and underlying_price:
                try:
                    analytics = get_option_analytics_service()
                    time_to_exp = analytics.calculate_time_to_expiration(expiration_date)
                    if time_to_exp > 0:
                        calculated = analytics.calculate_greeks(
                            option_type=option_type,
                            strike_price=strike_price,
                            underlying_price=underlying_price,
                            time_to_expiration=time_to_exp,
                            implied_volatility=iv  # Already decimal from Yahoo
                        )
                        # Use calculated values for any that are missing
                        if not greeks["delta"] and calculated.get("delta"):
                            greeks["delta"] = calculated["delta"]
                        if not greeks["gamma"] and calculated.get("gamma"):
                            greeks["gamma"] = calculated["gamma"]
                        if not greeks["theta"] and calculated.get("theta"):
                            greeks["theta"] = calculated["theta"]
                        if not greeks["vega"] and calculated.get("vega"):
                            greeks["vega"] = calculated["vega"]
                        if not greeks["rho"] and calculated.get("rho"):
                            greeks["rho"] = calculated["rho"]
                except Exception as e:
                    logger.warning(f"Error calculating Greeks for {ticker}: {e}")
            
            result = {
                "underlying_ticker": ticker,
                "underlying_price": underlying_price,
                "strike_price": strike_price,
                "expiration_date": expiration_date.isoformat(),
                "option_type": option_type,
                "current_price": mid_price,
                "last_price": last_price,
                "bid": bid if bid else None,
                "ask": ask if ask else None,
                "implied_volatility": round(iv * 100, 2) if iv else None,  # Convert to percentage
                "open_interest": int(open_interest) if open_interest else None,
                "volume": int(volume) if volume and volume > 0 else None,
                "greeks": greeks,
                "fetched_at": datetime.now().isoformat()
            }
            
            logger.debug(
                f"Fetched option price for {ticker} {option_type} "
                f"${strike_price}: ${mid_price}"
            )
            return result
            
        except Exception as e:
            logger.error(f"Error fetching option price for {ticker}: {e}")
            return self._empty_price_response(ticker, None)
    
    def _empty_price_response(
        self, 
        ticker: str, 
        underlying_price: Optional[float]
    ) -> Dict[str, Any]:
        """Return empty price response."""
        return {
            "underlying_ticker": ticker,
            "underlying_price": underlying_price,
            "current_price": None,
            "last_price": None,
            "bid": None,
            "ask": None,
            "implied_volatility": None,
            "open_interest": None,
            "volume": None,
            "greeks": {
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "rho": None,
            },
            "fetched_at": datetime.now().isoformat()
        }
    
    async def get_option_price(
        self,
        underlying_ticker: str,
        expiration_date: date,
        option_type: str,
        strike_price: float
    ) -> Dict[str, Any]:
        """Async wrapper for option price fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_option_price_sync,
            underlying_ticker,
            expiration_date,
            option_type,
            strike_price
        )
    
    def _get_batch_option_prices_sync(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Fetch prices for multiple option positions.
        
        Groups by underlying ticker to minimize API calls.
        
        Args:
            positions: List of dicts with id, underlying_ticker, expiration_date,
                      option_type, strike_price
                      
        Returns:
            Dict mapping position id to price data
        """
        results = {}
        
        # Group by underlying ticker
        by_ticker: Dict[str, List[Dict]] = {}
        for pos in positions:
            ticker = pos['underlying_ticker'].upper()
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(pos)
        
        # Fetch each ticker's option chain once
        for ticker, ticker_positions in by_ticker.items():
            try:
                t = YQTicker(ticker)
                
                # Get underlying price (including after-hours if available)
                quote_data = t.price.get(ticker, {})
                if isinstance(quote_data, str):
                    quote_data = {}
                underlying_price = self._get_current_price(quote_data)
                
                # Get option chain
                option_chain = t.option_chain
                
                if isinstance(option_chain, str) or option_chain is None:
                    # No options data - return empty for all positions
                    for pos in ticker_positions:
                        results[pos['id']] = self._empty_price_response(ticker, underlying_price)
                    continue
                
                if hasattr(option_chain, 'empty') and option_chain.empty:
                    for pos in ticker_positions:
                        results[pos['id']] = self._empty_price_response(ticker, underlying_price)
                    continue
                
                # Reset index
                if hasattr(option_chain, 'reset_index'):
                    option_chain = option_chain.reset_index()
                
                # Parse expiration dates once
                if 'expiration' in option_chain.columns:
                    option_chain['exp_date'] = option_chain['expiration'].apply(
                        lambda x: x.date() if isinstance(x, datetime) else 
                                  datetime.strptime(str(x)[:10], '%Y-%m-%d').date()
                    )
                
                # Find each position's contract
                for pos in ticker_positions:
                    exp_date = pos['expiration_date']
                    if isinstance(exp_date, str):
                        exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                    
                    option_type = pos['option_type']
                    strike = pos['strike_price']
                    
                    # Filter chain
                    filtered = option_chain
                    
                    if 'exp_date' in filtered.columns:
                        filtered = filtered[filtered['exp_date'] == exp_date]
                    
                    if 'optionType' in filtered.columns:
                        type_filter = 'calls' if option_type.lower() == 'call' else 'puts'
                        filtered = filtered[filtered['optionType'] == type_filter]
                    
                    if 'strike' in filtered.columns:
                        filtered = filtered[abs(filtered['strike'] - strike) < 0.01]
                    
                    if filtered.empty:
                        results[pos['id']] = self._empty_price_response(ticker, underlying_price)
                        continue
                    
                    contract = filtered.iloc[0]
                    
                    # Extract data
                    last_price = contract.get('lastPrice')
                    bid = contract.get('bid')
                    ask = contract.get('ask')
                    iv = contract.get('impliedVolatility')
                    
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                    else:
                        mid_price = last_price
                    
                    # Build Greeks dict
                    greeks = {
                        "delta": round(contract.get('delta'), 4) if contract.get('delta') else None,
                        "gamma": round(contract.get('gamma'), 4) if contract.get('gamma') else None,
                        "theta": round(contract.get('theta'), 4) if contract.get('theta') else None,
                        "vega": round(contract.get('vega'), 4) if contract.get('vega') else None,
                        "rho": round(contract.get('rho'), 4) if contract.get('rho') else None,
                    }
                    
                    # Calculate Greeks if Yahoo didn't provide them
                    if (not greeks["delta"] or not greeks["gamma"]) and iv and underlying_price:
                        try:
                            analytics = get_option_analytics_service()
                            time_to_exp = analytics.calculate_time_to_expiration(exp_date)
                            if time_to_exp > 0:
                                calculated = analytics.calculate_greeks(
                                    option_type=option_type,
                                    strike_price=strike,
                                    underlying_price=underlying_price,
                                    time_to_expiration=time_to_exp,
                                    implied_volatility=iv
                                )
                                if not greeks["delta"] and calculated.get("delta"):
                                    greeks["delta"] = calculated["delta"]
                                if not greeks["gamma"] and calculated.get("gamma"):
                                    greeks["gamma"] = calculated["gamma"]
                                if not greeks["theta"] and calculated.get("theta"):
                                    greeks["theta"] = calculated["theta"]
                                if not greeks["vega"] and calculated.get("vega"):
                                    greeks["vega"] = calculated["vega"]
                                if not greeks["rho"] and calculated.get("rho"):
                                    greeks["rho"] = calculated["rho"]
                        except Exception as e:
                            logger.warning(f"Error calculating Greeks for {ticker}: {e}")
                    
                    results[pos['id']] = {
                        "underlying_ticker": ticker,
                        "underlying_price": underlying_price,
                        "strike_price": strike,
                        "expiration_date": exp_date.isoformat(),
                        "option_type": option_type,
                        "current_price": mid_price,
                        "last_price": last_price,
                        "bid": bid if bid else None,
                        "ask": ask if ask else None,
                        "implied_volatility": round(iv * 100, 2) if iv else None,
                        "open_interest": int(contract.get('openInterest', 0)) if contract.get('openInterest') else None,
                        "volume": int(contract.get('volume', 0)) if contract.get('volume') and contract.get('volume') > 0 else None,
                        "greeks": greeks,
                        "fetched_at": datetime.now().isoformat()
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching options for {ticker}: {e}")
                for pos in ticker_positions:
                    results[pos['id']] = self._empty_price_response(ticker, None)
        
        return results
    
    async def get_batch_option_prices(
        self,
        positions: List[Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """Async wrapper for batch option price fetching."""
        if not positions:
            return {}
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_batch_option_prices_sync,
            positions
        )
    
    def _get_available_expirations_sync(self, ticker: str) -> List[str]:
        """Get available expiration dates for a ticker."""
        ticker = ticker.upper()
        
        try:
            t = YQTicker(ticker)
            option_chain = t.option_chain
            
            if isinstance(option_chain, str) or option_chain is None:
                return []
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return []
            
            if hasattr(option_chain, 'reset_index'):
                option_chain = option_chain.reset_index()
            
            if 'expiration' not in option_chain.columns:
                return []
            
            expirations = option_chain['expiration'].unique()
            dates = []
            for exp in expirations:
                if isinstance(exp, datetime):
                    dates.append(exp.strftime('%Y-%m-%d'))
                else:
                    dates.append(str(exp)[:10])
            
            return sorted(set(dates))
            
        except Exception as e:
            logger.error(f"Error fetching expirations for {ticker}: {e}")
            return []
    
    async def get_available_expirations(self, ticker: str) -> List[str]:
        """Get available expiration dates for a ticker."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_available_expirations_sync,
            ticker
        )
    
    def _get_strikes_for_expiration_sync(
        self, 
        ticker: str, 
        expiration: str,
        option_type: Optional[str] = None
    ) -> List[float]:
        """Get available strikes for a specific expiration."""
        ticker = ticker.upper()
        
        try:
            t = YQTicker(ticker)
            option_chain = t.option_chain
            
            if isinstance(option_chain, str) or option_chain is None:
                return []
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return []
            
            if hasattr(option_chain, 'reset_index'):
                option_chain = option_chain.reset_index()
            
            # Filter by expiration
            if 'expiration' in option_chain.columns:
                option_chain['exp_str'] = option_chain['expiration'].apply(
                    lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else str(x)[:10]
                )
                filtered = option_chain[option_chain['exp_str'] == expiration]
            else:
                filtered = option_chain
            
            # Optionally filter by type
            if option_type and 'optionType' in filtered.columns:
                type_filter = 'calls' if option_type.lower() == 'call' else 'puts'
                filtered = filtered[filtered['optionType'] == type_filter]
            
            if 'strike' not in filtered.columns:
                return []
            
            strikes = sorted(filtered['strike'].unique().tolist())
            return strikes
            
        except Exception as e:
            logger.error(f"Error fetching strikes for {ticker}: {e}")
            return []
    
    async def get_strikes_for_expiration(
        self, 
        ticker: str, 
        expiration: str,
        option_type: Optional[str] = None
    ) -> List[float]:
        """Get available strikes for a specific expiration."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_strikes_for_expiration_sync,
            ticker,
            expiration,
            option_type
        )


# Singleton instance
_pricing_service: Optional[OptionPricingService] = None


def get_option_pricing_service() -> OptionPricingService:
    """Get the singleton option pricing service."""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = OptionPricingService()
    return _pricing_service

