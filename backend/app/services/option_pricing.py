"""
Option Pricing Service - Fetches real-time option prices from Yahoo Finance.

Uses yfinance (primary) and yahooquery (fallback) to fetch option chain data
and match user positions to current market prices.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from typing import Dict, Any, Optional, List
import time

import yfinance as yf
from yahooquery import Ticker as YQTicker

from ..logging_config import get_logger
from .option_analytics import get_option_analytics_service
from .retry import make_yahoo_request

logger = get_logger(__name__)


class OptionPricingService:
    """
    Service for fetching real-time option prices.
    
    Uses yfinance to fetch option chains and extract pricing data
    for specific contracts based on strike/expiration/type.
    
    Includes caching to avoid repeated API calls during rate limiting.
    """
    
    # Class-level cache for option chains (shared across instances)
    _chain_cache: Dict[str, Dict[str, Any]] = {}
    _cache_ttl = 300  # Cache for 5 minutes
    
    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers or 4)
    
    def _get_cached_chain(self, ticker: str, exp_str: str) -> Optional[Any]:
        """Get cached option chain if still valid."""
        cache_key = f"{ticker}_{exp_str}"
        if cache_key in self._chain_cache:
            cached = self._chain_cache[cache_key]
            if time.time() - cached['timestamp'] < self._cache_ttl:
                logger.debug(f"Using cached chain for {ticker} {exp_str}")
                return cached['data']
        return None
    
    def _set_cached_chain(self, ticker: str, exp_str: str, chain: Any):
        """Cache option chain data."""
        cache_key = f"{ticker}_{exp_str}"
        self._chain_cache[cache_key] = {
            'data': chain,
            'timestamp': time.time()
        }
    
    def _get_cached_expirations(self, ticker: str) -> Optional[tuple]:
        """Get cached expirations if still valid."""
        cache_key = f"{ticker}_expirations"
        if cache_key in self._chain_cache:
            cached = self._chain_cache[cache_key]
            if time.time() - cached['timestamp'] < self._cache_ttl:
                logger.debug(f"Using cached expirations for {ticker}")
                return cached['data']
        return None
    
    def _set_cached_expirations(self, ticker: str, expirations: tuple):
        """Cache expirations data."""
        cache_key = f"{ticker}_expirations"
        self._chain_cache[cache_key] = {
            'data': expirations,
            'timestamp': time.time()
        }
    
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
            
            # Get the underlying stock price (including after-hours if available) with retry
            quote_data = make_yahoo_request(
                lambda: t.price.get(ticker, {}),
                description=f"fetch option underlying price for {ticker}",
                default_value={}
            )
            if isinstance(quote_data, str):
                quote_data = {}
            underlying_price = self._get_current_price(quote_data)
            logger.debug(f"Fetching option for {ticker}: underlying_price={underlying_price}")
            
            # Get option chain with retry
            option_chain = make_yahoo_request(
                lambda: t.option_chain,
                description=f"fetch option chain for {ticker}",
                default_value=None
            )
            
            if isinstance(option_chain, str) or option_chain is None:
                logger.warning(f"No options chain available for {ticker}")
                return self._empty_price_response(
                    ticker, underlying_price,
                    expiration_date=expiration_date,
                    option_type=option_type,
                    strike_price=strike_price
                )
            
            logger.debug(f"Option chain columns: {option_chain.columns.tolist() if hasattr(option_chain, 'columns') else 'N/A'}")
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return self._empty_price_response(
                    ticker, underlying_price,
                    expiration_date=expiration_date,
                    option_type=option_type,
                    strike_price=strike_price
                )
            
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
                logger.warning(
                    f"No matching contract for {ticker} {option_type} "
                    f"${strike_price} exp {expiration_date}"
                )
                return self._empty_price_response(
                    ticker, underlying_price,
                    expiration_date=expiration_date,
                    option_type=option_type,
                    strike_price=strike_price
                )
            
            # Get the first matching row
            contract = filtered.iloc[0]
            logger.debug(f"Found contract for {ticker}: IV={contract.get('impliedVolatility')}, delta={contract.get('delta')}")
            
            # Extract price data
            last_price = contract.get('lastPrice')
            bid = contract.get('bid')
            ask = contract.get('ask')
            
            # Calculate mid price if available
            if bid and ask and bid > 0 and ask > 0:
                mid_price = (bid + ask) / 2
            else:
                mid_price = last_price
            
            # Helper to safely extract numeric values from pandas (handles NaN)
            def safe_float(val):
                if val is None:
                    return None
                try:
                    import math
                    f = float(val)
                    return None if math.isnan(f) else f
                except (TypeError, ValueError):
                    return None
            
            # Extract Greeks (if available from Yahoo)
            delta = safe_float(contract.get('delta'))
            gamma = safe_float(contract.get('gamma'))
            theta = safe_float(contract.get('theta'))
            vega = safe_float(contract.get('vega'))
            rho = safe_float(contract.get('rho'))
            
            # Extract other data
            iv = safe_float(contract.get('impliedVolatility'))
            open_interest = contract.get('openInterest')
            volume = contract.get('volume')
            
            # Build initial Greeks dict from Yahoo data
            greeks = {
                "delta": round(delta, 4) if delta is not None else None,
                "gamma": round(gamma, 4) if gamma is not None else None,
                "theta": round(theta, 4) if theta is not None else None,
                "vega": round(vega, 4) if vega is not None else None,
                "rho": round(rho, 4) if rho is not None else None,
            }
            
            # Calculate Greeks ourselves if Yahoo didn't provide them
            # We need IV and underlying price to calculate
            has_missing_greeks = greeks["delta"] is None or greeks["gamma"] is None or greeks["theta"] is None
            
            if has_missing_greeks and iv is not None and iv > 0 and underlying_price is not None:
                try:
                    analytics = get_option_analytics_service()
                    time_to_exp = analytics.calculate_time_to_expiration(expiration_date)
                    
                    if time_to_exp > 0:
                        logger.debug(
                            f"Calculating Greeks for {ticker} {option_type} ${strike_price}: "
                            f"S={underlying_price}, T={time_to_exp:.4f}, IV={iv:.4f}"
                        )
                        calculated = analytics.calculate_greeks(
                            option_type=option_type,
                            strike_price=strike_price,
                            underlying_price=underlying_price,
                            time_to_expiration=time_to_exp,
                            implied_volatility=iv  # Already decimal from Yahoo (e.g., 0.30 for 30%)
                        )
                        logger.debug(f"Calculated Greeks: {calculated}")
                        
                        # Use calculated values for any that are missing
                        if greeks["delta"] is None and calculated.get("delta") is not None:
                            greeks["delta"] = calculated["delta"]
                        if greeks["gamma"] is None and calculated.get("gamma") is not None:
                            greeks["gamma"] = calculated["gamma"]
                        if greeks["theta"] is None and calculated.get("theta") is not None:
                            greeks["theta"] = calculated["theta"]
                        if greeks["vega"] is None and calculated.get("vega") is not None:
                            greeks["vega"] = calculated["vega"]
                        if greeks["rho"] is None and calculated.get("rho") is not None:
                            greeks["rho"] = calculated["rho"]
                    else:
                        logger.debug(f"Skipping Greeks calculation - option expired or expiring today")
                except Exception as e:
                    logger.warning(f"Error calculating Greeks for {ticker}: {e}", exc_info=True)
            
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
        underlying_price: Optional[float],
        expiration_date: Optional[date] = None,
        option_type: Optional[str] = None,
        strike_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return empty price response when data cannot be fetched."""
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
        # Note: Removed fixed 2-second delay - exponential backoff in make_yahoo_request
        # handles rate limiting more efficiently
        for ticker, ticker_positions in by_ticker.items():
            logger.info(f"Fetching option chain for {ticker} ({len(ticker_positions)} positions)")
            try:
                # Note: Don't pass session - yfinance now uses curl_cffi which is incompatible with requests_cache
                yf_ticker = yf.Ticker(ticker)
                
                # Get underlying price with retry
                underlying_price = make_yahoo_request(
                    lambda: yf_ticker.fast_info.get('lastPrice') or yf_ticker.fast_info.get('regularMarketPrice'),
                    description=f"fetch fast_info for {ticker}",
                    default_value=None
                )
                
                # Fallback to history for price
                if underlying_price is None:
                    hist = make_yahoo_request(
                        lambda: yf_ticker.history(period="1d"),
                        description=f"fetch history for {ticker}",
                        default_value=None
                    )
                    if hist is not None and not hist.empty:
                        underlying_price = hist['Close'].iloc[-1]
                
                logger.info(f"Underlying price for {ticker}: {underlying_price}")
                
                # Check cache first for expirations
                expirations = self._get_cached_expirations(ticker)
                
                if not expirations:
                    # Get available expiration dates with retry
                    expirations = make_yahoo_request(
                        lambda: yf_ticker.options,
                        description=f"fetch expirations for {ticker}",
                        default_value=None
                    )
                    if expirations:
                        logger.info(f"Got {len(expirations)} expirations for {ticker}")
                        self._set_cached_expirations(ticker, expirations)
                
                if not expirations:
                    logger.warning(f"No option expirations for {ticker}")
                    for pos in ticker_positions:
                        exp_date = pos['expiration_date']
                        if isinstance(exp_date, str):
                            exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        results[pos['id']] = self._empty_price_response(
                            ticker, underlying_price,
                            expiration_date=exp_date,
                            option_type=pos['option_type'],
                            strike_price=pos['strike_price']
                        )
                    continue
                
                # Collect required expiration dates from positions
                required_expirations = set()
                for pos in ticker_positions:
                    exp_date = pos['expiration_date']
                    if isinstance(exp_date, str):
                        required_expirations.add(exp_date)
                    else:
                        required_expirations.add(exp_date.strftime('%Y-%m-%d'))
                
                # Fetch option chains for required dates with caching and retry
                import pandas as pd
                option_chain_parts = []
                for exp_str in required_expirations:
                    if exp_str in expirations:
                        # Check cache first
                        chain = self._get_cached_chain(ticker, exp_str)
                        
                        if chain is None:
                            chain = make_yahoo_request(
                                lambda exp=exp_str: yf_ticker.option_chain(exp),
                                description=f"fetch chain for {ticker} {exp_str}",
                                default_value=None
                            )
                            if chain is not None:
                                self._set_cached_chain(ticker, exp_str, chain)
                        
                        if chain is not None:
                            # Combine calls and puts
                            calls = chain.calls.copy()
                            calls['optionType'] = 'calls'
                            calls['exp_date'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
                            puts = chain.puts.copy()
                            puts['optionType'] = 'puts'
                            puts['exp_date'] = datetime.strptime(exp_str, '%Y-%m-%d').date()
                            option_chain_parts.append(calls)
                            option_chain_parts.append(puts)
                            logger.info(f"Fetched {len(calls) + len(puts)} contracts for {ticker} {exp_str}")
                    else:
                        logger.warning(f"Expiration {exp_str} not available for {ticker}. Available: {expirations[:5]}...")
                
                if not option_chain_parts:
                    logger.warning(f"No matching option chains for {ticker}")
                    for pos in ticker_positions:
                        exp_date = pos['expiration_date']
                        if isinstance(exp_date, str):
                            exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                        results[pos['id']] = self._empty_price_response(
                            ticker, underlying_price,
                            expiration_date=exp_date,
                            option_type=pos['option_type'],
                            strike_price=pos['strike_price']
                        )
                    continue
                
                option_chain = pd.concat(option_chain_parts, ignore_index=True)
                logger.debug(f"Option chain columns for {ticker}: {option_chain.columns.tolist()}")
                
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
                    
                    logger.debug(f"Looking for {ticker} {option_type} ${strike} exp {exp_date}")
                    
                    # Filter chain
                    filtered = option_chain
                    initial_count = len(filtered)
                    
                    if 'exp_date' in filtered.columns:
                        filtered = filtered[filtered['exp_date'] == exp_date]
                        logger.debug(f"  After exp filter: {len(filtered)} rows (was {initial_count})")
                    
                    if 'optionType' in filtered.columns:
                        type_filter = 'calls' if option_type.lower() == 'call' else 'puts'
                        count_before = len(filtered)
                        filtered = filtered[filtered['optionType'] == type_filter]
                        logger.debug(f"  After type filter ({type_filter}): {len(filtered)} rows (was {count_before})")
                    
                    if 'strike' in filtered.columns:
                        count_before = len(filtered)
                        filtered = filtered[abs(filtered['strike'] - strike) < 0.01]
                        logger.debug(f"  After strike filter (${strike}): {len(filtered)} rows (was {count_before})")
                    
                    if filtered.empty:
                        logger.warning(f"No matching contract for {ticker} {option_type} ${strike} exp {exp_date}")
                        results[pos['id']] = self._empty_price_response(
                            ticker, underlying_price,
                            expiration_date=exp_date,
                            option_type=option_type,
                            strike_price=strike
                        )
                        continue
                    
                    contract = filtered.iloc[0]
                    
                    # Helper to safely extract numeric values from pandas (handles NaN)
                    def safe_float(val):
                        if val is None:
                            return None
                        try:
                            import math
                            f = float(val)
                            return None if math.isnan(f) else f
                        except (TypeError, ValueError):
                            return None
                    
                    # Extract data
                    last_price = safe_float(contract.get('lastPrice'))
                    bid = safe_float(contract.get('bid'))
                    ask = safe_float(contract.get('ask'))
                    iv = safe_float(contract.get('impliedVolatility'))
                    
                    if bid and ask and bid > 0 and ask > 0:
                        mid_price = (bid + ask) / 2
                    else:
                        mid_price = last_price
                    
                    # Extract Greeks from Yahoo
                    delta = safe_float(contract.get('delta'))
                    gamma = safe_float(contract.get('gamma'))
                    theta = safe_float(contract.get('theta'))
                    vega = safe_float(contract.get('vega'))
                    rho = safe_float(contract.get('rho'))
                    
                    # Build Greeks dict
                    greeks = {
                        "delta": round(delta, 4) if delta is not None else None,
                        "gamma": round(gamma, 4) if gamma is not None else None,
                        "theta": round(theta, 4) if theta is not None else None,
                        "vega": round(vega, 4) if vega is not None else None,
                        "rho": round(rho, 4) if rho is not None else None,
                    }
                    
                    # Calculate Greeks if Yahoo didn't provide them
                    has_missing = greeks["delta"] is None or greeks["gamma"] is None or greeks["theta"] is None
                    
                    if has_missing and iv is not None and iv > 0 and underlying_price is not None:
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
                                if greeks["delta"] is None and calculated.get("delta") is not None:
                                    greeks["delta"] = calculated["delta"]
                                if greeks["gamma"] is None and calculated.get("gamma") is not None:
                                    greeks["gamma"] = calculated["gamma"]
                                if greeks["theta"] is None and calculated.get("theta") is not None:
                                    greeks["theta"] = calculated["theta"]
                                if greeks["vega"] is None and calculated.get("vega") is not None:
                                    greeks["vega"] = calculated["vega"]
                                if greeks["rho"] is None and calculated.get("rho") is not None:
                                    greeks["rho"] = calculated["rho"]
                        except Exception as e:
                            logger.warning(f"Error calculating Greeks for {ticker}: {e}", exc_info=True)
                    
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
            option_chain = make_yahoo_request(
                lambda: t.option_chain,
                description=f"fetch expirations for {ticker}",
                default_value=None
            )
            
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
            option_chain = make_yahoo_request(
                lambda: t.option_chain,
                description=f"fetch strikes for {ticker}",
                default_value=None
            )
            
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

