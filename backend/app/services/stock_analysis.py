"""
Stock Analysis Service - Fundamentals and Options Data

Provides additional data for stock scoring:
- Fundamental metrics (ROIC, sector, industry)
- Options data (call/put ratio, open interest)
- Sector correlation data

Uses SQLite caching with 15-minute TTL to reduce API calls.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import Session
from yahooquery import Ticker as YQTicker

from ..config import settings
from ..logging_config import get_logger
from ..models import StockAnalysisCache
from .retry import make_yahoo_request

logger = get_logger(__name__)

# Cache TTL in seconds (15 minutes)
ANALYSIS_CACHE_TTL = 900


class StockAnalysisService:
    """
    Service for fetching fundamental and options data.
    
    Uses yahooquery for:
    - Key statistics (ROE, ROIC approximation)
    - Asset profile (sector, industry)
    - Options chains (call/put ratios)
    
    Caches results in SQLite with 15-minute TTL.
    """
    
    def __init__(self, max_workers: Optional[int] = None, database_url: Optional[str] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers or 4)
        url = database_url or settings.database_url_sync
        self._engine = create_engine(url, connect_args={"check_same_thread": False})
    
    # ============================================================================
    # SQLite Cache Methods
    # ============================================================================
    
    def _get_cached_analysis(self, ticker: str) -> Optional[StockAnalysisCache]:
        """Get cached analysis if still valid (within 15 min TTL)."""
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
            result = session.execute(
                select(StockAnalysisCache).where(StockAnalysisCache.ticker == ticker)
            )
            cached = result.scalar_one_or_none()
            
            if cached:
                age_seconds = (datetime.utcnow() - cached.fetched_at).total_seconds()
                if age_seconds < ANALYSIS_CACHE_TTL:
                    logger.debug(f"Cache hit for {ticker} (age: {age_seconds:.0f}s)")
                    return cached
                else:
                    logger.debug(f"Cache expired for {ticker} (age: {age_seconds:.0f}s)")
            
            return None
    
    def _store_analysis_cache(
        self, 
        ticker: str, 
        fundamentals: Dict[str, Any], 
        options: Dict[str, Any]
    ) -> None:
        """Store analysis data in SQLite cache."""
        ticker = ticker.upper()
        
        with Session(self._engine) as session:
            # Check if exists
            existing = session.execute(
                select(StockAnalysisCache).where(StockAnalysisCache.ticker == ticker)
            ).scalar_one_or_none()
            
            if existing:
                # Update existing
                existing.sector = fundamentals.get('sector')
                existing.industry = fundamentals.get('industry')
                existing.roic = fundamentals.get('roic')
                existing.roe = fundamentals.get('roe')
                existing.roa = fundamentals.get('roa')
                existing.profit_margin = fundamentals.get('profit_margin')
                existing.operating_margin = fundamentals.get('operating_margin')
                existing.beta = fundamentals.get('beta')
                existing.market_cap = fundamentals.get('market_cap')
                existing.forward_pe = fundamentals.get('forward_pe')
                existing.dividend_yield = fundamentals.get('dividend_yield')
                existing.next_earnings_date = self._parse_earnings_date(fundamentals.get('next_earnings_date'))
                existing.call_open_interest = options.get('call_open_interest')
                existing.put_open_interest = options.get('put_open_interest')
                existing.call_volume = options.get('call_volume')
                existing.put_volume = options.get('put_volume')
                existing.call_put_ratio_oi = options.get('call_put_ratio_oi')
                existing.call_put_ratio_volume = options.get('call_put_ratio_volume')
                existing.avg_implied_volatility = options.get('avg_implied_volatility')
                existing.iv_percentile = options.get('iv_percentile')
                existing.options_sentiment = options.get('options_sentiment')
                existing.has_options = options.get('has_options', False)
                existing.fetched_at = datetime.utcnow()
            else:
                # Create new
                new_cache = StockAnalysisCache(
                    ticker=ticker,
                    sector=fundamentals.get('sector'),
                    industry=fundamentals.get('industry'),
                    roic=fundamentals.get('roic'),
                    roe=fundamentals.get('roe'),
                    roa=fundamentals.get('roa'),
                    profit_margin=fundamentals.get('profit_margin'),
                    operating_margin=fundamentals.get('operating_margin'),
                    beta=fundamentals.get('beta'),
                    market_cap=fundamentals.get('market_cap'),
                    forward_pe=fundamentals.get('forward_pe'),
                    dividend_yield=fundamentals.get('dividend_yield'),
                    next_earnings_date=self._parse_earnings_date(fundamentals.get('next_earnings_date')),
                    call_open_interest=options.get('call_open_interest'),
                    put_open_interest=options.get('put_open_interest'),
                    call_volume=options.get('call_volume'),
                    put_volume=options.get('put_volume'),
                    call_put_ratio_oi=options.get('call_put_ratio_oi'),
                    call_put_ratio_volume=options.get('call_put_ratio_volume'),
                    avg_implied_volatility=options.get('avg_implied_volatility'),
                    iv_percentile=options.get('iv_percentile'),
                    options_sentiment=options.get('options_sentiment'),
                    has_options=options.get('has_options', False),
                    fetched_at=datetime.utcnow()
                )
                session.add(new_cache)
            
            session.commit()
            logger.debug(f"Stored analysis cache for {ticker}")
    
    def _parse_earnings_date(self, date_value) -> Optional[datetime]:
        """Parse earnings date from various formats."""
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            try:
                # Try ISO format first
                return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try common date format
                    return datetime.strptime(date_value, '%Y-%m-%d')
                except ValueError:
                    return None
        return None
    
    def _cache_to_fundamentals(self, cached: StockAnalysisCache) -> Dict[str, Any]:
        """Convert cached data to fundamentals dict."""
        return {
            "ticker": cached.ticker,
            "sector": cached.sector,
            "industry": cached.industry,
            "roic": cached.roic,
            "roe": cached.roe,
            "roa": cached.roa,
            "profit_margin": cached.profit_margin,
            "operating_margin": cached.operating_margin,
            "beta": cached.beta,
            "market_cap": cached.market_cap,
            "forward_pe": cached.forward_pe,
            "dividend_yield": cached.dividend_yield,
            "next_earnings_date": cached.next_earnings_date.isoformat() if cached.next_earnings_date else None,
        }
    
    def _cache_to_options(self, cached: StockAnalysisCache) -> Dict[str, Any]:
        """Convert cached data to options dict."""
        return {
            "ticker": cached.ticker,
            "call_open_interest": cached.call_open_interest,
            "put_open_interest": cached.put_open_interest,
            "total_open_interest": (
                (cached.call_open_interest or 0) + (cached.put_open_interest or 0)
            ) if cached.call_open_interest is not None else None,
            "call_volume": cached.call_volume,
            "put_volume": cached.put_volume,
            "total_volume": (
                (cached.call_volume or 0) + (cached.put_volume or 0)
            ) if cached.call_volume is not None else None,
            "call_put_ratio_oi": cached.call_put_ratio_oi,
            "call_put_ratio_volume": cached.call_put_ratio_volume,
            "avg_implied_volatility": cached.avg_implied_volatility,
            "iv_percentile": cached.iv_percentile,
            "options_sentiment": cached.options_sentiment,
            "has_options": cached.has_options if cached.has_options is not None else False,
        }
    
    # ============================================================================
    # Fundamental Data
    # ============================================================================
    
    def _get_fundamentals_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a stock.
        
        Checks SQLite cache first (15-min TTL), then fetches from API if needed.
        
        Returns:
            Dict with ROIC, ROE, ROA, sector, industry, profit margins, etc.
        """
        ticker = ticker.upper()
        
        # Check cache first
        cached = self._get_cached_analysis(ticker)
        if cached:
            return self._cache_to_fundamentals(cached)
        
        try:
            # Use single YQTicker instance for all properties
            t = YQTicker(ticker)
            
            # Fetch all needed data from single instance
            # yahooquery batches these internally
            key_stats = make_yahoo_request(
                lambda: t.key_stats.get(ticker, {}),
                description=f"fetch key stats for {ticker}",
                default_value={}
            )
            if isinstance(key_stats, str):
                key_stats = {}
            
            financial_data = make_yahoo_request(
                lambda: t.financial_data.get(ticker, {}),
                description=f"fetch financial data for {ticker}",
                default_value={}
            )
            if isinstance(financial_data, str):
                financial_data = {}
            
            asset_profile = make_yahoo_request(
                lambda: t.asset_profile.get(ticker, {}),
                description=f"fetch asset profile for {ticker}",
                default_value={}
            )
            if isinstance(asset_profile, str):
                asset_profile = {}
            
            summary_detail = make_yahoo_request(
                lambda: t.summary_detail.get(ticker, {}),
                description=f"fetch summary detail for {ticker}",
                default_value={}
            )
            if isinstance(summary_detail, str):
                summary_detail = {}
            
            # Fetch calendar events for earnings date
            calendar_events = make_yahoo_request(
                lambda: t.calendar_events.get(ticker, {}),
                description=f"fetch calendar events for {ticker}",
                default_value={}
            )
            if isinstance(calendar_events, str):
                calendar_events = {}
            
            # Extract next earnings date
            next_earnings_date = None
            earnings_data = calendar_events.get('earnings', {})
            if isinstance(earnings_data, dict):
                earnings_date = earnings_data.get('earningsDate')
                if earnings_date:
                    # Can be a list or single value
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        next_earnings_date = earnings_date[0]
                    elif isinstance(earnings_date, (str, datetime)):
                        next_earnings_date = earnings_date
            
            # Extract values with safe defaults
            roe = financial_data.get('returnOnEquity')
            roa = financial_data.get('returnOnAssets')
            profit_margin = financial_data.get('profitMargins')
            operating_margin = financial_data.get('operatingMargins')
            
            # Approximate ROIC from available data
            roic = None
            if roe is not None and roa is not None:
                roic = (roe + roa) / 2
            elif roe is not None:
                roic = roe * 0.8
            
            beta = key_stats.get('beta') or summary_detail.get('beta')
            market_cap = key_stats.get('marketCap') or summary_detail.get('marketCap')
            forward_pe = key_stats.get('forwardPE') or summary_detail.get('forwardPE')
            dividend_yield = summary_detail.get('dividendYield')
            
            fundamentals = {
                "ticker": ticker,
                "sector": asset_profile.get('sector'),
                "industry": asset_profile.get('industry'),
                "roic": round(roic * 100, 2) if roic else None,
                "roe": round(roe * 100, 2) if roe else None,
                "roa": round(roa * 100, 2) if roa else None,
                "profit_margin": round(profit_margin * 100, 2) if profit_margin else None,
                "operating_margin": round(operating_margin * 100, 2) if operating_margin else None,
                "beta": round(beta, 3) if beta else None,
                "market_cap": market_cap,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
                "next_earnings_date": next_earnings_date,
            }
            
            # Also fetch options data while we're here to store complete cache
            options = self._fetch_options_from_api(t, ticker)
            
            # Store in cache
            self._store_analysis_cache(ticker, fundamentals, options)
            
            logger.debug(f"Fetched fundamentals for {ticker}: sector={fundamentals['sector']}, ROIC={fundamentals['roic']}")
            return fundamentals
            
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
            return {
                "ticker": ticker,
                "sector": None,
                "industry": None,
                "roic": None,
                "roe": None,
                "roa": None,
                "profit_margin": None,
                "operating_margin": None,
                "beta": None,
                "market_cap": None,
                "forward_pe": None,
                "dividend_yield": None,
                "next_earnings_date": None,
            }
    
    async def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Async wrapper for fundamentals fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_fundamentals_sync,
            ticker
        )
    
    # ============================================================================
    # Options Data
    # ============================================================================
    
    def _fetch_options_from_api(self, t: YQTicker, ticker: str) -> Dict[str, Any]:
        """
        Fetch options data from API using existing YQTicker instance.
        
        Args:
            t: Existing YQTicker instance (reused for efficiency)
            ticker: Stock ticker symbol
            
        Returns:
            Options data dict
        """
        try:
            option_chain = make_yahoo_request(
                lambda: t.option_chain,
                description=f"fetch options data for {ticker}",
                default_value=None
            )
            
            if isinstance(option_chain, str) or option_chain is None:
                return self._empty_options_response(ticker)
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return self._empty_options_response(ticker)
            
            if hasattr(option_chain, 'reset_index'):
                option_chain = option_chain.reset_index()
            
            # Filter to near-term options (next 30 days)
            today = datetime.now()
            near_term_cutoff = today + timedelta(days=30)
            
            if 'expiration' in option_chain.columns:
                option_chain['expiration'] = option_chain['expiration'].apply(
                    lambda x: x if isinstance(x, datetime) else datetime.strptime(str(x)[:10], '%Y-%m-%d')
                )
                near_term = option_chain[option_chain['expiration'] <= near_term_cutoff]
                
                if near_term.empty:
                    near_term = option_chain
            else:
                near_term = option_chain
            
            # Separate calls and puts
            if 'optionType' in near_term.columns:
                calls = near_term[near_term['optionType'] == 'calls']
                puts = near_term[near_term['optionType'] == 'puts']
            elif 'contractSymbol' in near_term.columns:
                calls = near_term[near_term['contractSymbol'].str.contains('C', case=False, na=False)]
                puts = near_term[near_term['contractSymbol'].str.contains('P', case=False, na=False)]
            else:
                return self._empty_options_response(ticker)
            
            # Calculate metrics
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            
            call_put_ratio_oi = call_oi / put_oi if put_oi > 0 else None
            call_put_ratio_vol = call_volume / put_volume if put_volume > 0 else None
            
            avg_iv = None
            iv_percentile = None
            if 'impliedVolatility' in near_term.columns:
                iv_values = near_term['impliedVolatility'].dropna()
                if len(iv_values) > 0:
                    avg_iv = iv_values.mean()
                    iv_percentile = min(100, max(0, (avg_iv - 0.2) / 0.6 * 100))
            
            options_sentiment = 'neutral'
            if call_put_ratio_oi:
                if call_put_ratio_oi > 1.5:
                    options_sentiment = 'bullish'
                elif call_put_ratio_oi < 0.7:
                    options_sentiment = 'bearish'
            
            return {
                "ticker": ticker,
                "call_open_interest": int(call_oi),
                "put_open_interest": int(put_oi),
                "total_open_interest": int(call_oi + put_oi),
                "call_volume": int(call_volume),
                "put_volume": int(put_volume),
                "total_volume": int(call_volume + put_volume),
                "call_put_ratio_oi": round(call_put_ratio_oi, 3) if call_put_ratio_oi else None,
                "call_put_ratio_volume": round(call_put_ratio_vol, 3) if call_put_ratio_vol else None,
                "avg_implied_volatility": round(avg_iv * 100, 2) if avg_iv else None,
                "iv_percentile": round(iv_percentile, 1) if iv_percentile else None,
                "options_sentiment": options_sentiment,
                "has_options": True
            }
            
        except Exception as e:
            logger.warning(f"Error fetching options for {ticker}: {e}")
            return self._empty_options_response(ticker)
    
    def _get_options_data_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch options data for a stock.
        
        Checks SQLite cache first (15-min TTL), then fetches from API if needed.
        
        Returns:
            Dict with call/put ratio, total open interest, IV percentile, etc.
        """
        ticker = ticker.upper()
        
        # Check cache first
        cached = self._get_cached_analysis(ticker)
        if cached:
            return self._cache_to_options(cached)
        
        # Need to fetch from API
        try:
            t = YQTicker(ticker)
            options = self._fetch_options_from_api(t, ticker)
            
            # Also fetch fundamentals to store complete cache
            fundamentals = self._fetch_fundamentals_from_api(t, ticker)
            
            # Store in cache
            self._store_analysis_cache(ticker, fundamentals, options)
            
            return options
            
        except Exception as e:
            logger.warning(f"Error fetching options for {ticker}: {e}")
            return self._empty_options_response(ticker)
    
    def _fetch_fundamentals_from_api(self, t: YQTicker, ticker: str) -> Dict[str, Any]:
        """Fetch fundamentals from API using existing YQTicker instance."""
        try:
            key_stats = make_yahoo_request(
                lambda: t.key_stats.get(ticker, {}),
                description=f"fetch key stats for {ticker}",
                default_value={}
            )
            if isinstance(key_stats, str):
                key_stats = {}
            
            financial_data = make_yahoo_request(
                lambda: t.financial_data.get(ticker, {}),
                description=f"fetch financial data for {ticker}",
                default_value={}
            )
            if isinstance(financial_data, str):
                financial_data = {}
            
            asset_profile = make_yahoo_request(
                lambda: t.asset_profile.get(ticker, {}),
                description=f"fetch asset profile for {ticker}",
                default_value={}
            )
            if isinstance(asset_profile, str):
                asset_profile = {}
            
            summary_detail = make_yahoo_request(
                lambda: t.summary_detail.get(ticker, {}),
                description=f"fetch summary detail for {ticker}",
                default_value={}
            )
            if isinstance(summary_detail, str):
                summary_detail = {}
            
            # Fetch calendar events for earnings date
            calendar_events = make_yahoo_request(
                lambda: t.calendar_events.get(ticker, {}),
                description=f"fetch calendar events for {ticker}",
                default_value={}
            )
            if isinstance(calendar_events, str):
                calendar_events = {}
            
            # Extract next earnings date
            next_earnings_date = None
            earnings_data = calendar_events.get('earnings', {})
            if isinstance(earnings_data, dict):
                earnings_date = earnings_data.get('earningsDate')
                if earnings_date:
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        next_earnings_date = earnings_date[0]
                    elif isinstance(earnings_date, (str, datetime)):
                        next_earnings_date = earnings_date
            
            roe = financial_data.get('returnOnEquity')
            roa = financial_data.get('returnOnAssets')
            profit_margin = financial_data.get('profitMargins')
            operating_margin = financial_data.get('operatingMargins')
            
            roic = None
            if roe is not None and roa is not None:
                roic = (roe + roa) / 2
            elif roe is not None:
                roic = roe * 0.8
            
            beta = key_stats.get('beta') or summary_detail.get('beta')
            market_cap = key_stats.get('marketCap') or summary_detail.get('marketCap')
            forward_pe = key_stats.get('forwardPE') or summary_detail.get('forwardPE')
            dividend_yield = summary_detail.get('dividendYield')
            
            return {
                "ticker": ticker,
                "sector": asset_profile.get('sector'),
                "industry": asset_profile.get('industry'),
                "roic": round(roic * 100, 2) if roic else None,
                "roe": round(roe * 100, 2) if roe else None,
                "roa": round(roa * 100, 2) if roa else None,
                "profit_margin": round(profit_margin * 100, 2) if profit_margin else None,
                "operating_margin": round(operating_margin * 100, 2) if operating_margin else None,
                "beta": round(beta, 3) if beta else None,
                "market_cap": market_cap,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
                "next_earnings_date": next_earnings_date,
            }
        except Exception as e:
            logger.warning(f"Error fetching fundamentals for {ticker}: {e}")
            return {
                "ticker": ticker,
                "sector": None,
                "industry": None,
                "roic": None,
                "roe": None,
                "roa": None,
                "profit_margin": None,
                "operating_margin": None,
                "beta": None,
                "market_cap": None,
                "forward_pe": None,
                "dividend_yield": None,
                "next_earnings_date": None,
            }
    
    def _empty_options_response(self, ticker: str) -> Dict[str, Any]:
        """Return empty options response."""
        return {
            "ticker": ticker,
            "call_open_interest": None,
            "put_open_interest": None,
            "total_open_interest": None,
            "call_volume": None,
            "put_volume": None,
            "total_volume": None,
            "call_put_ratio_oi": None,
            "call_put_ratio_volume": None,
            "avg_implied_volatility": None,
            "iv_percentile": None,
            "options_sentiment": None,
            "has_options": False
        }
    
    async def get_options_data(self, ticker: str) -> Dict[str, Any]:
        """Async wrapper for options data fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_options_data_sync,
            ticker
        )
    
    # ============================================================================
    # Combined Analysis Data
    # ============================================================================
    
    def _get_analysis_data_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch all analysis data for a stock in one call.
        
        Checks SQLite cache first (15-min TTL). If cached, returns immediately.
        If not cached, fetches from API and stores in cache.
        """
        ticker = ticker.upper()
        
        # Check cache first
        cached = self._get_cached_analysis(ticker)
        if cached:
            return {
                "ticker": ticker,
                "fundamentals": self._cache_to_fundamentals(cached),
                "options": self._cache_to_options(cached),
                "fetched_at": cached.fetched_at.isoformat(),
                "from_cache": True
            }
        
        # Need to fetch from API
        try:
            t = YQTicker(ticker)
            fundamentals = self._fetch_fundamentals_from_api(t, ticker)
            options = self._fetch_options_from_api(t, ticker)
            
            # Store in cache
            self._store_analysis_cache(ticker, fundamentals, options)
            
            return {
                "ticker": ticker,
                "fundamentals": fundamentals,
                "options": options,
                "fetched_at": datetime.now().isoformat(),
                "from_cache": False
            }
        except Exception as e:
            logger.error(f"Error fetching analysis data for {ticker}: {e}")
            return {
                "ticker": ticker,
                "fundamentals": self._fetch_fundamentals_from_api(YQTicker(ticker), ticker),
                "options": self._empty_options_response(ticker),
                "fetched_at": datetime.now().isoformat(),
                "from_cache": False
            }
    
    async def get_analysis_data(self, ticker: str) -> Dict[str, Any]:
        """Async wrapper for combined analysis data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_analysis_data_sync,
            ticker
        )
    
    # ============================================================================
    # Batch Analysis Methods
    # ============================================================================
    
    def _get_batch_analysis_sync(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch analysis data for multiple tickers in one batch call.
        
        Uses yahooquery's batch capabilities to reduce API calls.
        Checks cache first for each ticker.
        
        Returns:
            Dict mapping ticker to analysis data (fundamentals + options).
        """
        tickers = [t.upper() for t in tickers]
        results = {}
        tickers_needing_fetch = []
        
        # Check cache for each ticker
        for ticker in tickers:
            cached = self._get_cached_analysis(ticker)
            if cached:
                results[ticker] = {
                    "fundamentals": self._cache_to_fundamentals(cached),
                    "options": self._cache_to_options(cached)
                }
            else:
                tickers_needing_fetch.append(ticker)
        
        if not tickers_needing_fetch:
            logger.info(f"Batch analysis: all {len(tickers)} tickers served from cache")
            return results
        
        logger.info(f"Batch analysis: {len(tickers_needing_fetch)}/{len(tickers)} need API fetch")
        
        try:
            # Create single YQTicker with all tickers for batch fetching
            t = YQTicker(tickers_needing_fetch)
            
            # Batch fetch all needed data types
            all_key_stats = make_yahoo_request(
                lambda: t.key_stats,
                description=f"batch key stats for {len(tickers_needing_fetch)} tickers",
                default_value={}
            )
            
            all_financial_data = make_yahoo_request(
                lambda: t.financial_data,
                description=f"batch financial data for {len(tickers_needing_fetch)} tickers",
                default_value={}
            )
            
            all_asset_profiles = make_yahoo_request(
                lambda: t.asset_profile,
                description=f"batch asset profiles for {len(tickers_needing_fetch)} tickers",
                default_value={}
            )
            
            all_summary_details = make_yahoo_request(
                lambda: t.summary_detail,
                description=f"batch summary details for {len(tickers_needing_fetch)} tickers",
                default_value={}
            )
            
            # Process each ticker
            for ticker in tickers_needing_fetch:
                try:
                    key_stats = all_key_stats.get(ticker, {})
                    if isinstance(key_stats, str):
                        key_stats = {}
                    
                    financial_data = all_financial_data.get(ticker, {})
                    if isinstance(financial_data, str):
                        financial_data = {}
                    
                    asset_profile = all_asset_profiles.get(ticker, {})
                    if isinstance(asset_profile, str):
                        asset_profile = {}
                    
                    summary_detail = all_summary_details.get(ticker, {})
                    if isinstance(summary_detail, str):
                        summary_detail = {}
                    
                    # Extract fundamental values
                    roe = financial_data.get('returnOnEquity')
                    roa = financial_data.get('returnOnAssets')
                    profit_margin = financial_data.get('profitMargins')
                    operating_margin = financial_data.get('operatingMargins')
                    
                    roic = None
                    if roe is not None and roa is not None:
                        roic = (roe + roa) / 2
                    elif roe is not None:
                        roic = roe * 0.8
                    
                    beta = key_stats.get('beta') or summary_detail.get('beta')
                    market_cap = key_stats.get('marketCap') or summary_detail.get('marketCap')
                    forward_pe = key_stats.get('forwardPE') or summary_detail.get('forwardPE')
                    dividend_yield = summary_detail.get('dividendYield')
                    
                    fundamentals = {
                        "ticker": ticker,
                        "sector": asset_profile.get('sector'),
                        "industry": asset_profile.get('industry'),
                        "roic": round(roic * 100, 2) if roic else None,
                        "roe": round(roe * 100, 2) if roe else None,
                        "roa": round(roa * 100, 2) if roa else None,
                        "profit_margin": round(profit_margin * 100, 2) if profit_margin else None,
                        "operating_margin": round(operating_margin * 100, 2) if operating_margin else None,
                        "beta": round(beta, 3) if beta else None,
                        "market_cap": market_cap,
                        "forward_pe": round(forward_pe, 2) if forward_pe else None,
                        "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
                    }
                    
                    # Fetch options data for this ticker
                    # This is important for showing options actions in analysis
                    options = self._get_options_data_sync(ticker)
                    
                    # Store in cache
                    self._store_analysis_cache(ticker, fundamentals, options)
                    
                    results[ticker] = {
                        "fundamentals": fundamentals,
                        "options": options
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing batch analysis for {ticker}: {e}")
                    results[ticker] = {
                        "fundamentals": self._empty_fundamentals_response(ticker),
                        "options": self._empty_options_response(ticker)
                    }
                    
        except Exception as e:
            logger.error(f"Batch analysis fetch failed: {e}")
            # Return empty responses for failed tickers
            for ticker in tickers_needing_fetch:
                if ticker not in results:
                    results[ticker] = {
                        "fundamentals": self._empty_fundamentals_response(ticker),
                        "options": self._empty_options_response(ticker)
                    }
        
        return results
    
    def _empty_fundamentals_response(self, ticker: str) -> Dict[str, Any]:
        """Return empty fundamentals response."""
        return {
            "ticker": ticker,
            "sector": None,
            "industry": None,
            "roic": None,
            "roe": None,
            "roa": None,
            "profit_margin": None,
            "operating_margin": None,
            "beta": None,
            "market_cap": None,
            "forward_pe": None,
            "dividend_yield": None,
            "next_earnings_date": None,
        }
    
    async def get_batch_analysis(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Async wrapper for batch analysis fetching."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_batch_analysis_sync,
            tickers
        )
    
    # ============================================================================
    # Sector Data (for correlation)
    # ============================================================================
    
    # Mapping of sectors to their ETF proxies
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financial Services": "XLF",
        "Consumer Cyclical": "XLY",
        "Consumer Defensive": "XLP",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB",
        "Communication Services": "XLC",
        "Aerospace & Defense": "ITA",
    }
    
    def _get_sector_correlation_sync(
        self, 
        ticker: str, 
        days: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate correlation between stock and its sector ETF.
        
        Uses cached sector/industry data when available.
        """
        ticker = ticker.upper()
        
        try:
            # Try to get sector from cache first
            cached = self._get_cached_analysis(ticker)
            if cached and cached.sector:
                sector = cached.sector
                industry = cached.industry
            else:
                # Fetch from API
                t = YQTicker(ticker)
                asset_profile = make_yahoo_request(
                    lambda: t.asset_profile.get(ticker, {}),
                    description=f"fetch sector for {ticker}",
                    default_value={}
                )
                
                if isinstance(asset_profile, str):
                    return {"ticker": ticker, "sector": None, "sector_etf": None, "correlation": None}
                
                sector = asset_profile.get('sector')
                industry = asset_profile.get('industry')
            
            # Find the appropriate sector ETF
            sector_etf = self.SECTOR_ETFS.get(sector)
            
            # Special case for defense stocks
            if industry and 'defense' in industry.lower():
                sector_etf = self.SECTOR_ETFS.get("Aerospace & Defense", sector_etf)
            
            if not sector_etf:
                return {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "sector_etf": None,
                    "correlation": None,
                    "beta_to_sector": None
                }
            
            # Fetch historical data for both
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 10)
            
            combined = YQTicker([ticker, sector_etf])
            hist = make_yahoo_request(
                lambda: combined.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adj_ohlc=True
                ),
                description=f"fetch correlation history for {ticker}",
                default_value=None
            )
            
            if hist is None or isinstance(hist, str) or hist.empty:
                return {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "sector_etf": sector_etf,
                    "correlation": None,
                    "beta_to_sector": None
                }
            
            hist = hist.reset_index()
            
            ticker_prices = hist[hist['symbol'] == ticker]['close'].values
            etf_prices = hist[hist['symbol'] == sector_etf]['close'].values
            
            min_len = min(len(ticker_prices), len(etf_prices))
            if min_len < 10:
                return {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "sector_etf": sector_etf,
                    "correlation": None,
                    "beta_to_sector": None
                }
            
            ticker_prices = ticker_prices[-min_len:]
            etf_prices = etf_prices[-min_len:]
            
            # Calculate returns
            ticker_returns = []
            etf_returns = []
            for i in range(1, len(ticker_prices)):
                if ticker_prices[i-1] > 0 and etf_prices[i-1] > 0:
                    ticker_returns.append(
                        (ticker_prices[i] - ticker_prices[i-1]) / ticker_prices[i-1]
                    )
                    etf_returns.append(
                        (etf_prices[i] - etf_prices[i-1]) / etf_prices[i-1]
                    )
            
            if len(ticker_returns) < 10:
                return {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "sector_etf": sector_etf,
                    "correlation": None,
                    "beta_to_sector": None
                }
            
            import numpy as np
            correlation = np.corrcoef(ticker_returns, etf_returns)[0, 1]
            
            etf_variance = np.var(etf_returns)
            if etf_variance > 0:
                covariance = np.cov(ticker_returns, etf_returns)[0, 1]
                beta_to_sector = covariance / etf_variance
            else:
                beta_to_sector = None
            
            return {
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "sector_etf": sector_etf,
                "correlation": round(correlation, 3) if correlation else None,
                "beta_to_sector": round(beta_to_sector, 3) if beta_to_sector else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculating sector correlation for {ticker}: {e}")
            return {
                "ticker": ticker,
                "sector": None,
                "industry": None,
                "sector_etf": None,
                "correlation": None,
                "beta_to_sector": None
            }
    
    async def get_sector_correlation(self, ticker: str, days: int = 60) -> Dict[str, Any]:
        """Async wrapper for sector correlation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._get_sector_correlation_sync,
            ticker,
            days
        )


# Singleton instance
_analysis_service: Optional[StockAnalysisService] = None


def get_stock_analysis_service() -> StockAnalysisService:
    """Get the singleton stock analysis service."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = StockAnalysisService()
    return _analysis_service
