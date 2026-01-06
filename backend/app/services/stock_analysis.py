"""
Stock Analysis Service - Fundamentals and Options Data

Provides additional data for stock scoring:
- Fundamental metrics (ROIC, sector, industry)
- Options data (call/put ratio, open interest)
- Sector correlation data
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from yahooquery import Ticker as YQTicker

from ..logging_config import get_logger

logger = get_logger(__name__)


class StockAnalysisService:
    """
    Service for fetching fundamental and options data.
    
    Uses yahooquery for:
    - Key statistics (ROE, ROIC approximation)
    - Asset profile (sector, industry)
    - Options chains (call/put ratios)
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers or 4)
    
    # ============================================================================
    # Fundamental Data
    # ============================================================================
    
    def _get_fundamentals_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch fundamental data for a stock.
        
        Returns:
            Dict with ROIC, ROE, ROA, sector, industry, profit margins, etc.
        """
        ticker = ticker.upper()
        
        try:
            t = YQTicker(ticker)
            
            # Get key statistics
            key_stats = t.key_stats.get(ticker, {})
            if isinstance(key_stats, str):
                key_stats = {}
            
            # Get financial data for ROIC calculation
            # ROIC = NOPAT / Invested Capital
            # We'll use approximations from available data
            financial_data = t.financial_data.get(ticker, {})
            if isinstance(financial_data, str):
                financial_data = {}
            
            # Get asset profile for sector/industry
            asset_profile = t.asset_profile.get(ticker, {})
            if isinstance(asset_profile, str):
                asset_profile = {}
            
            # Get summary detail for additional metrics
            summary_detail = t.summary_detail.get(ticker, {})
            if isinstance(summary_detail, str):
                summary_detail = {}
            
            # Extract values with safe defaults
            roe = financial_data.get('returnOnEquity')
            roa = financial_data.get('returnOnAssets')
            profit_margin = financial_data.get('profitMargins')
            operating_margin = financial_data.get('operatingMargins')
            
            # Approximate ROIC from available data
            # If we have operating income and total assets, we can estimate
            # Otherwise, use ROE as a proxy (less accurate but available)
            roic = None
            if roe is not None and roa is not None:
                # ROIC is typically between ROE and ROA
                # This is a rough approximation
                roic = (roe + roa) / 2
            elif roe is not None:
                roic = roe * 0.8  # Conservative estimate
            
            # Get beta
            beta = key_stats.get('beta') or summary_detail.get('beta')
            
            # Get market cap
            market_cap = key_stats.get('marketCap') or summary_detail.get('marketCap')
            
            # Get forward P/E
            forward_pe = key_stats.get('forwardPE') or summary_detail.get('forwardPE')
            
            # Get dividend yield
            dividend_yield = summary_detail.get('dividendYield')
            
            result = {
                "ticker": ticker,
                "sector": asset_profile.get('sector'),
                "industry": asset_profile.get('industry'),
                "roic": round(roic * 100, 2) if roic else None,  # As percentage
                "roe": round(roe * 100, 2) if roe else None,
                "roa": round(roa * 100, 2) if roa else None,
                "profit_margin": round(profit_margin * 100, 2) if profit_margin else None,
                "operating_margin": round(operating_margin * 100, 2) if operating_margin else None,
                "beta": round(beta, 3) if beta else None,
                "market_cap": market_cap,
                "forward_pe": round(forward_pe, 2) if forward_pe else None,
                "dividend_yield": round(dividend_yield * 100, 2) if dividend_yield else None,
            }
            
            logger.debug(f"Fetched fundamentals for {ticker}: sector={result['sector']}, ROIC={result['roic']}")
            return result
            
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
    
    def _get_options_data_sync(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch options data for a stock.
        
        Returns:
            Dict with call/put ratio, total open interest, IV percentile, etc.
        """
        ticker = ticker.upper()
        
        try:
            t = YQTicker(ticker)
            
            # Get option chain - this returns all available expiration dates
            option_chain = t.option_chain
            
            if isinstance(option_chain, str) or option_chain is None:
                # No options data available
                return self._empty_options_response(ticker)
            
            if hasattr(option_chain, 'empty') and option_chain.empty:
                return self._empty_options_response(ticker)
            
            # Reset index if it's a multi-index DataFrame
            if hasattr(option_chain, 'reset_index'):
                option_chain = option_chain.reset_index()
            
            # Filter to near-term options (next 30 days) for more relevant signals
            today = datetime.now()
            near_term_cutoff = today + timedelta(days=30)
            
            # Check if expiration column exists
            if 'expiration' in option_chain.columns:
                # Filter to near-term expirations
                option_chain['expiration'] = option_chain['expiration'].apply(
                    lambda x: x if isinstance(x, datetime) else datetime.strptime(str(x)[:10], '%Y-%m-%d')
                )
                near_term = option_chain[option_chain['expiration'] <= near_term_cutoff]
                
                if near_term.empty:
                    # Fall back to all options if no near-term
                    near_term = option_chain
            else:
                near_term = option_chain
            
            # Separate calls and puts
            if 'optionType' in near_term.columns:
                calls = near_term[near_term['optionType'] == 'calls']
                puts = near_term[near_term['optionType'] == 'puts']
            elif 'contractSymbol' in near_term.columns:
                # Infer from contract symbol (typically contains 'C' or 'P')
                calls = near_term[near_term['contractSymbol'].str.contains('C', case=False, na=False)]
                puts = near_term[near_term['contractSymbol'].str.contains('P', case=False, na=False)]
            else:
                return self._empty_options_response(ticker)
            
            # Calculate open interest totals
            call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
            total_oi = call_oi + put_oi
            
            # Calculate volume totals
            call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0
            total_volume = call_volume + put_volume
            
            # Calculate call/put ratio (using open interest)
            call_put_ratio_oi = call_oi / put_oi if put_oi > 0 else None
            
            # Calculate call/put ratio (using volume)
            call_put_ratio_vol = call_volume / put_volume if put_volume > 0 else None
            
            # Get implied volatility stats
            if 'impliedVolatility' in near_term.columns:
                iv_values = near_term['impliedVolatility'].dropna()
                if len(iv_values) > 0:
                    avg_iv = iv_values.mean()
                    # Approximate IV percentile by comparing to a typical range
                    # This is a rough estimate - proper IV percentile requires historical IV data
                    iv_percentile = min(100, max(0, (avg_iv - 0.2) / 0.6 * 100))
                else:
                    avg_iv = None
                    iv_percentile = None
            else:
                avg_iv = None
                iv_percentile = None
            
            # Determine if bullish or bearish based on call/put ratio
            options_sentiment = 'neutral'
            if call_put_ratio_oi:
                if call_put_ratio_oi > 1.5:
                    options_sentiment = 'bullish'
                elif call_put_ratio_oi < 0.7:
                    options_sentiment = 'bearish'
            
            result = {
                "ticker": ticker,
                "call_open_interest": int(call_oi),
                "put_open_interest": int(put_oi),
                "total_open_interest": int(total_oi),
                "call_volume": int(call_volume),
                "put_volume": int(put_volume),
                "total_volume": int(total_volume),
                "call_put_ratio_oi": round(call_put_ratio_oi, 3) if call_put_ratio_oi else None,
                "call_put_ratio_volume": round(call_put_ratio_vol, 3) if call_put_ratio_vol else None,
                "avg_implied_volatility": round(avg_iv * 100, 2) if avg_iv else None,
                "iv_percentile": round(iv_percentile, 1) if iv_percentile else None,
                "options_sentiment": options_sentiment,
                "has_options": True
            }
            
            logger.debug(
                f"Fetched options for {ticker}: C/P ratio={result['call_put_ratio_oi']}, "
                f"sentiment={options_sentiment}"
            )
            return result
            
        except Exception as e:
            logger.warning(f"Error fetching options for {ticker}: {e}")
            return self._empty_options_response(ticker)
    
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
        
        Combines fundamentals and options data.
        """
        fundamentals = self._get_fundamentals_sync(ticker)
        options = self._get_options_data_sync(ticker)
        
        return {
            "ticker": ticker,
            "fundamentals": fundamentals,
            "options": options,
            "fetched_at": datetime.now().isoformat()
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
        # Defense-specific for geopolitical analysis
        "Aerospace & Defense": "ITA",
    }
    
    def _get_sector_correlation_sync(
        self, 
        ticker: str, 
        days: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate correlation between stock and its sector ETF.
        
        This helps identify if a stock moves with or against its sector.
        """
        ticker = ticker.upper()
        
        try:
            # First get the stock's sector
            t = YQTicker(ticker)
            asset_profile = t.asset_profile.get(ticker, {})
            
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
            start_date = end_date - timedelta(days=days + 10)  # Buffer for non-trading days
            
            combined = YQTicker([ticker, sector_etf])
            hist = combined.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adj_ohlc=True
            )
            
            if isinstance(hist, str) or hist.empty:
                return {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "sector_etf": sector_etf,
                    "correlation": None,
                    "beta_to_sector": None
                }
            
            # Reset index and pivot to get price series
            hist = hist.reset_index()
            
            # Get closing prices for each
            ticker_prices = hist[hist['symbol'] == ticker]['close'].values
            etf_prices = hist[hist['symbol'] == sector_etf]['close'].values
            
            # Ensure same length
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
            
            # Calculate correlation
            import numpy as np
            correlation = np.corrcoef(ticker_returns, etf_returns)[0, 1]
            
            # Calculate beta to sector
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

