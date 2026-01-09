"""
Market Regime Classification Engine (6-State Model)

Final-grade regime classifier using 3-factor logic:
1. Trend (SMA200 + SMA50 alignment)
2. Volatility (Bollinger BandWidth Percentile)
3. Momentum (ADX Strength)

Regime States:
- BULL_QUIET: Trending up, low volatility (Buy Calls, Trend Following)
- BULL_VOLATILE: Trending up, high volatility (Sell Puts, Dip Buying)
- BEAR_QUIET: Trending down, low volatility (Slow Bleed - Avoid)
- BEAR_VOLATILE: Trending down, high volatility (Long Puts, Crash Mode)
- NEUTRAL_CHOP: Flat, low volatility (Mean Reversion, Iron Condors)
- NEUTRAL_VOLATILE: Flat, high volatility (Breakout Imminent - Wait)
"""

import numpy as np
import pandas as pd
from typing import Optional
from enum import Enum


class MarketRegime(str, Enum):
    """6-State Market Regime Classification."""
    BULL_QUIET = "BULL_QUIET"
    BULL_VOLATILE = "BULL_VOLATILE"
    BEAR_QUIET = "BEAR_QUIET"
    BEAR_VOLATILE = "BEAR_VOLATILE"
    NEUTRAL_CHOP = "NEUTRAL_CHOP"
    NEUTRAL_VOLATILE = "NEUTRAL_VOLATILE"


# Regime strategy recommendations
REGIME_STRATEGIES = {
    MarketRegime.BULL_QUIET: {
        "description": "Steady uptrend with low volatility",
        "optimal_strategies": ["Buy Calls", "Trend Following", "Buy Shares"],
        "avoid": ["Shorting", "Put Buying"],
        "mean_reversion_safe": False,  # Don't fight the trend
    },
    MarketRegime.BULL_VOLATILE: {
        "description": "Uptrend with shakeouts",
        "optimal_strategies": ["Sell Puts (CSP)", "Buy Dips", "Covered Calls"],
        "avoid": ["Naked Calls"],
        "mean_reversion_safe": True,  # Dips are buyable
    },
    MarketRegime.BEAR_QUIET: {
        "description": "Slow bleed downtrend",
        "optimal_strategies": ["Cash", "Reduce Exposure"],
        "avoid": ["Buying Dips", "Mean Reversion"],
        "mean_reversion_safe": False,
    },
    MarketRegime.BEAR_VOLATILE: {
        "description": "Crash mode - catching knives kills",
        "optimal_strategies": ["Long Puts", "Cash", "VIX Calls"],
        "avoid": ["Buying Dips", "Selling Puts", "Mean Reversion"],
        "mean_reversion_safe": False,  # CRITICAL: Do NOT buy RSI oversold
    },
    MarketRegime.NEUTRAL_CHOP: {
        "description": "Range-bound, low volatility",
        "optimal_strategies": ["Iron Condors", "Mean Reversion", "Range Trading"],
        "avoid": ["Trend Following"],
        "mean_reversion_safe": True,  # This is where RSI works best
    },
    MarketRegime.NEUTRAL_VOLATILE: {
        "description": "Breakout imminent",
        "optimal_strategies": ["Straddles", "Wait for Direction"],
        "avoid": ["Selling Premium"],
        "mean_reversion_safe": False,
    },
}


def calculate_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Calculate 6-state market regime for each row.
    
    Uses 3-factor logic:
    1. Trend: SMA200 + SMA50 alignment with price
    2. Volatility: Bollinger BandWidth percentile (6-month rolling)
    3. Direction validation: Current price vs SMAs
    
    Args:
        df: DataFrame with columns: close, high, low (open, volume optional)
        
    Returns:
        Series with MarketRegime values for each row
    """
    close = df['close']
    
    # ========================================================================
    # Factor 1: Trend (SMA Alignment)
    # ========================================================================
    sma200 = close.rolling(window=200, min_periods=200).mean()
    sma50 = close.rolling(window=50, min_periods=50).mean()
    
    # Bull: Price AND SMA50 above SMA200
    is_bull_trend = (close > sma200) & (sma50 > sma200)
    
    # Bear: Price AND SMA50 below SMA200
    is_bear_trend = (close < sma200) & (sma50 < sma200)
    
    # ========================================================================
    # Factor 2: Volatility (Bollinger BandWidth Percentile)
    # ========================================================================
    
    # Calculate Bollinger Bands
    bb_period = 20
    bb_middle = close.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = close.rolling(window=bb_period, min_periods=bb_period).std(ddof=1)
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Normalized BandWidth
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # 6-month (126 trading days) rolling percentile of bandwidth
    vol_lookback = 126
    
    def rolling_percentile_rank(series, window):
        """Calculate percentile rank within rolling window."""
        return series.rolling(window=window, min_periods=min(20, window)).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    
    vol_percentile = rolling_percentile_rank(bb_width, vol_lookback)
    
    # High vol: Top 30% (vol_percentile > 0.7)
    # Low vol: Bottom 30% (vol_percentile < 0.3)
    is_high_vol = vol_percentile > 0.7
    is_low_vol = vol_percentile < 0.3
    
    # ========================================================================
    # Classification Logic (Vectorized)
    # ========================================================================
    
    # Initialize with default
    regime = pd.Series(MarketRegime.NEUTRAL_CHOP.value, index=df.index)
    
    # Bull Regimes
    # BULL_QUIET: Trending up + Low/Normal volatility (The "Grind Up")
    mask_bull_quiet = is_bull_trend & ~is_high_vol
    regime[mask_bull_quiet] = MarketRegime.BULL_QUIET.value
    
    # BULL_VOLATILE: Trending up + High volatility (Shakeouts common)
    mask_bull_vol = is_bull_trend & is_high_vol
    regime[mask_bull_vol] = MarketRegime.BULL_VOLATILE.value
    
    # Bear Regimes
    # BEAR_VOLATILE: Trending down + High volatility (Crash Mode)
    mask_bear_vol = is_bear_trend & is_high_vol
    regime[mask_bear_vol] = MarketRegime.BEAR_VOLATILE.value
    
    # BEAR_QUIET: Trending down + Low/Normal volatility (Slow Bleed)
    mask_bear_quiet = is_bear_trend & ~is_high_vol
    regime[mask_bear_quiet] = MarketRegime.BEAR_QUIET.value
    
    # Neutral Regimes (not clearly bullish or bearish)
    mask_neutral = ~is_bull_trend & ~is_bear_trend
    
    # NEUTRAL_VOLATILE: Flat price but exploding volatility (Breakout imminent)
    regime[mask_neutral & is_high_vol] = MarketRegime.NEUTRAL_VOLATILE.value
    
    # NEUTRAL_CHOP: Flat price, low/normal vol (Dead money, range trading)
    regime[mask_neutral & ~is_high_vol] = MarketRegime.NEUTRAL_CHOP.value
    
    return regime


def get_regime_for_date(df: pd.DataFrame, date: str) -> Optional[str]:
    """
    Get the market regime for a specific date.
    
    Args:
        df: DataFrame with price history
        date: Date string (YYYY-MM-DD)
        
    Returns:
        MarketRegime value or None if date not found
    """
    regimes = calculate_market_regime(df)
    
    if 'date' in df.columns:
        df_with_date = df.set_index('date') if df.index.name != 'date' else df
        if date in df_with_date.index:
            idx = df_with_date.index.get_loc(date)
            if isinstance(idx, int):
                return regimes.iloc[idx]
    
    return None


def tag_trades_with_regime(trades_df: pd.DataFrame, 
                           price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each trade with the market regime at entry date.
    
    Args:
        trades_df: DataFrame with 'entry_date' column
        price_history: DataFrame with price data for regime calculation
        
    Returns:
        trades_df with 'market_regime' column added
    """
    # Calculate regimes for entire history
    regimes = calculate_market_regime(price_history)
    
    # Create date-to-regime mapping
    if 'date' in price_history.columns:
        regime_map = pd.Series(regimes.values, index=price_history['date'])
    else:
        regime_map = regimes
    
    # Map entry dates to regimes
    trades_df = trades_df.copy()
    trades_df['market_regime'] = trades_df['entry_date'].map(regime_map)
    
    return trades_df


def is_mean_reversion_safe(regime: str) -> bool:
    """
    Check if mean reversion strategies (RSI oversold buying) are safe.
    
    CRITICAL: In BEAR_VOLATILE, buying RSI oversold = catching falling knives.
    
    Args:
        regime: MarketRegime value
        
    Returns:
        True if mean reversion is appropriate, False otherwise
    """
    try:
        regime_enum = MarketRegime(regime)
        return REGIME_STRATEGIES[regime_enum]["mean_reversion_safe"]
    except (ValueError, KeyError):
        return False


def get_regime_weight_adjustments(regime: str) -> dict:
    """
    Get weight adjustments for a specific regime.
    
    In un-tradeable regimes (BEAR_VOLATILE), force cash for mean reversion
    strategies by zeroing out their weights.
    
    Args:
        regime: MarketRegime value
        
    Returns:
        Dict of indicator weight multipliers (0.0 = disable, 1.0 = normal)
    """
    if regime == MarketRegime.BEAR_VOLATILE.value:
        # Crash mode: Zero out mean reversion indicators
        return {
            'rsi': 0.0,  # RSI oversold = falling knife
            'bollinger': 0.0,  # Lower band = no support
            'cmf': 0.5,  # CMF accumulation less reliable
            'position': 0.0,  # "Cheap" = not support
            # Trend indicators still valid
            'macd': 1.0,
            'adx': 1.0,
            'momentum': 1.0,
            'sma': 1.0,
        }
    
    elif regime == MarketRegime.BEAR_QUIET.value:
        # Slow bleed: Reduce mean reversion, favor trend
        return {
            'rsi': 0.3,
            'bollinger': 0.3,
            'macd': 1.2,
            'adx': 1.0,
            'momentum': 1.0,
        }
    
    elif regime == MarketRegime.NEUTRAL_CHOP.value:
        # Range trading: Boost mean reversion
        return {
            'rsi': 1.5,
            'bollinger': 1.5,
            'cmf': 1.2,
            'macd': 0.5,  # MACD whipsaws in chop
            'adx': 0.5,
            'momentum': 0.5,
        }
    
    elif regime in (MarketRegime.BULL_QUIET.value, MarketRegime.BULL_VOLATILE.value):
        # Bull markets: Standard weights, slight trend bias
        return {
            'rsi': 0.8 if regime == MarketRegime.BULL_QUIET.value else 1.2,
            'bollinger': 0.8 if regime == MarketRegime.BULL_QUIET.value else 1.2,
            'macd': 1.2,
            'adx': 1.0,
            'momentum': 1.0,
        }
    
    # Default: no adjustment
    return {}


def get_regime_summary(df: pd.DataFrame) -> dict:
    """
    Get summary statistics of regime distribution in the data.
    
    Args:
        df: DataFrame with price history
        
    Returns:
        Dict with regime counts and percentages
    """
    regimes = calculate_market_regime(df)
    
    # Count valid (non-null) regimes
    valid_regimes = regimes.dropna()
    counts = valid_regimes.value_counts()
    total = len(valid_regimes)
    
    summary = {
        'total_days': total,
        'regimes': {}
    }
    
    for regime in MarketRegime:
        count = counts.get(regime.value, 0)
        summary['regimes'][regime.value] = {
            'count': int(count),
            'percentage': round(count / total * 100, 1) if total > 0 else 0,
            'description': REGIME_STRATEGIES[regime]['description'],
            'mean_reversion_safe': REGIME_STRATEGIES[regime]['mean_reversion_safe'],
        }
    
    return summary

