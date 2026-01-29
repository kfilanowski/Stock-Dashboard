"""
Relative Strength Indicators

Calculates stock-vs-sector relative performance indicators for institutional-grade
analysis. These indicators measure how a stock performs relative to its sector,
which is a key factor in professional stock selection.

Key Indicators:
1. Relative Momentum: Stock return minus sector return over a lookback period
2. RS Ratio: RRG-style ratio showing relative strength (normalized to 100)
3. RS Momentum: Rate of change of RS Ratio

Usage:
    stock_df = load_stock_prices('AAPL')
    sector_df = get_benchmark_for_stock('AAPL')

    rs_data = calculate_all_relative_strength(stock_df, sector_df)
    # Returns DataFrame with rel_momentum, rs_ratio, rs_momentum columns
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Lookback periods for relative calculations
REL_MOMENTUM_LOOKBACK = 20  # 1 month (~20 trading days)
RS_RATIO_LOOKBACK = 52      # ~2.5 months for smoothing
RS_MOMENTUM_LOOKBACK = 10   # 2 weeks for momentum of ratio

# Normalization for RS Ratio (RRG-style centers at 100)
RS_RATIO_CENTER = 100.0


# ============================================================================
# Relative Momentum
# ============================================================================

def calculate_relative_momentum(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    lookback: int = REL_MOMENTUM_LOOKBACK
) -> pd.Series:
    """
    Calculate relative momentum: stock return minus benchmark return.

    Positive values indicate the stock is outperforming its sector.
    Negative values indicate underperformance.

    Args:
        stock_close: Stock closing prices
        benchmark_close: Benchmark (sector ETF) closing prices
        lookback: Lookback period in days

    Returns:
        Series of relative momentum values (as percentages)
    """
    # Calculate returns over lookback period
    stock_return = stock_close.pct_change(lookback) * 100  # As percentage
    benchmark_return = benchmark_close.pct_change(lookback) * 100

    # Relative momentum = stock return - benchmark return
    rel_momentum = stock_return - benchmark_return

    return rel_momentum


# ============================================================================
# RS Ratio (RRG-Style)
# ============================================================================

def calculate_rs_ratio(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    lookback: int = RS_RATIO_LOOKBACK
) -> pd.Series:
    """
    Calculate RS Ratio (Relative Rotation Graph style).

    The RS Ratio measures the relative strength of a stock vs its benchmark,
    normalized to 100. Values > 100 indicate outperformance, < 100 underperformance.

    Method:
    1. Calculate raw relative strength: stock / benchmark
    2. Normalize using rolling mean to center at 100

    Args:
        stock_close: Stock closing prices
        benchmark_close: Benchmark closing prices
        lookback: Smoothing period

    Returns:
        Series of RS Ratio values (centered at 100)
    """
    # Raw relative strength
    raw_rs = stock_close / benchmark_close

    # Rolling mean for normalization
    rs_mean = raw_rs.rolling(window=lookback, min_periods=lookback // 2).mean()

    # Normalize to center at 100
    # RS Ratio = 100 * (raw_rs / rolling_mean)
    rs_ratio = RS_RATIO_CENTER * (raw_rs / rs_mean)

    return rs_ratio


def calculate_rs_momentum(
    rs_ratio: pd.Series,
    lookback: int = RS_MOMENTUM_LOOKBACK
) -> pd.Series:
    """
    Calculate RS Momentum: rate of change of RS Ratio.

    Positive RS Momentum indicates improving relative strength.
    Negative RS Momentum indicates deteriorating relative strength.

    Args:
        rs_ratio: RS Ratio series
        lookback: Lookback period for momentum calculation

    Returns:
        Series of RS Momentum values
    """
    # Momentum = current RS Ratio - RS Ratio N days ago
    rs_momentum = rs_ratio - rs_ratio.shift(lookback)

    return rs_momentum


# ============================================================================
# Signal Interpretation
# ============================================================================

def calculate_rel_momentum_signal(rel_momentum: pd.Series) -> pd.Series:
    """
    Convert relative momentum to a normalized signal (-1 to +1).

    Interpretation:
    - Strong outperformance (> 10%): +1.0
    - Moderate outperformance (5-10%): +0.6
    - Slight outperformance (0-5%): +0.3
    - Slight underperformance (-5 to 0%): -0.3
    - Moderate underperformance (-10 to -5%): -0.6
    - Strong underperformance (< -10%): -1.0
    """
    signal = np.where(rel_momentum > 10, 1.0,
             np.where(rel_momentum > 5, 0.6,
             np.where(rel_momentum > 0, 0.3,
             np.where(rel_momentum > -5, -0.3,
             np.where(rel_momentum > -10, -0.6, -1.0)))))

    return pd.Series(np.clip(signal, -1, 1), index=rel_momentum.index).round(4)


def calculate_rs_ratio_signal(
    rs_ratio: pd.Series,
    rs_momentum: pd.Series
) -> pd.Series:
    """
    Convert RS Ratio and RS Momentum to a combined signal (-1 to +1).

    Uses RRG quadrant logic:
    - Leading (RS > 100, Mom > 0): Strong bullish (+0.8 to +1.0)
    - Weakening (RS > 100, Mom < 0): Neutral-bearish (-0.2 to +0.4)
    - Lagging (RS < 100, Mom < 0): Strong bearish (-0.8 to -1.0)
    - Improving (RS < 100, Mom > 0): Neutral-bullish (+0.2 to +0.6)
    """
    # Determine quadrant
    rs_above_100 = rs_ratio >= RS_RATIO_CENTER
    mom_positive = rs_momentum >= 0

    # Base signal from RS Ratio distance from 100
    rs_distance = (rs_ratio - RS_RATIO_CENTER) / 10  # Scale: 110 -> 1.0, 90 -> -1.0

    # Momentum modifier
    mom_modifier = np.clip(rs_momentum / 5, -0.3, 0.3)  # ±0.3 modifier

    # Combined signal
    signal = np.where(
        rs_above_100 & mom_positive,  # Leading quadrant
        np.clip(rs_distance + mom_modifier, 0.5, 1.0),
        np.where(
            rs_above_100 & ~mom_positive,  # Weakening quadrant
            np.clip(rs_distance + mom_modifier, -0.3, 0.5),
            np.where(
                ~rs_above_100 & ~mom_positive,  # Lagging quadrant
                np.clip(rs_distance + mom_modifier, -1.0, -0.3),
                # Improving quadrant
                np.clip(rs_distance + mom_modifier, 0.0, 0.7)
            )
        )
    )

    return pd.Series(np.clip(signal, -1, 1), index=rs_ratio.index).round(4)


# ============================================================================
# Main Entry Point
# ============================================================================

def calculate_all_relative_strength(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    align_dates: bool = True
) -> pd.DataFrame:
    """
    Calculate all relative strength indicators for a stock.

    Args:
        stock_df: Stock DataFrame with 'close' and 'date' columns
        benchmark_df: Benchmark DataFrame with 'close' and 'date' columns
        align_dates: If True, align dataframes by date

    Returns:
        DataFrame with columns:
        - rel_momentum: Relative momentum (stock return - sector return)
        - rs_ratio: RS Ratio (normalized to 100)
        - rs_momentum: Rate of change of RS Ratio
        - rel_momentum_signal: Signal from relative momentum (-1 to +1)
        - rs_ratio_signal: Combined RS signal (-1 to +1)
    """
    if benchmark_df is None or len(benchmark_df) == 0:
        logger.warning("[RS] No benchmark data available, returning empty")
        return pd.DataFrame(index=stock_df.index)

    # Align data by date if requested
    if align_dates and 'date' in stock_df.columns and 'date' in benchmark_df.columns:
        # Merge on date to align
        merged = pd.merge(
            stock_df[['date', 'close']].rename(columns={'close': 'stock_close'}),
            benchmark_df[['date', 'close']].rename(columns={'close': 'benchmark_close'}),
            on='date',
            how='inner'
        )

        if len(merged) < 60:
            logger.warning(f"[RS] Insufficient aligned data: {len(merged)} days")
            return pd.DataFrame(index=stock_df.index)

        stock_close = merged['stock_close']
        benchmark_close = merged['benchmark_close']
        result_index = merged.index
    else:
        # Assume already aligned
        if len(stock_df) != len(benchmark_df):
            logger.warning(f"[RS] Length mismatch: stock={len(stock_df)}, benchmark={len(benchmark_df)}")
            min_len = min(len(stock_df), len(benchmark_df))
            stock_close = stock_df['close'].iloc[-min_len:].reset_index(drop=True)
            benchmark_close = benchmark_df['close'].iloc[-min_len:].reset_index(drop=True)
            result_index = stock_df.index[-min_len:]
        else:
            stock_close = stock_df['close']
            benchmark_close = benchmark_df['close']
            result_index = stock_df.index

    # Calculate indicators
    rel_momentum = calculate_relative_momentum(stock_close, benchmark_close)
    rs_ratio = calculate_rs_ratio(stock_close, benchmark_close)
    rs_momentum = calculate_rs_momentum(rs_ratio)

    # Calculate signals
    rel_momentum_signal = calculate_rel_momentum_signal(rel_momentum)
    rs_ratio_signal = calculate_rs_ratio_signal(rs_ratio, rs_momentum)

    # Build result DataFrame
    result = pd.DataFrame({
        'rel_momentum': rel_momentum.values,
        'rs_ratio': rs_ratio.values,
        'rs_momentum': rs_momentum.values,
        'rel_momentum_signal': rel_momentum_signal.values,
        'rs_ratio_signal': rs_ratio_signal.values,
    }, index=result_index)

    logger.debug(f"[RS] Calculated relative strength: {len(result)} rows")

    return result


def get_rs_quadrant(rs_ratio: float, rs_momentum: float) -> str:
    """
    Determine RRG quadrant from RS Ratio and Momentum.

    Returns:
        One of: 'leading', 'weakening', 'lagging', 'improving'
    """
    if rs_ratio >= RS_RATIO_CENTER:
        return 'leading' if rs_momentum >= 0 else 'weakening'
    else:
        return 'improving' if rs_momentum >= 0 else 'lagging'
