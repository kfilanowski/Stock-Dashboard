"""
Vectorized Technical Indicators Calculator

Python implementation of technical indicators using Pandas/NumPy for
Walk-Forward Optimization. Must produce identical results to the
TypeScript frontend (technicalIndicators.ts) for calibration accuracy.

IMPORTANT: All calculations use standard periods to prevent curve fitting:
- RSI: 14
- MACD: 12, 26, 9
- Bollinger Bands: 20, 2.0
- ADX: 14
- SMAs: 20, 50, 200
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .optimization_params import OptimizationParams


# ============================================================================
# Configuration - Standard Periods (DO NOT OPTIMIZE THESE)
# ============================================================================

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD_DEV = 2.0
ADX_PERIOD = 14
VWAP_PERIOD = 20
CMF_PERIOD = 20
RVOL_PERIOD = 20


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period, min_periods=period).mean()


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    Uses smoothing factor: 2 / (period + 1)
    """
    return prices.ewm(span=period, adjust=False, min_periods=period).mean()


# ============================================================================
# RSI (Relative Strength Index)
# ============================================================================

def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Calculate RSI with hook detection using TRUE VECTORIZATION.
    
    Uses Pandas ewm() with alpha=1/period which is mathematically equivalent
    to Wilder's smoothing: avg = (prev * (period-1) + current) / period
    
    O(n) complexity instead of O(n²) - critical for WFO speed.
    
    Returns DataFrame with columns:
    - rsi_value: 0-100 scale
    - rsi_prev: Previous period RSI (for hook detection)
    - rsi_oversold: Boolean, < 30
    - rsi_overbought: Boolean, > 70
    - rsi_hooking_up: Boolean, turning upward from oversold
    - rsi_hooking_down: Boolean, turning downward from overbought
    """
    close = df['close']
    
    # Calculate price changes (vectorized)
    delta = close.diff()
    
    # Separate gains and losses (vectorized)
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    
    # Wilder's smoothing using ewm with alpha=1/period
    # This is mathematically equivalent to: (prev * (period-1) + current) / period
    avg_gain = gains.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    # Calculate RS and RSI (vectorized)
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))
    
    # Handle edge case where avg_loss is 0
    rsi_values = rsi_values.fillna(100)  # When avgLoss=0, RSI=100
    
    # Previous RSI for hook detection
    rsi_prev = rsi_values.shift(1)
    
    result = pd.DataFrame(index=df.index)
    result['rsi_value'] = rsi_values.round(2)
    result['rsi_prev'] = rsi_prev.round(2)
    result['rsi_oversold'] = rsi_values < 30
    result['rsi_overbought'] = rsi_values > 70
    result['rsi_hooking_up'] = (rsi_values > rsi_prev) & (rsi_prev < 40)
    result['rsi_hooking_down'] = (rsi_values < rsi_prev) & (rsi_prev > 60)
    
    return result


# ============================================================================
# MACD (Moving Average Convergence Divergence)
# ============================================================================

def calculate_macd(df: pd.DataFrame, 
                   fast: int = MACD_FAST, 
                   slow: int = MACD_SLOW, 
                   signal: int = MACD_SIGNAL) -> pd.DataFrame:
    """
    Calculate MACD indicator using TRUE VECTORIZATION.
    
    Uses Pandas ewm() for O(n) complexity instead of O(n³) loops.
    
    Standard EMA formula: multiplier = 2 / (period + 1)
    ewm(span=period) uses this automatically.
    
    Returns DataFrame with columns:
    - macd_line: EMA(fast) - EMA(slow)
    - macd_signal: EMA of MACD line
    - macd_histogram: MACD - Signal
    - macd_bullish: Boolean, MACD > Signal
    - macd_bearish: Boolean, MACD < Signal
    """
    close = df['close']
    
    # Vectorized EMA calculations using ewm()
    # span=period means multiplier = 2/(period+1) which matches TypeScript
    fast_ema = close.ewm(span=fast, min_periods=fast, adjust=False).mean()
    slow_ema = close.ewm(span=slow, min_periods=slow, adjust=False).mean()
    
    # MACD Line = Fast EMA - Slow EMA
    macd_line = fast_ema - slow_ema
    
    # Signal Line = EMA of MACD Line
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    
    # Histogram = MACD - Signal
    histogram = macd_line - signal_line
    
    result = pd.DataFrame(index=df.index)
    result['macd_line'] = macd_line.round(3)
    result['macd_signal'] = signal_line.round(3)
    result['macd_histogram'] = histogram.round(3)
    result['macd_bullish'] = macd_line > signal_line
    result['macd_bearish'] = macd_line < signal_line
    
    return result


# ============================================================================
# Bollinger Bands
# ============================================================================

def calculate_bollinger_bands(df: pd.DataFrame, 
                               period: int = BB_PERIOD, 
                               std_dev: float = BB_STD_DEV) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    
    Note: Uses SAMPLE standard deviation (n-1) to match TradingView/ThinkOrSwim
    
    Returns DataFrame with columns:
    - bb_upper, bb_middle, bb_lower
    - bb_bandwidth: (upper - lower) / middle
    - bb_percent_b: (price - lower) / (upper - lower), 0-1 scale
    - bb_above_upper, bb_below_lower: Boolean flags
    """
    close = df['close']
    
    # Calculate middle band (SMA)
    middle = close.rolling(window=period, min_periods=period).mean()
    
    # Calculate SAMPLE standard deviation (ddof=1) to match trading platforms
    std = close.rolling(window=period, min_periods=period).std(ddof=1)
    
    # Upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # Bandwidth and %B
    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    
    result = pd.DataFrame(index=df.index)
    result['bb_upper'] = upper.round(2)
    result['bb_middle'] = middle.round(2)
    result['bb_lower'] = lower.round(2)
    result['bb_bandwidth'] = bandwidth.round(3)
    result['bb_percent_b'] = percent_b.round(3)
    result['bb_above_upper'] = close > upper
    result['bb_below_lower'] = close < lower
    
    return result


# ============================================================================
# Bollinger Squeeze (Volatility Contraction)
# ============================================================================

def calculate_bollinger_squeeze(df: pd.DataFrame, 
                                 period: int = BB_PERIOD, 
                                 lookback: int = 100) -> pd.DataFrame:
    """
    Detect Bollinger Band Squeeze (volatility contraction).
    
    Returns DataFrame with columns:
    - squeeze_bandwidth: Current bandwidth
    - squeeze_avg_bandwidth: Average bandwidth over lookback
    - squeeze_percentile: 0-100, where low = squeeze
    - squeeze_is_squeeze: Boolean, bandwidth < 20th percentile
    - squeeze_is_expansion: Boolean, bandwidth > 80th percentile
    """
    # First calculate Bollinger Bands
    bb = calculate_bollinger_bands(df, period)
    bandwidth = bb['bb_bandwidth']
    
    # Calculate rolling percentile of bandwidth
    def rolling_percentile(series, window, current_val):
        """Calculate percentile rank within rolling window."""
        return series.rolling(window=window, min_periods=min(20, window)).apply(
            lambda x: (x <= x.iloc[-1]).sum() / len(x) * 100, raw=False
        )
    
    percentile = rolling_percentile(bandwidth, lookback, bandwidth)
    avg_bandwidth = bandwidth.rolling(window=lookback, min_periods=20).mean()
    
    result = pd.DataFrame(index=df.index)
    result['squeeze_bandwidth'] = bandwidth.round(3)
    result['squeeze_avg_bandwidth'] = avg_bandwidth.round(3)
    result['squeeze_percentile'] = percentile.round(0)
    result['squeeze_is_squeeze'] = percentile <= 20
    result['squeeze_is_expansion'] = percentile >= 80
    
    return result


# ============================================================================
# ADX (Average Directional Index)
# ============================================================================

def calculate_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index).
    
    MUST match TypeScript implementation exactly (technicalIndicators.ts):
    1. Calculate TR, +DM, -DM for each period
    2. Initialize smoothed values with SUM of first period
    3. Apply Wilder's smoothing and calculate DX as we go
    4. ADX = smoothed average of DX values
    
    Returns DataFrame with columns:
    - adx_value: 0-100 scale
    - adx_plus_di, adx_minus_di: Directional indicators
    - adx_trending: Boolean, ADX > 25
    - adx_strong_trend: Boolean, ADX > 40
    - adx_direction: 'bullish', 'bearish', or 'neutral'
    """
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Calculate TR, +DM, -DM for each period (TypeScript: starting from i=1)
    true_ranges = []
    plus_dms = []
    minus_dms = []
    
    for i in range(1, n):
        # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        true_ranges.append(tr)
        
        # +DM and -DM
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        plus_dm = 0.0
        minus_dm = 0.0
        
        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        
        plus_dms.append(plus_dm)
        minus_dms.append(minus_dm)
    
    # Initialize result arrays
    adx_values = np.full(n, np.nan)
    plus_di_values = np.full(n, np.nan)
    minus_di_values = np.full(n, np.nan)
    
    if len(true_ranges) < period:
        result = pd.DataFrame(index=df.index)
        result['adx_value'] = adx_values
        result['adx_plus_di'] = plus_di_values
        result['adx_minus_di'] = minus_di_values
        result['adx_trending'] = False
        result['adx_strong_trend'] = False
        result['adx_direction'] = 'neutral'
        return result
    
    # Calculate initial smoothed values using SUM (TypeScript: slice(0, period).reduce)
    smoothed_tr = sum(true_ranges[:period])
    smoothed_plus_dm = sum(plus_dms[:period])
    smoothed_minus_dm = sum(minus_dms[:period])
    
    # Apply Wilder's smoothing and calculate DX values
    dx_values = []
    
    for i in range(period, len(true_ranges)):
        # Wilder's smoothing: smoothed = prev - (prev/period) + current
        smoothed_tr = smoothed_tr - (smoothed_tr / period) + true_ranges[i]
        smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dms[i]
        smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dms[i]
        
        # Calculate +DI and -DI
        plus_di = (smoothed_plus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
        minus_di = (smoothed_minus_dm / smoothed_tr) * 100 if smoothed_tr > 0 else 0
        
        # Calculate DX
        di_sum = plus_di + minus_di
        dx = (abs(plus_di - minus_di) / di_sum) * 100 if di_sum > 0 else 0
        dx_values.append(dx)
        
        # Store +DI and -DI for current position
        # Index in original df: i + 1 (because true_ranges starts at index 1)
        df_idx = i + 1
        plus_di_values[df_idx] = plus_di
        minus_di_values[df_idx] = minus_di
    
    if len(dx_values) < period:
        result = pd.DataFrame(index=df.index)
        result['adx_value'] = adx_values
        result['adx_plus_di'] = plus_di_values
        result['adx_minus_di'] = minus_di_values
        result['adx_trending'] = False
        result['adx_strong_trend'] = False
        result['adx_direction'] = 'neutral'
        return result
    
    # Calculate ADX as smoothed average of DX
    # First ADX = SMA of first period DX values
    adx = sum(dx_values[:period]) / period
    
    # Store first ADX value
    # DX values start at index period in true_ranges, which is index period+1 in df
    # After period DX values, we're at index period + period in true_ranges = index period*2+1 in df
    first_adx_idx = period * 2
    adx_values[first_adx_idx] = adx
    
    # Apply Wilder's smoothing to ADX for remaining values
    for i in range(period, len(dx_values)):
        adx = ((adx * (period - 1)) + dx_values[i]) / period
        df_idx = period + i + 1
        if df_idx < n:
            adx_values[df_idx] = adx
    
    # Get final +DI and -DI for direction
    final_plus_di = plus_di_values[~np.isnan(plus_di_values)][-1] if np.any(~np.isnan(plus_di_values)) else 0
    final_minus_di = minus_di_values[~np.isnan(minus_di_values)][-1] if np.any(~np.isnan(minus_di_values)) else 0
    
    # Determine direction
    direction = np.where(plus_di_values > minus_di_values + 5, 'bullish',
                np.where(minus_di_values > plus_di_values + 5, 'bearish', 'neutral'))
    
    result = pd.DataFrame(index=df.index)
    result['adx_value'] = np.round(adx_values, 2)
    result['adx_plus_di'] = np.round(plus_di_values, 2)
    result['adx_minus_di'] = np.round(minus_di_values, 2)
    result['adx_trending'] = adx_values >= 25
    result['adx_strong_trend'] = adx_values >= 40
    result['adx_direction'] = direction
    
    return result


# ============================================================================
# VWAP (Volume Weighted Average Price)
# ============================================================================

def calculate_vwap(df: pd.DataFrame, period: int = VWAP_PERIOD) -> pd.DataFrame:
    """
    Calculate VWAP (Volume Weighted Average Price).
    
    For daily data, uses rolling window (not true intraday VWAP).
    
    Returns DataFrame with columns:
    - vwap_value
    - vwap_price_vs: Percentage above/below VWAP
    - vwap_above, vwap_below: Boolean flags
    """
    # Typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Volume-weighted
    tpv = typical_price * df['volume']
    
    # Rolling VWAP
    cum_tpv = tpv.rolling(window=period, min_periods=1).sum()
    cum_vol = df['volume'].rolling(window=period, min_periods=1).sum()
    
    vwap = cum_tpv / cum_vol.replace(0, np.inf)
    
    # Price vs VWAP
    price_vs = ((df['close'] - vwap) / vwap) * 100
    
    result = pd.DataFrame(index=df.index)
    result['vwap_value'] = vwap.round(2)
    result['vwap_price_vs'] = price_vs.round(2)
    result['vwap_above'] = df['close'] > vwap
    result['vwap_below'] = df['close'] < vwap
    
    return result


# ============================================================================
# Momentum
# ============================================================================

def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price momentum over various periods.
    
    Momentum = ((Current Price - Price N days ago) / Price N days ago) * 100
    
    Returns DataFrame with columns:
    - momentum_short: 5-day momentum (%)
    - momentum_medium: 20-day momentum (%)
    - momentum_long: 50-day momentum (%)
    - momentum_trend: 'bullish', 'bearish', or 'neutral'
    """
    close = df['close']
    
    short_term = ((close - close.shift(5)) / close.shift(5)) * 100
    medium_term = ((close - close.shift(20)) / close.shift(20)) * 100
    long_term = ((close - close.shift(50)) / close.shift(50)) * 100
    
    # Determine trend
    trend = np.where((short_term > 0) & (medium_term > 0) & (long_term > 0), 'bullish',
            np.where((short_term < 0) & (medium_term < 0) & (long_term < 0), 'bearish', 'neutral'))
    
    result = pd.DataFrame(index=df.index)
    result['momentum_short'] = short_term.round(2)
    result['momentum_medium'] = medium_term.round(2)
    result['momentum_long'] = long_term.round(2)
    result['momentum_trend'] = trend
    
    return result


# ============================================================================
# Volume Analysis
# ============================================================================

def calculate_volume_analysis(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Analyze volume patterns.
    
    Returns DataFrame with columns:
    - volume_avg: 20-day average volume
    - volume_ratio: Current / Average
    - volume_high: Boolean, > 1.5x average
    - volume_low: Boolean, < 0.5x average
    - volume_trend: 'increasing', 'decreasing', or 'stable'
    """
    volume = df['volume']
    
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume / avg_volume.replace(0, np.inf)
    
    # Volume trend (compare first half to second half)
    half = period // 2
    first_half = volume.rolling(window=half).mean()
    second_half = volume.rolling(window=half).mean()
    
    trend = np.where(second_half > first_half * 1.2, 'increasing',
            np.where(second_half < first_half * 0.8, 'decreasing', 'stable'))
    
    result = pd.DataFrame(index=df.index)
    result['volume_avg'] = avg_volume.round(0)
    result['volume_current'] = volume
    result['volume_ratio'] = volume_ratio.round(2)
    result['volume_high'] = volume_ratio > 1.5
    result['volume_low'] = volume_ratio < 0.5
    result['volume_trend'] = trend
    
    return result


# ============================================================================
# RVOL (Relative Volume)
# ============================================================================

def calculate_rvol(df: pd.DataFrame, period: int = RVOL_PERIOD) -> pd.DataFrame:
    """
    Calculate Relative Volume (RVOL).
    
    RVOL = Current Volume / Average Volume
    
    Returns DataFrame with columns:
    - rvol_value: Ratio
    - rvol_interpretation: 'very_high', 'high', 'normal', 'low', 'very_low'
    - rvol_significant: Boolean, RVOL > 2.0 or < 0.5
    """
    volume = df['volume']
    
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    rvol = volume / avg_volume.replace(0, np.inf)
    
    # Interpretation
    interpretation = np.where(rvol > 3.0, 'very_high',
                     np.where(rvol > 1.5, 'high',
                     np.where(rvol >= 0.8, 'normal',
                     np.where(rvol >= 0.5, 'low', 'very_low'))))
    
    result = pd.DataFrame(index=df.index)
    result['rvol_value'] = rvol.round(2)
    result['rvol_interpretation'] = interpretation
    result['rvol_significant'] = (rvol > 2.0) | (rvol < 0.5)
    
    return result


# ============================================================================
# CMF (Chaikin Money Flow)
# ============================================================================

def calculate_cmf(df: pd.DataFrame, period: int = CMF_PERIOD) -> pd.DataFrame:
    """
    Calculate Chaikin Money Flow (CMF).
    
    CMF measures buying/selling pressure based on where price closes
    within its range, weighted by volume.
    
    Returns DataFrame with columns:
    - cmf_value: -1 to +1 scale
    - cmf_interpretation: 'strong_accumulation' to 'strong_distribution'
    - cmf_accumulating, cmf_distributing: Boolean flags
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Money Flow Multiplier: -1 (closed at low) to +1 (closed at high)
    hl_range = high - low
    mfm = ((close - low) - (high - close)) / hl_range.replace(0, np.inf)
    
    # Money Flow Volume
    mfv = mfm * volume
    
    # CMF = sum(MFV) / sum(Volume) over period
    sum_mfv = mfv.rolling(window=period, min_periods=period).sum()
    sum_vol = volume.rolling(window=period, min_periods=period).sum()
    cmf = sum_mfv / sum_vol.replace(0, np.inf)
    
    # Interpretation
    interpretation = np.where(cmf > 0.25, 'strong_accumulation',
                     np.where(cmf > 0.1, 'accumulation',
                     np.where(cmf >= -0.1, 'neutral',
                     np.where(cmf >= -0.25, 'distribution', 'strong_distribution'))))
    
    result = pd.DataFrame(index=df.index)
    result['cmf_value'] = cmf.round(3)
    result['cmf_interpretation'] = interpretation
    result['cmf_accumulating'] = cmf > 0.1
    result['cmf_distributing'] = cmf < -0.1
    
    return result


# ============================================================================
# SMA Alignment & Cross Patterns
# ============================================================================

def calculate_sma_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate SMA alignment and detect cross patterns.
    
    Returns DataFrame with columns:
    - sma_20, sma_50, sma_200: SMA values
    - sma_bullish_aligned: Price > SMA50 > SMA200
    - sma_bearish_aligned: Price < SMA50 < SMA200
    - cross_golden: Boolean, recent golden cross
    - cross_death: Boolean, recent death cross
    """
    close = df['close']
    
    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
    # Alignment
    bullish_aligned = (close > sma50) & (sma50 > sma200)
    bearish_aligned = (close < sma50) & (sma50 < sma200)
    
    # Cross detection (look for cross in last 20 days)
    sma50_prev = sma50.shift(1)
    sma200_prev = sma200.shift(1)
    
    # Golden cross: SMA50 crosses above SMA200
    golden_cross_today = (sma50_prev <= sma200_prev) & (sma50 > sma200)
    # Death cross: SMA50 crosses below SMA200
    death_cross_today = (sma50_prev >= sma200_prev) & (sma50 < sma200)
    
    # Recent cross (within 20 days)
    golden_recent = golden_cross_today.rolling(window=20, min_periods=1).max().astype(bool)
    death_recent = death_cross_today.rolling(window=20, min_periods=1).max().astype(bool)
    
    result = pd.DataFrame(index=df.index)
    result['sma_20'] = sma20.round(2)
    result['sma_50'] = sma50.round(2)
    result['sma_200'] = sma200.round(2)
    result['sma_bullish_aligned'] = bullish_aligned
    result['sma_bearish_aligned'] = bearish_aligned
    result['cross_golden'] = golden_recent
    result['cross_death'] = death_recent
    
    return result


# ============================================================================
# Price Position (52-week range)
# ============================================================================

def calculate_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze price position relative to 52-week range.
    
    Returns DataFrame with columns:
    - price_52w_high, price_52w_low: Rolling 52-week extremes
    - price_range_position: 0-1, where 0 = at low, 1 = at high
    - price_near_high, price_near_low: Within 5% of extreme
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # 252 trading days ≈ 1 year
    high_52w = high.rolling(window=252, min_periods=50).max()
    low_52w = low.rolling(window=252, min_periods=50).min()
    
    # Position in range
    range_size = high_52w - low_52w
    range_position = (close - low_52w) / range_size.replace(0, np.inf)
    
    # Near extremes
    pct_from_high = ((close - high_52w) / high_52w) * 100
    pct_from_low = ((close - low_52w) / low_52w) * 100
    
    result = pd.DataFrame(index=df.index)
    result['price_52w_high'] = high_52w.round(2)
    result['price_52w_low'] = low_52w.round(2)
    result['price_range_position'] = range_position.round(3)
    result['price_near_high'] = pct_from_high >= -5
    result['price_near_low'] = pct_from_low <= 5
    result['price_vs_high'] = pct_from_high.round(2)
    result['price_vs_low'] = pct_from_low.round(2)
    
    return result


# ============================================================================
# Main Aggregation Function
# ============================================================================

def calculate_all_indicators(
    df: pd.DataFrame,
    sector_df: pd.DataFrame = None,
    params: 'OptimizationParams' = None
) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame.

    Args:
        df: DataFrame with columns: date, open, high, low, close, volume
        sector_df: Optional sector ETF DataFrame for relative strength calculations
        params: Optional OptimizationParams with custom indicator periods

    Returns:
        DataFrame with all indicator columns appended
    """
    if len(df) < 50:
        raise ValueError("Need at least 50 data points for indicator calculation")

    result = df.copy()

    # Get indicator periods from params (use defaults if not provided)
    if params is not None and hasattr(params, 'indicator_periods'):
        rsi_period = params.indicator_periods.rsi_period
        macd_fast = params.indicator_periods.macd_fast
        macd_slow = params.indicator_periods.macd_slow
        macd_signal = params.indicator_periods.macd_signal
        adx_period = params.indicator_periods.adx_period
        bb_period = params.indicator_periods.bb_period
    else:
        rsi_period = RSI_PERIOD
        macd_fast = MACD_FAST
        macd_slow = MACD_SLOW
        macd_signal = MACD_SIGNAL
        adx_period = ADX_PERIOD
        bb_period = BB_PERIOD

    # Calculate each indicator with custom or default periods
    result = pd.concat([result, calculate_rsi(df, period=rsi_period)], axis=1)
    result = pd.concat([result, calculate_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)], axis=1)
    result = pd.concat([result, calculate_bollinger_bands(df, period=bb_period)], axis=1)
    result = pd.concat([result, calculate_bollinger_squeeze(df, period=bb_period)], axis=1)
    result = pd.concat([result, calculate_adx(df, period=adx_period)], axis=1)
    result = pd.concat([result, calculate_vwap(df)], axis=1)
    result = pd.concat([result, calculate_momentum(df)], axis=1)
    result = pd.concat([result, calculate_volume_analysis(df)], axis=1)
    result = pd.concat([result, calculate_rvol(df)], axis=1)
    result = pd.concat([result, calculate_cmf(df)], axis=1)
    result = pd.concat([result, calculate_sma_alignment(df)], axis=1)
    result = pd.concat([result, calculate_price_position(df)], axis=1)

    # Add EMA columns directly
    result['ema_12'] = calculate_ema(df['close'], 12).round(2)
    result['ema_26'] = calculate_ema(df['close'], 26).round(2)

    # Add relative strength indicators if sector data is available
    if sector_df is not None and len(sector_df) > 0:
        try:
            from .relative_strength import calculate_all_relative_strength
            rs_data = calculate_all_relative_strength(df, sector_df, align_dates=True)
            if not rs_data.empty:
                # Align RS data with result index
                for col in ['rel_momentum', 'rs_ratio', 'rs_momentum']:
                    if col in rs_data.columns:
                        result[col] = rs_data[col].reindex(result.index).values
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[INDICATORS] Could not calculate RS: {e}")

    return result


# ============================================================================
# Signal Interpretation (for WFO scoring)
# ============================================================================

def calculate_indicator_signals(
    df: pd.DataFrame,
    params: 'OptimizationParams' = None,
    directions: Optional[Dict[str, float]] = None,
    sector_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Convert raw indicator values to normalized signals (-1 to +1).

    This mirrors the signal interpretation logic in stockScoring.ts
    and is used for Walk-Forward Optimization.

    Args:
        df: DataFrame with OHLCV data
        params: Optional OptimizationParams for configurable breakpoints
        directions: Optional dict of signal direction multipliers for mean-reversion
                   indicators. Keys are '{indicator}_dir', values in [-1, +1].
                   +1 = mean reversion (oversold = bullish)
                   -1 = momentum (oversold = bearish continuation)
                   Default: +1 for all (mean reversion)
        sector_df: Optional sector ETF DataFrame for relative strength calculations

    Returns DataFrame with signal columns (one per indicator per action).
    """
    # Default directions (mean reversion for all)
    if directions is None:
        directions = {}

    # First calculate all indicators (with custom periods if params provided)
    indicators = calculate_all_indicators(df, sector_df=sector_df, params=params)

    signals = pd.DataFrame(index=df.index)

    # Get signal breakpoints (use defaults if params not provided)
    if params is not None:
        bp = params.signal_breakpoints
        rsi_oversold = bp.rsi_oversold
        rsi_overbought = bp.rsi_overbought
        adx_no_trend = bp.adx_no_trend
        adx_trending = bp.adx_trending
        adx_strong_trend = bp.adx_strong_trend
        momentum_strong_bull = bp.momentum_strong_bull
        momentum_bull = bp.momentum_bull
        momentum_bear = bp.momentum_bear
        momentum_strong_bear = bp.momentum_strong_bear
        volume_very_high = bp.volume_very_high
        volume_high = bp.volume_high
        volume_above_normal = bp.volume_above_normal
    else:
        # Default hardcoded values (backward compatibility)
        rsi_oversold = 30.0
        rsi_overbought = 70.0
        adx_no_trend = 20.0
        adx_trending = 25.0
        adx_strong_trend = 40.0
        momentum_strong_bull = 10.0
        momentum_bull = 5.0
        momentum_bear = -5.0
        momentum_strong_bear = -10.0
        volume_very_high = 2.0
        volume_high = 1.5
        volume_above_normal = 1.0

    # RSI Signal (mean reversion interpretation, with learnable direction)
    rsi = indicators['rsi_value']
    adx_is_trending = indicators['adx_trending']

    # Get direction multiplier: +1 = mean reversion, -1 = momentum
    rsi_dir = directions.get('rsi_dir', 1.0)

    # Base RSI signal (mean reversion) using configurable breakpoints
    rsi_mid = (rsi_oversold + rsi_overbought) / 2  # Typically 50
    base_rsi = np.where(rsi <= rsi_oversold, 1.0 - (rsi / rsi_oversold) * 0.5,
               np.where(rsi <= rsi_mid, 0.5 - ((rsi - rsi_oversold) / (rsi_mid - rsi_oversold)) * 0.5,
               np.where(rsi <= rsi_overbought, -((rsi - rsi_mid) / (rsi_overbought - rsi_mid)) * 0.5,
                        -0.5 - ((rsi - rsi_overbought) / (100 - rsi_overbought)) * 0.5)))

    # Apply direction multiplier (flips signal for momentum stocks)
    # When rsi_dir = -1: oversold becomes bearish, overbought becomes bullish
    base_rsi = base_rsi * rsi_dir

    # Regime override: Trending market (ADX above trending threshold)
    # Only apply if not in momentum mode (rsi_dir > 0)
    trending_bullish = (rsi >= 65) & (rsi < 85) & adx_is_trending & (rsi_dir > 0)
    rsi_signal = np.where(trending_bullish, 0.8, base_rsi)
    signals['rsi_signal'] = np.clip(rsi_signal, -1, 1).round(4)
    
    # MACD Signal
    macd_bullish = indicators['macd_bullish']
    histogram = indicators['macd_histogram'].abs()
    histogram_strength = np.minimum(histogram / 2, 1)
    
    macd_signal = np.where(macd_bullish, 
                          0.3 + histogram_strength * 0.7,
                          -(0.3 + histogram_strength * 0.7))
    signals['macd_signal'] = np.clip(macd_signal, -1, 1).round(4)
    
    # Bollinger Bands Signal (mean reversion with learnable direction)
    # Get direction multiplier: +1 = mean reversion, -1 = momentum
    bollinger_dir = directions.get('bollinger_dir', 1.0)

    percent_b = indicators['bb_percent_b']
    bb_signal = np.where(percent_b <= 0, 1.0,
                np.where(percent_b <= 0.2, 0.8,
                np.where(percent_b <= 0.4, 0.4,
                np.where(percent_b <= 0.6, 0.0,
                np.where(percent_b <= 0.8, -0.4,
                np.where(percent_b <= 1.0, -0.8, -1.0))))))

    # Apply direction multiplier (flips signal for momentum/breakout stocks)
    bb_signal = bb_signal * bollinger_dir
    signals['bollinger_signal'] = np.clip(bb_signal, -1, 1).round(4)
    
    # ADX Signal (trend strength) using configurable breakpoints
    adx = indicators['adx_value']
    adx_direction = indicators['adx_direction']

    # ADX signal based on trend strength
    adx_signal = np.where(adx < adx_no_trend, 0.0,  # No trend
                 np.where(adx < adx_trending, 0.3,
                 np.where(adx < adx_strong_trend, 0.6, 0.9)))
    # Direction adjustment
    adx_signal = np.where(adx_direction == 'bearish', -adx_signal, adx_signal)
    signals['adx_signal'] = np.clip(adx_signal, -1, 1).round(4)
    
    # CMF Signal
    cmf = indicators['cmf_value']
    cmf_signal = cmf * 2  # Scale -0.5 to 0.5 → -1 to 1
    signals['cmf_signal'] = np.clip(cmf_signal, -1, 1).round(4)
    
    # Momentum Signal using configurable breakpoints
    momentum_short = indicators['momentum_short']
    momentum_signal = np.where(momentum_short > momentum_strong_bull, 1.0,
                      np.where(momentum_short > momentum_bull, 0.6,
                      np.where(momentum_short > 0, 0.2,
                      np.where(momentum_short > momentum_bear, -0.2,
                      np.where(momentum_short > momentum_strong_bear, -0.6, -1.0)))))
    signals['momentum_signal'] = np.clip(momentum_signal, -1, 1).round(4)
    
    # Volume Signal (direction-aware like TypeScript) using configurable breakpoints
    # High volume CONFIRMS the price direction
    volume_ratio = indicators['volume_ratio']
    price_change = df['close'].pct_change().fillna(0)
    is_price_up = price_change > 0

    # Direction: +1 for up, -1 for down
    direction = np.where(is_price_up, 1, -1)

    # Volume magnitude using configurable breakpoints
    vol_magnitude = np.where(volume_ratio > volume_very_high, 1.0,
                    np.where(volume_ratio > volume_high, 0.6,
                    np.where(volume_ratio > volume_above_normal, 0.3, 0.0)))

    # Volume signal = direction × magnitude (high volume confirms direction)
    vol_signal = direction * vol_magnitude
    signals['volume_signal'] = np.clip(vol_signal, -1, 1).round(4)
    
    # RVOL Signal
    rvol = indicators['rvol_value']
    rvol_signal = np.where(rvol > 3.0, 0.9,
                  np.where(rvol > 2.0, 0.6,
                  np.where(rvol > 1.5, 0.3,
                  np.where(rvol >= 0.8, 0.0,
                  np.where(rvol >= 0.5, -0.3, -0.6)))))
    signals['rvol_signal'] = np.clip(rvol_signal, -1, 1).round(4)
    
    # SMA Alignment Signal
    bullish_aligned = indicators['sma_bullish_aligned']
    bearish_aligned = indicators['sma_bearish_aligned']
    sma_signal = np.where(bullish_aligned, 0.8,
                 np.where(bearish_aligned, -0.8, 0.0))
    signals['sma_signal'] = sma_signal.round(4)
    
    # Cross Pattern Signal
    golden = indicators['cross_golden']
    death = indicators['cross_death']
    cross_signal = np.where(golden, 0.9,
                   np.where(death, -0.9, 0.0))
    signals['cross_signal'] = cross_signal.round(4)
    
    # Price Position Signal (with learnable direction)
    # Get direction multiplier: +1 = mean reversion, -1 = momentum
    position_dir = directions.get('position_dir', 1.0)

    # In TRENDING markets (ADX > 25), high price is neutral/positive (trend continuation)
    # In RANGING markets (ADX < 25), use mean-reversion logic (low = buy, high = sell)
    range_pos = indicators['price_range_position']
    is_trending = adx_trending  # Already calculated above

    # Mean-reversion signal (for ranging markets)
    mr_signal = np.where(range_pos <= 0.2, 0.8,
                np.where(range_pos <= 0.4, 0.4,
                np.where(range_pos <= 0.6, 0.0,
                np.where(range_pos <= 0.8, -0.4, -0.8))))

    # Apply direction multiplier to mean-reversion signal
    mr_signal = mr_signal * position_dir

    # Trend-following signal (for trending markets)
    # At highs in uptrend = neutral/slightly positive (momentum)
    # At lows in downtrend = neutral/slightly negative
    adx_direction = indicators['adx_direction']
    tf_signal = np.where(
        adx_direction == 'bullish',
        np.where(range_pos >= 0.6, 0.2, 0.0),  # At highs in uptrend = slightly bullish
        np.where(range_pos <= 0.4, -0.2, 0.0)  # At lows in downtrend = slightly bearish
    )

    # Use trend signal when trending (unaffected by direction), mean-reversion when ranging
    position_signal = np.where(is_trending, tf_signal, mr_signal)
    signals['position_signal'] = np.clip(position_signal, -1, 1).round(4)
    
    # Squeeze Signal (for options)
    is_squeeze = indicators['squeeze_is_squeeze']
    is_expansion = indicators['squeeze_is_expansion']
    squeeze_signal = np.where(is_squeeze, 0.8,  # Cheap options before explosion
                     np.where(is_expansion, -0.5, 0.0))  # Expensive options
    signals['squeeze_signal'] = squeeze_signal.round(4)

    # Relative Strength Signals (if available in indicators)
    if 'rel_momentum' in indicators.columns:
        # Relative momentum signal: outperformance vs sector
        rel_mom = indicators['rel_momentum']
        rel_mom_signal = np.where(rel_mom > 10, 1.0,
                         np.where(rel_mom > 5, 0.6,
                         np.where(rel_mom > 0, 0.3,
                         np.where(rel_mom > -5, -0.3,
                         np.where(rel_mom > -10, -0.6, -1.0)))))
        signals['rel_momentum_signal'] = np.clip(rel_mom_signal, -1, 1).round(4)

    if 'rs_ratio' in indicators.columns and 'rs_momentum' in indicators.columns:
        # RS Ratio signal: RRG-style quadrant logic
        rs_ratio = indicators['rs_ratio']
        rs_momentum = indicators['rs_momentum']
        rs_center = 100.0

        # Determine quadrant and assign signal
        rs_above = rs_ratio >= rs_center
        mom_pos = rs_momentum >= 0

        # Distance from center (scaled)
        rs_distance = (rs_ratio - rs_center) / 10
        mom_modifier = np.clip(rs_momentum / 5, -0.3, 0.3)

        # Leading quadrant (RS > 100, Mom > 0): strong bullish
        # Weakening (RS > 100, Mom < 0): neutral-bearish
        # Lagging (RS < 100, Mom < 0): strong bearish
        # Improving (RS < 100, Mom > 0): neutral-bullish
        rs_signal = np.where(
            rs_above & mom_pos,
            np.clip(rs_distance + mom_modifier, 0.5, 1.0),
            np.where(
                rs_above & ~mom_pos,
                np.clip(rs_distance + mom_modifier, -0.3, 0.5),
                np.where(
                    ~rs_above & ~mom_pos,
                    np.clip(rs_distance + mom_modifier, -1.0, -0.3),
                    np.clip(rs_distance + mom_modifier, 0.0, 0.7)
                )
            )
        )
        signals['rs_ratio_signal'] = np.clip(rs_signal, -1, 1).round(4)

    # VWAP Signal (mean reversion with learnable direction)
    # Get direction multiplier: +1 = mean reversion, -1 = momentum
    vwap_dir = directions.get('vwap_dir', 1.0)

    # Price below VWAP = bullish (undervalued relative to volume-weighted price)
    # Price above VWAP = bearish (overvalued relative to volume-weighted price)
    if 'vwap_price_vs' in indicators.columns:
        vwap_deviation = indicators['vwap_price_vs']  # Already in percentage
        # Scale: -3% or more below = +0.8, +3% or more above = -0.8
        vwap_signal = np.where(vwap_deviation < -3, 0.8,
                     np.where(vwap_deviation < -1, 0.4,
                     np.where(vwap_deviation <= 1, 0.0,
                     np.where(vwap_deviation <= 3, -0.4, -0.8))))

        # Apply direction multiplier (flips signal for momentum stocks)
        vwap_signal = vwap_signal * vwap_dir
        signals['vwap_signal'] = np.clip(vwap_signal, -1, 1).round(4)

    return signals

