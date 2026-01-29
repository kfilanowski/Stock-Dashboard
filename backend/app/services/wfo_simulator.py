"""
Walk-Forward Optimization Simulator

Vectorized backtesting engine for WFO calibration.
Uses Pandas/NumPy operations for 100-1000x speed improvement over loops.

CONSTRAINTS (Pre-Flight Checklist):
1. Only optimize WEIGHTS (0.0-2.0), not indicator periods
2. Transaction cost >= 0.1% (realistic friction)
3. Force cash in un-tradeable regimes (BEAR_VOLATILE)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .indicators import calculate_indicator_signals
from .regime import calculate_market_regime, MarketRegime
from .optimization_params import OptimizationParams, DEFAULT_PARAMS


# ============================================================================
# Configuration
# ============================================================================

# Minimum transaction cost (0.1% = 10 basis points)
MIN_TRANSACTION_COST = 0.001

# Slippage configuration (adaptive based on volatility)
# Models the cost of executing at next-day open vs signal close
MIN_SLIPPAGE = 0.0005  # Minimum 0.05% = 5 basis points (liquid stocks like SPY)
SLIPPAGE_ATR_MULTIPLIER = 0.1  # 10% of ATR (was 30% - too pessimistic)

# Minimum trades required for statistical significance
# For swing trading (1-2 trades/month), we need a lower bar:
# - 30 trades is ideal for reliable SQN estimates
# - 20 trades is acceptable with reduced confidence (allows stable stocks)
# - With 750+ days of history (3+ years), most stocks can hit 20+
# - Existing safeguards (shrinkage, min improvement) help prevent overfitting
MIN_TRADES_FOR_SIGNIFICANCE = 20  # Lowered from 30 to allow stable stocks

# Threshold for "full confidence" - stocks with fewer trades get flagged
FULL_CONFIDENCE_TRADES = 30


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    sqn: float                      # System Quality Number (Net, after costs)
    expectancy: float               # Average P&L per trade
    total_trades: int               # Number of trades
    win_rate: float                 # Percentage of winning trades
    avg_win: float                  # Average winning trade %
    avg_loss: float                 # Average losing trade %
    max_drawdown: float             # Maximum drawdown %
    profit_factor: float            # Gross profit / Gross loss
    total_return: float             # Total return %

    # Gross SQN (before costs) - shows signal quality independent of tradability
    gross_sqn: float = 0.0

    # Trade details for analysis
    trades: Optional[pd.DataFrame] = None
    
    def is_valid(self) -> bool:
        """Check if result meets minimum statistical requirements."""
        return (
            self.total_trades >= MIN_TRADES_FOR_SIGNIFICANCE and
            self.expectancy > 0 and
            not np.isnan(self.sqn) and
            self.sqn > 0
        )


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


# ============================================================================
# Adaptive Slippage & Next-Day-Open Execution
# ============================================================================

def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range for adaptive slippage.

    Returns ATR as a percentage of close price.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()

    # Return as percentage of close
    return (atr / close).fillna(0.01)  # Default 1% if insufficient data


def get_adaptive_slippage(atr_pct: float) -> float:
    """
    Calculate slippage based on volatility.

    Higher volatility stocks have larger overnight gaps,
    so we scale slippage with ATR.

    Args:
        atr_pct: ATR as percentage of price (e.g., 0.02 = 2%)

    Returns:
        Slippage as decimal (e.g., 0.002 = 0.2%)
    """
    # Scale slippage with ATR, with minimum floor
    adaptive_slippage = atr_pct * SLIPPAGE_ATR_MULTIPLIER
    return max(MIN_SLIPPAGE, adaptive_slippage)


def calculate_next_day_open_returns(
    df: pd.DataFrame,
    horizon: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate forward returns using next-day-open execution.

    This models realistic execution where:
    - Signal triggers at close of day N
    - Entry happens at open of day N+1
    - Exit happens at close of day N+1+horizon

    Args:
        df: DataFrame with OHLCV data
        horizon: Number of trading days to hold

    Returns:
        Tuple of (forward_returns, entry_gaps, atr_series)
        - forward_returns: Return from open[N+1] to close[N+1+horizon]
        - entry_gaps: Gap from close[N] to open[N+1] (for tracking)
        - atr_series: ATR percentage for adaptive slippage
    """
    close = df['close']
    open_price = df['open']

    # Entry gap: difference between signal day close and next day open
    # This is what we "miss" by not being able to trade at close
    entry_gap = (open_price.shift(-1) - close) / close

    # Exit price: close of day (entry_day + horizon)
    # Entry day is N+1, so exit is at N+1+horizon = N + (horizon+1)
    exit_close = close.shift(-(horizon + 1))
    entry_open = open_price.shift(-1)

    # Forward return from actual entry (next-day open) to exit (close)
    forward_returns = (exit_close - entry_open) / entry_open
    forward_returns = forward_returns.fillna(0)

    # ATR for adaptive slippage
    atr_pct = calculate_atr_series(df)

    return forward_returns, entry_gap.fillna(0), atr_pct


# ============================================================================
# Weight Application
# ============================================================================

def apply_weights_to_signals(
    signals_df: pd.DataFrame,
    weights: Dict[str, float],
    context_series: pd.Series = None,
    params: OptimizationParams = None,
    regime_series: pd.Series = None
) -> pd.Series:
    """
    Apply indicator weights to normalized signals with context and regime adjustment.

    Args:
        signals_df: DataFrame with signal columns (rsi_signal, macd_signal, etc.)
        weights: Dict of {indicator_name: weight}
        context_series: Optional Series of context ('trend', 'value', 'neutral')
        params: OptimizationParams for context multipliers and regime adjustments (uses DEFAULT_PARAMS if None)
        regime_series: Optional Series of market regime (BULL_QUIET, BEAR_VOLATILE, etc.)

    Returns:
        Series of weighted composite scores
    """
    # Use default params if not provided (backward compatibility)
    if params is None:
        params = DEFAULT_PARAMS

    # Map weight keys to signal column names
    signal_columns = {
        'rsi': 'rsi_signal',
        'macd': 'macd_signal',
        'bollinger': 'bollinger_signal',
        'adx': 'adx_signal',
        'cmf': 'cmf_signal',
        'momentum': 'momentum_signal',
        'volume': 'volume_signal',
        'rvol': 'rvol_signal',
        'sma': 'sma_signal',
        'position': 'position_signal',
        'squeeze': 'squeeze_signal',
        'cross': 'cross_signal',
        # Relative strength indicators (vs sector)
        'rel_momentum': 'rel_momentum_signal',
        'rs_ratio': 'rs_ratio_signal',
    }

    # Mean-reversion indicators (reduce weight in trending markets)
    mean_reversion_indicators = {'bollinger', 'position'}
    # Trend-following indicators (boost weight in trending markets)
    trend_following_indicators = {'sma', 'macd', 'adx'}

    # Get context multipliers from params
    mr_trend = params.context_multipliers.mr_trend
    mr_value = params.context_multipliers.mr_value
    tf_trend = params.context_multipliers.tf_trend

    # Calculate weighted sum
    weighted_sum = pd.Series(0.0, index=signals_df.index)
    total_weight = pd.Series(0.0, index=signals_df.index)

    # Pre-compute regime multipliers for each indicator if regime_series is provided
    # This is vectorized for performance
    regime_multipliers = {}
    if regime_series is not None:
        unique_regimes = regime_series.unique()
        for indicator in weights.keys():
            # Build a series of multipliers for this indicator based on regime
            multiplier_series = pd.Series(1.0, index=signals_df.index)
            for regime in unique_regimes:
                regime_mask = regime_series == regime
                adjustments = params.regime_adjustments.get_adjustments(regime)
                if adjustments and indicator in adjustments:
                    multiplier_series[regime_mask] = adjustments[indicator]
            regime_multipliers[indicator] = multiplier_series

    for indicator, base_weight in weights.items():
        col_name = signal_columns.get(indicator)
        if col_name and col_name in signals_df.columns:
            signal = signals_df[col_name].fillna(0)

            # Apply context adjustment using configurable multipliers
            if context_series is not None:
                is_trending = context_series == 'trend'
                is_value = context_series == 'value'

                if indicator in mean_reversion_indicators:
                    # Reduce weight in trending markets, boost in value contexts
                    weight = np.where(is_trending, base_weight * mr_trend,
                             np.where(is_value, base_weight * mr_value, base_weight))
                elif indicator in trend_following_indicators:
                    # Boost in trending markets
                    weight = np.where(is_trending, base_weight * tf_trend, base_weight)
                else:
                    weight = base_weight

                weight = pd.Series(weight, index=signals_df.index)
            else:
                weight = base_weight

            # Apply regime-specific multipliers (from learned params)
            if indicator in regime_multipliers:
                if isinstance(weight, (int, float)):
                    weight = pd.Series(weight, index=signals_df.index)
                weight = weight * regime_multipliers[indicator]

            weighted_sum += signal * weight
            total_weight += np.abs(weight)

    # Normalize by total weight if non-zero
    result = weighted_sum / total_weight.replace(0, 1)

    return result


# ============================================================================
# Vectorized Simulation Engine
# ============================================================================

def fast_simulate(
    df: pd.DataFrame,
    weights: Dict[str, float],
    horizon: int = 3,
    buy_threshold: float = None,
    sell_threshold: float = None,
    transaction_cost: float = MIN_TRANSACTION_COST,
    apply_regime_filter: bool = True,
    params: OptimizationParams = None,
    sector_df: pd.DataFrame = None
) -> SimulationResult:
    """
    Vectorized backtesting simulation.

    Uses NumPy/Pandas operations for speed (100-1000x faster than loops).

    Args:
        df: DataFrame with OHLCV data
        weights: Dict of {indicator: weight}
        horizon: Holding period in days (3 for swing, 15 for trend)
        buy_threshold: Score threshold for long entry (uses params if None)
        sell_threshold: Score threshold for short entry (uses params if None)
        transaction_cost: Per-trade cost as decimal (0.001 = 0.1%)
        apply_regime_filter: If True, force cash in BEAR_VOLATILE
        params: OptimizationParams for thresholds/multipliers (uses DEFAULT_PARAMS if None)
        sector_df: Optional sector ETF DataFrame for relative strength calculations

    Returns:
        SimulationResult with all metrics
    """
    # Use default params if not provided (backward compatibility)
    if params is None:
        params = DEFAULT_PARAMS

    # Use params thresholds if not explicitly provided
    if buy_threshold is None:
        buy_threshold = params.trade_thresholds.buy_threshold
    if sell_threshold is None:
        sell_threshold = params.trade_thresholds.sell_threshold
    import logging
    logger = logging.getLogger(__name__)
    
    n = len(df)
    
    if n < 100:
        logger.warning(f"[WFO SIM] Data too short: {n} rows (need 100)")
        return _empty_result()
    
    # 1. Calculate indicator signals
    logger.info(f"[WFO SIM] Calculating indicator signals for {n} rows...")

    # Extract direction parameters from weights (keys ending in '_dir')
    directions = {k: v for k, v in weights.items() if k.endswith('_dir')}
    if directions:
        logger.info(f"[WFO SIM] Using signal directions: {directions}")

    signals_df = calculate_indicator_signals(df, params=params, directions=directions, sector_df=sector_df)
    logger.info(f"[WFO SIM] Signal columns: {list(signals_df.columns)}")
    
    # Debug: show signal stats
    for col in signals_df.columns:
        if col.endswith('_signal'):
            vals = signals_df[col].dropna()
            logger.info(f"[WFO SIM] {col}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")
    
    # 2. Calculate market regime (for filtering)
    if apply_regime_filter:
        regimes = calculate_market_regime(df)
    else:
        regimes = pd.Series(MarketRegime.NEUTRAL_CHOP.value, index=df.index)
    
    # 2b. Detect buying context (like TypeScript detectBuyingContext)
    # TREND: Price above SMA50 and SMA200
    # VALUE: Price below SMA50 with oversold RSI or near lower Bollinger
    sma50 = df['close'].rolling(50).mean()
    sma200 = df['close'].rolling(200).mean()
    price_above_sma50 = df['close'] > sma50
    price_above_sma200 = df['close'] > sma200
    
    # Value conditions
    price_below_sma50 = df['close'] < sma50
    rsi_oversold = signals_df['rsi_signal'] > 0.5 if 'rsi_signal' in signals_df.columns else False
    bb_oversold = signals_df['bollinger_signal'] > 0.5 if 'bollinger_signal' in signals_df.columns else False
    
    # Context series: 'trend', 'value', 'neutral'
    context = pd.Series('neutral', index=df.index)
    context = np.where(price_above_sma50 & price_above_sma200, 'trend', context)
    context = np.where(price_below_sma50 & (rsi_oversold | bb_oversold), 'value', context)
    context = pd.Series(context, index=df.index)
    
    trend_pct = (context == 'trend').mean() * 100
    value_pct = (context == 'value').mean() * 100
    logger.info(f"[WFO SIM] Context: {trend_pct:.1f}% trend, {value_pct:.1f}% value, {100-trend_pct-value_pct:.1f}% neutral")
    
    # 3. Apply weights to get composite score (with context and regime adjustment)
    # Pass regime_series to apply learned regime-specific multipliers from params
    composite_score = apply_weights_to_signals(
        signals_df, weights, context, params,
        regime_series=regimes if apply_regime_filter else None
    )
    logger.info(f"[WFO SIM] Composite score: min={composite_score.min():.3f}, max={composite_score.max():.3f}, mean={composite_score.mean():.3f}")
    
    # 4. Generate trade signals (vectorized)
    # Long signal: score >= buy_threshold (inclusive)
    # Short signal: score <= sell_threshold (inclusive)
    long_signals = composite_score >= buy_threshold
    short_signals = composite_score <= sell_threshold
    logger.info(f"[WFO SIM] Long signals: {long_signals.sum()}, Short signals: {short_signals.sum()}, threshold={buy_threshold}/{sell_threshold}")
    
    # 5. Apply regime filter (zero out signals in un-tradeable regimes)
    if apply_regime_filter:
        # In BEAR_VOLATILE, mean reversion is suicide
        untradeable_mask = regimes == MarketRegime.BEAR_VOLATILE.value
        long_signals = long_signals & ~untradeable_mask
        short_signals = short_signals & ~untradeable_mask
    
    # 6. Calculate forward returns using NEXT-DAY-OPEN execution
    # This models realistic execution: signal at close -> entry at next day's open
    forward_returns, entry_gaps, atr_series = calculate_next_day_open_returns(df, horizon)
    avg_gap = entry_gaps.abs().mean() * 100
    logger.info(f"[WFO SIM] Forward returns (next-day-open): {forward_returns.notna().sum()} valid, min={forward_returns.min():.4f}, max={forward_returns.max():.4f}")
    logger.info(f"[WFO SIM] Avg overnight gap: {avg_gap:.2f}% (accounted for in returns)")
    
    # 7. Calculate strategy returns
    # Long: get the forward return
    # Short: get negative of forward return
    long_returns = np.where(long_signals, forward_returns, 0)
    short_returns = np.where(short_signals, -forward_returns, 0)
    strategy_returns = long_returns + short_returns
    logger.info(f"[WFO SIM] Strategy returns: {np.sum(strategy_returns != 0)} non-zero trades")
    
    # 8. Identify DISTINCT trades (not holding days!)
    # A trade = entry when signal goes 0->1, exit after horizon days
    # We use the signal transitions, not the daily returns
    trade_mask = (long_signals | short_signals).astype(int).values
    
    # Detect trade entries (0->1 transitions)
    trade_entries = np.diff(trade_mask, prepend=0) == 1
    trade_ids = np.cumsum(trade_entries) * trade_mask
    
    # Count actual distinct trades
    unique_trade_ids = np.unique(trade_ids)
    unique_trade_ids = unique_trade_ids[unique_trade_ids != 0]
    num_trades = len(unique_trade_ids)
    
    logger.info(f"[WFO SIM] Distinct trades: {num_trades} (not holding days)")
    
    # 9. Aggregate returns PER TRADE (not per day)
    # Each trade gets one return = the forward return at entry - ADAPTIVE transaction cost
    trade_returns = []
    trade_slippages = []  # Track for logging

    for tid in unique_trade_ids:
        # Get the entry day for this trade
        entry_mask = (trade_ids == tid)
        entry_idx = np.argmax(entry_mask)  # First True index

        # Get the forward return at entry (already accounts for next-day-open)
        entry_return = forward_returns.iloc[entry_idx]

        # Direction: long or short
        is_long = long_signals.iloc[entry_idx] if hasattr(long_signals, 'iloc') else long_signals[entry_idx]
        direction = 1 if is_long else -1

        # Calculate ADAPTIVE slippage based on ATR at entry
        entry_atr = atr_series.iloc[entry_idx]
        adaptive_slip = get_adaptive_slippage(entry_atr)
        trade_slippages.append(adaptive_slip)

        # Round-trip cost = 2 * (base transaction cost + adaptive slippage)
        round_trip_cost = 2 * (transaction_cost + adaptive_slip)

        # Net return = directional return - costs
        net_return = (direction * entry_return) - round_trip_cost
        trade_returns.append(net_return)

    trade_returns = np.array(trade_returns)
    avg_slippage = np.mean(trade_slippages) * 100 if trade_slippages else 0
    logger.info(f"[WFO SIM] Avg adaptive slippage: {avg_slippage:.2f}% per side")
    
    # 10. Build trade log (for detailed analysis)
    trades_df = _build_trade_log(
        df, long_signals, short_signals,
        forward_returns, trade_ids,
        horizon, regimes, transaction_cost,
        atr_series, entry_gaps
    )
    
    # 11. Calculate performance metrics on TRADE returns (not daily returns)
    return _calculate_metrics_by_trade(trade_returns, trades_df)


def _empty_result() -> SimulationResult:
    """Return empty result for insufficient data."""
    return SimulationResult(
        sqn=0.0,
        expectancy=0.0,
        total_trades=0,
        win_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        total_return=0.0,
        gross_sqn=0.0,
        trades=None
    )


def _build_trade_log(
    df: pd.DataFrame,
    long_signals: pd.Series,
    short_signals: pd.Series,
    forward_returns: pd.Series,
    trade_ids: np.ndarray,
    horizon: int,
    regimes: pd.Series,
    base_transaction_cost: float,
    atr_series: pd.Series,
    entry_gaps: pd.Series
) -> pd.DataFrame:
    """
    Build detailed trade log for analysis - one row per TRADE.

    Uses next-day-open execution model:
    - Signal date: day when signal triggered (at close)
    - Entry date: next day (N+1)
    - Entry price: open of day N+1
    - Exit price: close of day N+1+horizon
    """
    unique_trade_ids = np.unique(trade_ids)
    unique_trade_ids = unique_trade_ids[unique_trade_ids != 0]

    if len(unique_trade_ids) == 0:
        return pd.DataFrame()

    trade_records = []
    for tid in unique_trade_ids:
        entry_mask = (trade_ids == tid)
        signal_idx = np.argmax(entry_mask)  # Signal day index

        is_long = long_signals.iloc[signal_idx] if hasattr(long_signals, 'iloc') else long_signals[signal_idx]

        # Signal date (when signal was generated at close)
        signal_date = df.iloc[signal_idx]['date'] if 'date' in df.columns else signal_idx

        # Entry day is signal_idx + 1, exit day is signal_idx + 1 + horizon
        entry_idx = signal_idx + 1
        exit_idx = min(signal_idx + 1 + horizon, len(df) - 1)

        # Prices: signal close, actual entry open, exit close
        signal_close_price = df.iloc[signal_idx]['close']

        # Entry at next day's open (realistic execution)
        if entry_idx < len(df):
            entry_price = df.iloc[entry_idx]['open']
            entry_date = df.iloc[entry_idx]['date'] if 'date' in df.columns else entry_idx
        else:
            entry_price = signal_close_price  # Fallback
            entry_date = signal_date

        # Exit at close of entry + horizon
        if exit_idx < len(df):
            exit_price = df.iloc[exit_idx]['close']
        else:
            exit_price = entry_price  # Fallback

        # Forward return already computed from open[N+1] to close[N+1+horizon]
        forward_ret = forward_returns.iloc[signal_idx]
        regime = regimes.iloc[signal_idx] if hasattr(regimes, 'iloc') else regimes[signal_idx]

        # Overnight gap (what we "missed" vs close-to-close)
        entry_gap = entry_gaps.iloc[signal_idx] if signal_idx < len(entry_gaps) else 0

        # Adaptive slippage based on ATR
        entry_atr = atr_series.iloc[signal_idx] if signal_idx < len(atr_series) else 0.01
        adaptive_slip = get_adaptive_slippage(entry_atr)
        round_trip_cost = 2 * (base_transaction_cost + adaptive_slip)

        direction = 1 if is_long else -1
        net_return = (direction * forward_ret) - round_trip_cost

        trade_records.append({
            'trade_id': tid,
            'signal_date': signal_date,
            'entry_date': entry_date,
            'signal_idx': signal_idx,
            'entry_idx': entry_idx,
            'direction': 'long' if is_long else 'short',
            'signal_close': signal_close_price,
            'entry_price': entry_price,  # Actual entry at next-day open
            'exit_price': exit_price,
            'entry_gap_pct': entry_gap * 100,  # Overnight gap %
            'horizon': horizon,
            'forward_return': forward_ret,
            'net_return': net_return,
            'market_regime': regime,
            'atr_pct': entry_atr * 100,
            'slippage_applied': adaptive_slip * 100,
            'costs': round_trip_cost
        })

    trades = pd.DataFrame(trade_records)

    # Calculate P&L percentage
    if len(trades) > 0:
        trades['pnl_pct'] = trades['net_return'] * 100

    return trades


def _calculate_metrics_by_trade(
    trade_returns: np.ndarray,
    trades_df: pd.DataFrame
) -> SimulationResult:
    """
    Calculate performance metrics from TRADE returns (not daily returns).
    
    This is the correct way to calculate SQN - based on distinct trades,
    not holding days. A 20-day trade counts as 1 trade, not 20.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Filter valid returns (excluding NaN)
    valid_returns = trade_returns[~np.isnan(trade_returns)]
    total_trades = len(valid_returns)
    
    logger.info(f"[WFO SIM] Metrics by trade: {total_trades} distinct trades")
    
    if total_trades == 0:
        return _empty_result()
    
    # Separate wins and losses
    wins = valid_returns[valid_returns > 0]
    losses = valid_returns[valid_returns < 0]
    
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    avg_win = np.mean(wins) * 100 if len(wins) > 0 else 0
    avg_loss = np.mean(losses) * 100 if len(losses) > 0 else 0
    
    # Expectancy (average return per TRADE)
    expectancy = np.mean(valid_returns)
    
    # Standard deviation of TRADE returns
    std_dev = np.std(valid_returns) if total_trades > 1 else 0.0001
    
    # SQN = sqrt(N) * (Expectancy / StdDev)
    # N is now correct: number of distinct trades
    if std_dev > 0 and total_trades >= MIN_TRADES_FOR_SIGNIFICANCE:
        sqn = np.sqrt(total_trades) * (expectancy / std_dev)
    else:
        sqn = 0.0

    # Calculate Gross SQN (before costs) to show signal quality independent of tradability
    # Recover gross returns by adding back costs
    # Note: trades_df['costs'] contains the cost per trade
    gross_sqn = 0.0
    if trades_df is not None and not trades_df.empty and 'costs' in trades_df.columns:
        avg_cost = trades_df['costs'].mean()
        gross_expectancy = expectancy + avg_cost
        gross_sqn = np.sqrt(total_trades) * (gross_expectancy / std_dev) if std_dev > 0 else 0
        logger.info(f"[WFO SIM] Net SQN={sqn:.3f} | Gross SQN={gross_sqn:.3f} | Cost Drag={gross_sqn - sqn:.3f}")
    else:
        # No cost info available - gross equals net
        gross_sqn = sqn

    logger.info(f"[WFO SIM] SQN={sqn:.3f}, Expectancy={expectancy*100:.2f}%, WinRate={win_rate*100:.1f}%")
    
    # Profit factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Total return (sum of trade returns)
    total_return = np.sum(valid_returns) * 100
    
    # Max drawdown (from sequential trade returns)
    cumulative = np.cumsum(valid_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
    
    return SimulationResult(
        sqn=round(sqn, 4),
        expectancy=round(expectancy * 100, 4),  # As percentage
        total_trades=total_trades,
        win_rate=round(win_rate * 100, 2),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        max_drawdown=round(max_drawdown, 2),
        profit_factor=round(profit_factor, 4),
        total_return=round(total_return, 2),
        gross_sqn=round(gross_sqn, 4),
        trades=trades_df
    )


# ============================================================================
# Batch Simulation (for optimizer)
# ============================================================================

def simulate_weight_grid(
    df: pd.DataFrame,
    indicator: str,
    base_weights: Dict[str, float],
    test_values: List[float],
    horizon: int = 3,
    **kwargs
) -> List[Tuple[float, SimulationResult]]:
    """
    Simulate a grid of weight values for a single indicator.
    
    Used by the optimizer to find optimal weights.
    
    Args:
        df: Price data
        indicator: Indicator to vary
        base_weights: Base weight configuration
        test_values: List of weight values to test
        horizon: Holding period
        **kwargs: Additional args for fast_simulate
        
    Returns:
        List of (weight_value, SimulationResult) tuples
    """
    results = []
    
    for weight_value in test_values:
        test_weights = base_weights.copy()
        test_weights[indicator] = weight_value
        
        result = fast_simulate(df, test_weights, horizon=horizon, **kwargs)
        results.append((weight_value, result))
    
    return results


# ============================================================================
# Validation Helper
# ============================================================================

def validate_simulation_params(
    weights: Dict[str, float],
    transaction_cost: float
) -> List[str]:
    """
    Validate simulation parameters against pre-flight constraints.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    for key, value in weights.items():
        # Direction parameters have range [-1, 1]
        if key.endswith('_dir'):
            if value < -1.0 or value > 1.0:
                errors.append(f"Direction {key}={value} out of range [-1.0, 1.0]")
        else:
            # Regular weights have range [0, 2.5]
            if value < 0.0 or value > 2.5:
                errors.append(f"Weight {key}={value} out of range [0.0, 2.5]")

    # Constraint 2: Transaction cost >= 0.1%
    if transaction_cost < MIN_TRANSACTION_COST:
        errors.append(
            f"Transaction cost {transaction_cost*100:.2f}% below minimum "
            f"{MIN_TRANSACTION_COST*100:.1f}%"
        )

    return errors

