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


# ============================================================================
# Configuration
# ============================================================================

# Minimum transaction cost (0.1% = 10 basis points)
MIN_TRANSACTION_COST = 0.001

# Slippage estimate (0.05% = 5 basis points each way)
SLIPPAGE = 0.0005

# Minimum trades required for statistical significance
# For swing trading (1-2 trades/month), we need a lower bar:
# - 30 is too aggressive for 2 years of daily data
# - 10 trades gives rough but usable SQN estimates
# - Combined with 5-year data fetch, this becomes more robust
MIN_TRADES_FOR_SIGNIFICANCE = 10


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    sqn: float                      # System Quality Number
    expectancy: float               # Average P&L per trade
    total_trades: int               # Number of trades
    win_rate: float                 # Percentage of winning trades
    avg_win: float                  # Average winning trade %
    avg_loss: float                 # Average losing trade %
    max_drawdown: float             # Maximum drawdown %
    profit_factor: float            # Gross profit / Gross loss
    total_return: float             # Total return %
    
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
# Weight Application
# ============================================================================

def apply_weights_to_signals(
    signals_df: pd.DataFrame,
    weights: Dict[str, float],
    context_series: pd.Series = None
) -> pd.Series:
    """
    Apply indicator weights to normalized signals with context adjustment.
    
    Args:
        signals_df: DataFrame with signal columns (rsi_signal, macd_signal, etc.)
        weights: Dict of {indicator_name: weight}
        context_series: Optional Series of context ('trend', 'value', 'neutral')
        
    Returns:
        Series of weighted composite scores
    """
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
    }
    
    # Mean-reversion indicators (reduce weight in trending markets)
    mean_reversion_indicators = {'bollinger', 'position'}
    # Trend-following indicators (boost weight in trending markets)  
    trend_following_indicators = {'sma', 'macd', 'adx'}
    
    # Calculate weighted sum
    weighted_sum = pd.Series(0.0, index=signals_df.index)
    total_weight = pd.Series(0.0, index=signals_df.index)
    
    for indicator, base_weight in weights.items():
        col_name = signal_columns.get(indicator)
        if col_name and col_name in signals_df.columns:
            signal = signals_df[col_name].fillna(0)
            
            # Apply context adjustment (like TypeScript does)
            if context_series is not None:
                is_trending = context_series == 'trend'
                is_value = context_series == 'value'
                
                if indicator in mean_reversion_indicators:
                    # Reduce weight in trending markets (0.7x), boost in value (1.5x)
                    weight = np.where(is_trending, base_weight * 0.7,
                             np.where(is_value, base_weight * 1.5, base_weight))
                elif indicator in trend_following_indicators:
                    # Boost in trending markets (1.3x)
                    weight = np.where(is_trending, base_weight * 1.3, base_weight)
                else:
                    weight = base_weight
                
                weight = pd.Series(weight, index=signals_df.index)
            else:
                weight = base_weight
            
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
    buy_threshold: float = 0.2,
    sell_threshold: float = -0.2,
    transaction_cost: float = MIN_TRANSACTION_COST,
    apply_regime_filter: bool = True
) -> SimulationResult:
    """
    Vectorized backtesting simulation.
    
    Uses NumPy/Pandas operations for speed (100-1000x faster than loops).
    
    Args:
        df: DataFrame with OHLCV data
        weights: Dict of {indicator: weight}
        horizon: Holding period in days (3 for swing, 15 for trend)
        buy_threshold: Score threshold for long entry (default 0.2)
        sell_threshold: Score threshold for short entry (default -0.2)
        transaction_cost: Per-trade cost as decimal (0.001 = 0.1%)
        apply_regime_filter: If True, force cash in BEAR_VOLATILE
        
    Returns:
        SimulationResult with all metrics
    """
    import logging
    logger = logging.getLogger(__name__)
    
    n = len(df)
    
    if n < 100:
        logger.warning(f"[WFO SIM] Data too short: {n} rows (need 100)")
        return _empty_result()
    
    # 1. Calculate indicator signals
    logger.info(f"[WFO SIM] Calculating indicator signals for {n} rows...")
    signals_df = calculate_indicator_signals(df)
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
    
    # 3. Apply weights to get composite score (with context adjustment)
    composite_score = apply_weights_to_signals(signals_df, weights, context)
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
    
    # 6. Calculate forward returns (shifted to prevent look-ahead)
    forward_returns = df['close'].pct_change(horizon).shift(-horizon).fillna(0)
    logger.info(f"[WFO SIM] Forward returns: {forward_returns.notna().sum()} valid, min={forward_returns.min():.4f}, max={forward_returns.max():.4f}")
    
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
    # Each trade gets one return = the forward return at entry - transaction cost
    round_trip_cost = 2 * (transaction_cost + SLIPPAGE)
    
    trade_returns = []
    for tid in unique_trade_ids:
        # Get the entry day for this trade
        entry_mask = (trade_ids == tid)
        entry_idx = np.argmax(entry_mask)  # First True index
        
        # Get the forward return at entry (already accounts for horizon)
        entry_return = forward_returns.iloc[entry_idx]
        
        # Direction: long or short
        is_long = long_signals.iloc[entry_idx] if hasattr(long_signals, 'iloc') else long_signals[entry_idx]
        direction = 1 if is_long else -1
        
        # Net return = directional return - costs
        net_return = (direction * entry_return) - round_trip_cost
        trade_returns.append(net_return)
    
    trade_returns = np.array(trade_returns)
    
    # 10. Build trade log (for detailed analysis)
    trades_df = _build_trade_log(
        df, long_signals, short_signals, 
        forward_returns, trade_ids,
        horizon, regimes, round_trip_cost
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
    round_trip_cost: float
) -> pd.DataFrame:
    """Build detailed trade log for analysis - one row per TRADE."""
    unique_trade_ids = np.unique(trade_ids)
    unique_trade_ids = unique_trade_ids[unique_trade_ids != 0]
    
    if len(unique_trade_ids) == 0:
        return pd.DataFrame()
    
    trade_records = []
    for tid in unique_trade_ids:
        entry_mask = (trade_ids == tid)
        entry_idx = np.argmax(entry_mask)  # First True index
        
        is_long = long_signals.iloc[entry_idx] if hasattr(long_signals, 'iloc') else long_signals[entry_idx]
        
        entry_date = df.iloc[entry_idx]['date'] if 'date' in df.columns else entry_idx
        entry_price = df.iloc[entry_idx]['close']
        forward_ret = forward_returns.iloc[entry_idx]
        regime = regimes.iloc[entry_idx] if hasattr(regimes, 'iloc') else regimes[entry_idx]
        
        direction = 1 if is_long else -1
        net_return = (direction * forward_ret) - round_trip_cost
        
        trade_records.append({
            'trade_id': tid,
            'entry_date': entry_date,
            'entry_idx': entry_idx,
            'direction': 'long' if is_long else 'short',
            'entry_price': entry_price,
            'horizon': horizon,
            'forward_return': forward_ret,
            'net_return': net_return,
            'market_regime': regime,
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
    
    # Constraint 1: Weights must be in range 0.0 to 2.5
    for indicator, weight in weights.items():
        if weight < 0.0 or weight > 2.5:
            errors.append(f"Weight {indicator}={weight} out of range [0.0, 2.5]")
    
    # Constraint 2: Transaction cost >= 0.1%
    if transaction_cost < MIN_TRANSACTION_COST:
        errors.append(
            f"Transaction cost {transaction_cost*100:.2f}% below minimum "
            f"{MIN_TRANSACTION_COST*100:.1f}%"
        )
    
    return errors

