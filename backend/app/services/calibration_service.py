"""
Walk-Forward Optimization Calibration Service

Main controller for the WFO engine. Orchestrates:
- Data fetching from SQLite cache
- Rolling window management
- Optimization execution
- Result storage and SSE streaming

CONSTRAINTS (Pre-Flight Checklist):
1. Only optimize WEIGHTS (0.0-2.5), NOT indicator periods
2. Transaction cost >= 0.1% (realistic friction)
3. Force cash in un-tradeable regimes (BEAR_VOLATILE)
"""

import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logger = logging.getLogger(__name__)

# Process pool for CPU-bound optimization work (bypasses GIL)
# Use 'spawn' to avoid issues with forking in async context
_process_pool: Optional[ProcessPoolExecutor] = None

def get_process_pool() -> ProcessPoolExecutor:
    """Get or create the process pool for parallel optimization."""
    global _process_pool
    if _process_pool is None:
        # Use number of CPUs minus 1 to leave room for the main process
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"[WFO] Creating process pool with {max_workers} workers")
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _process_pool

from ..models import (
    PriceHistory,
    CalibrationWeights,
    CalibrationWindow,
    CalibrationTrade
)
from .indicators import calculate_all_indicators
from .regime import calculate_market_regime
from .wfo_simulator import fast_simulate, SimulationResult
from .wfo_optimizer import (
    two_pass_coordinate_descent,
    optimize_for_ticker,
    get_adaptive_window,
    DEFAULT_WEIGHTS,
    InsufficientVolatilityError,
    FullOptimizationResult,
    OptimizationResult,
    OptimizerType,
    DEFAULT_OPTIMIZER
)
from .sector_data import ensure_sector_etf_data


# ============================================================================
# Configuration
# ============================================================================

# Rolling window configuration (Standard WFO - 4:1 ratio)
TRAIN_WINDOW_MONTHS = 24      # 2 years - enough for all indicators (200-day SMA, 52-week range)
TEST_WINDOW_MONTHS = 6        # 6 months - captures multiple market cycles, more trades per window
ROLL_STEP_MONTHS = 3          # Quarterly rolls - reduces redundancy while tracking regime changes

# Trading days approximation
DAYS_PER_MONTH = 21

# Horizons to calibrate
HORIZONS = [3, 15]  # Swing (3 days) and Trend (15 days)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CalibrationProgress:
    """Progress update for SSE streaming."""
    ticker: str
    horizon: int
    stage: str           # 'loading', 'optimizing', 'testing', 'saving', 'complete', 'error'
    progress: float      # 0-100
    current_indicator: Optional[str] = None
    message: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    train_sqn: Optional[float] = None
    test_sqn: Optional[float] = None
    gross_sqn: Optional[float] = None  # Gross SQN (before costs) - shows signal quality
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class CalibrationErrorResult:
    """Result when calibration fails for a horizon."""
    ticker: str
    horizon: int
    error_type: str  # 'insufficient_trades', 'insufficient_data', etc.
    message: str
    trades_found: int = 0
    window_days: int = 0


@dataclass
class WindowResult:
    """Result of a single rolling window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_days: int
    weights: Dict[str, float]
    train_sqn: float
    test_sqn: float
    trades_count: int
    expectancy: float
    test_gross_sqn: float = 0.0  # Gross SQN (before costs) for signal quality
    trades: Optional[pd.DataFrame] = None  # Out-of-sample trades for persistence


# ============================================================================
# Data Loading
# ============================================================================

async def load_price_history(
    db: AsyncSession,
    ticker: str,
    min_days: int = 750
) -> pd.DataFrame:
    """
    Load price history from SQLite cache.
    
    Following GUIDELINES.md: SQLite-first approach, no API calls.
    
    Defaults to 750 days (~3 years) to allow newer stocks while ensuring
    sufficient trade count for swing trading strategies.
    
    Args:
        db: Database session
        ticker: Stock ticker
        min_days: Minimum required history (default: 3 years)
        
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If insufficient data in cache
    """
    logger.info(f"[WFO] Loading price history for {ticker} (min_days={min_days})")
    
    result = await db.execute(
        select(PriceHistory)
        .where(PriceHistory.ticker == ticker)
        .order_by(PriceHistory.date.asc())
    )
    
    rows = result.scalars().all()
    
    logger.info(f"[WFO] Found {len(rows)} days of price history for {ticker}")
    
    if len(rows) < min_days:
        raise ValueError(
            f"{ticker}: Only {len(rows)} days in cache (need {min_days}). "
            f"Please fetch more history first."
        )
    
    df = pd.DataFrame([
        {
            'date': row.date,
            'open': row.open,
            'high': row.high,
            'low': row.low,
            'close': row.close,
            'volume': row.volume
        }
        for row in rows
    ])
    
    # Validation: Ensure strict daily granularity (no intraday pollution)
    if not df.empty:
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]
        unique_dates = df['date'].nunique()
        total_rows = len(df)
        
        logger.info(f"[WFO] Price history for {ticker}: {start_date} to {end_date}")
        logger.info(f"[WFO] Data check: {total_rows} rows, {unique_dates} unique dates")
        
        if total_rows != unique_dates:
            logger.warning(f"[WFO] DUPLICATE DATES DETECTED for {ticker}: {total_rows - unique_dates} duplicates. This may affect calibration.")
    
    return df


# ============================================================================
# Rolling Window Generator
# ============================================================================

def generate_rolling_windows(
    df: pd.DataFrame,
    train_months: int = TRAIN_WINDOW_MONTHS,
    test_months: int = TEST_WINDOW_MONTHS,
    roll_step_months: int = ROLL_STEP_MONTHS
) -> List[Dict]:
    """
    Generate rolling train/test window boundaries.
    
    Implements the 4:1 rolling window architecture:
    [ Jan | Feb | Mar | Apr | May | Jun ] -> Trade July
    [ Feb | Mar | Apr | May | Jun | Jul ] -> Trade Aug
    
    Args:
        df: Price data with 'date' column
        train_months: Training window size
        test_months: Test window size
        roll_step_months: Step size for rolling
        
    Returns:
        List of window dicts with start/end dates
    """
    windows = []
    
    train_days = train_months * DAYS_PER_MONTH
    test_days = test_months * DAYS_PER_MONTH
    step_days = roll_step_months * DAYS_PER_MONTH
    
    total_days = len(df)
    required_days = train_days + test_days
    
    if total_days < required_days:
        return []
    
    # Start from the earliest point where we have enough data
    start_idx = 0
    
    while start_idx + required_days <= total_days:
        train_end_idx = start_idx + train_days
        test_end_idx = train_end_idx + test_days
        
        if test_end_idx > total_days:
            break
        
        windows.append({
            'train_start_idx': start_idx,
            'train_end_idx': train_end_idx,
            'test_start_idx': train_end_idx,
            'test_end_idx': test_end_idx,
            'train_start': df.iloc[start_idx]['date'],
            'train_end': df.iloc[train_end_idx - 1]['date'],
            'test_start': df.iloc[train_end_idx]['date'],
            'test_end': df.iloc[test_end_idx - 1]['date'] if test_end_idx <= total_days else df.iloc[-1]['date']
        })
        
        start_idx += step_days
    
    return windows


# ============================================================================
# Rolling Window Helpers
# ============================================================================

def aggregate_weights_exponential_decay(
    window_results: List[WindowResult],
    decay: float = 0.1
) -> Dict[str, float]:
    """
    Aggregate weights from multiple windows using exponential decay.

    More recent windows get higher weight. Formula:
    weight_i = exp(-decay * (N - i)) for window i of N

    Args:
        window_results: List of WindowResult from rolling optimization
        decay: Decay rate (higher = more emphasis on recent)

    Returns:
        Aggregated weights dictionary
    """
    import numpy as np

    if not window_results:
        return DEFAULT_WEIGHTS.copy()

    n = len(window_results)

    # Calculate decay weights for each window
    decay_weights = [np.exp(-decay * (n - i - 1)) for i in range(n)]
    total_decay_weight = sum(decay_weights)
    normalized_weights = [w / total_decay_weight for w in decay_weights]

    # Aggregate indicator weights
    all_indicators = set()
    for wr in window_results:
        all_indicators.update(wr.weights.keys())

    aggregated = {}
    for indicator in all_indicators:
        weighted_sum = 0.0
        for i, wr in enumerate(window_results):
            weighted_sum += wr.weights.get(indicator, 1.0) * normalized_weights[i]
        aggregated[indicator] = round(weighted_sum, 2)

    return aggregated


def enrich_trades_with_exits(
    trades_df: pd.DataFrame,
    full_df: pd.DataFrame,
    window_offset: int,
    horizon: int
) -> pd.DataFrame:
    """
    Add/verify exit_date and exit_price in trades DataFrame.

    With next-day-open execution model:
    - signal_idx: day signal triggered (at close)
    - entry_idx: signal_idx + 1 (next-day open)
    - exit_idx: entry_idx + horizon (close of holding period)

    The simulator now computes exit_price, but we need exit_date for persistence.

    Args:
        trades_df: Trades from fast_simulate (has entry_date, entry_idx, signal_idx, etc.)
        full_df: Full price DataFrame
        window_offset: Index offset of the simulation window in full_df
        horizon: Holding period (days)

    Returns:
        DataFrame with exit_date column added
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    enriched = trades_df.copy()

    exit_dates = []

    for _, trade in enriched.iterrows():
        # Use entry_idx (actual entry day, N+1) relative to simulation window
        # Map back to full_df coordinates
        entry_idx = int(trade.get('entry_idx', trade.get('signal_idx', 0) + 1)) + window_offset
        exit_idx = min(entry_idx + horizon, len(full_df) - 1)

        exit_dates.append(str(full_df.iloc[exit_idx]['date']))

    enriched['exit_date'] = exit_dates

    # If exit_price wasn't set by simulator (backward compat), compute it
    if 'exit_price' not in enriched.columns:
        exit_prices = []
        for _, trade in enriched.iterrows():
            entry_idx = int(trade.get('entry_idx', trade.get('signal_idx', 0) + 1)) + window_offset
            exit_idx = min(entry_idx + horizon, len(full_df) - 1)
            exit_prices.append(float(full_df.iloc[exit_idx]['close']))
        enriched['exit_price'] = exit_prices

    return enriched


def optimize_single_window(
    df: pd.DataFrame,
    train_start_idx: int,
    train_end_idx: int,
    test_start_idx: int,
    test_end_idx: int,
    ticker: str,
    horizon: int,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    strategy_class: str = 'all',
    sector_df: pd.DataFrame = None
) -> WindowResult:
    """
    Optimize weights on training window, evaluate on test window.

    This is the core of Walk-Forward Optimization:
    1. Train on historical data (in-sample)
    2. Test on forward data (out-of-sample)

    Args:
        df: Full price DataFrame
        train_start_idx: Start index of training window
        train_end_idx: End index of training window (exclusive)
        test_start_idx: Start index of test window
        test_end_idx: End index of test window (exclusive)
        ticker: Stock ticker
        horizon: Holding period
        progress_callback: Optional callback for indicator progress
        strategy_class: Strategy class for optimization
        sector_df: Optional sector ETF DataFrame for relative strength calculations

    Returns:
        WindowResult with out-of-sample trades for persistence
    """
    # Extract train dataframe
    train_df = df.iloc[train_start_idx:train_end_idx].reset_index(drop=True)

    # For test evaluation, we need indicator warmup data.
    # Include the training window as warmup, then the test period.
    # This ensures indicators like SMA200, 52-week range, etc. are properly calculated.
    # We'll then filter trades to only those in the actual test period.
    INDICATOR_WARMUP_DAYS = 252  # ~1 year for 52-week indicators
    warmup_start_idx = max(0, test_start_idx - INDICATOR_WARMUP_DAYS)
    test_with_warmup_df = df.iloc[warmup_start_idx:test_end_idx].reset_index(drop=True)

    # The actual test period starts after the warmup
    test_period_start_in_warmup = test_start_idx - warmup_start_idx

    # Get date strings for window boundaries
    train_start = str(df.iloc[train_start_idx]['date'])
    train_end = str(df.iloc[train_end_idx - 1]['date'])
    test_start = str(df.iloc[test_start_idx]['date'])
    test_end = str(df.iloc[min(test_end_idx - 1, len(df) - 1)]['date'])

    # Run optimization on training data
    optimized_weights, per_indicator = two_pass_coordinate_descent(
        train_df,
        horizon=horizon,
        progress_callback=progress_callback,
        strategy_class=strategy_class,
        sector_df=sector_df
    )

    # Calculate training SQN
    train_result = fast_simulate(train_df, optimized_weights, horizon=horizon, sector_df=sector_df)

    # Evaluate on TEST data with warmup for indicator calculation
    test_result = fast_simulate(test_with_warmup_df, optimized_weights, horizon=horizon, sector_df=sector_df)

    # Filter trades to only those that occurred in the actual test period
    # (entry_idx >= test_period_start_in_warmup)
    test_trades_df = None
    test_trades_count = 0
    test_expectancy = 0.0

    if test_result.trades is not None and not test_result.trades.empty:
        # Filter to trades that entered during the test period
        test_period_trades = test_result.trades[
            test_result.trades['entry_idx'] >= test_period_start_in_warmup
        ].copy()

        if not test_period_trades.empty:
            test_trades_count = len(test_period_trades)
            test_expectancy = test_period_trades['pnl_pct'].mean() if 'pnl_pct' in test_period_trades.columns else 0.0

            # Adjust entry_idx to be relative to original df for enrichment
            # The entry_idx in test_period_trades is relative to test_with_warmup_df
            # We need to map it back: original_idx = warmup_start_idx + entry_idx
            test_period_trades['entry_idx'] = test_period_trades['entry_idx'] - test_period_start_in_warmup
            test_trades_df = test_period_trades

    # Enrich trades with exit_date and exit_price
    trades_df = enrich_trades_with_exits(
        test_trades_df,
        df,
        test_start_idx,
        horizon
    )

    return WindowResult(
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        window_days=train_end_idx - train_start_idx,
        weights=optimized_weights,
        train_sqn=train_result.sqn if not pd.isna(train_result.sqn) else 0.0,
        test_sqn=test_result.sqn if not pd.isna(test_result.sqn) else 0.0,
        trades_count=test_trades_count,
        expectancy=test_expectancy,
        test_gross_sqn=test_result.gross_sqn if not pd.isna(test_result.gross_sqn) else 0.0,
        trades=trades_df
    )


# ============================================================================
# Main Calibration Engine
# ============================================================================

# Default strategy classes to calibrate
DEFAULT_STRATEGY_CLASSES = ['directional', 'premium_sell', 'premium_buy']


async def calibrate_ticker(
    db: AsyncSession,
    ticker: str,
    horizons: List[int] = HORIZONS,
    strategy_classes: List[str] = DEFAULT_STRATEGY_CLASSES,
    progress_callback: Optional[Callable[[CalibrationProgress], None]] = None,
    use_rolling: bool = True,
    optimizer: OptimizerType = DEFAULT_OPTIMIZER
) -> Dict[str, Dict[int, FullOptimizationResult]]:
    """
    Run full calibration for a ticker across all horizons and strategy classes.

    When use_rolling=True (default), implements true Walk-Forward Optimization:
    - Generate rolling train/test windows
    - Optimize on each training window
    - Evaluate on each test window (out-of-sample)
    - Aggregate weights using exponential decay
    - Save all windows and trades to database

    Strategy classes:
    - 'directional': Optimized for price direction (buyShares, sellShares)
    - 'premium_sell': Optimized for premium selling (openCSP, openCC)
    - 'premium_buy': Optimized for breakout trades (buyCall, buyPut)

    Args:
        db: Database session
        ticker: Stock ticker
        horizons: List of holding periods to calibrate
        strategy_classes: List of strategy classes to calibrate
        progress_callback: Optional callback for progress updates
        use_rolling: If True, use rolling WFO; if False, use single-window (legacy)
        optimizer: Optimization algorithm to use (default: COORDINATE_DESCENT)

    Returns:
        Dict of {strategy_class: {horizon: FullOptimizationResult}}
    """
    import numpy as np
    results = {}

    # 1. Load price history
    if progress_callback:
        progress_callback(CalibrationProgress(
            ticker=ticker, horizon=0, stage='loading',
            progress=0, message='Loading price history...'
        ))

    try:
        df = await load_price_history(db, ticker)
    except ValueError as e:
        if progress_callback:
            progress_callback(CalibrationProgress(
                ticker=ticker, horizon=0, stage='error',
                progress=0, message=str(e)
            ))
        raise

    # 1b. Load sector ETF data for relative strength calculations
    # This will auto-fetch the sector ETF if not in database
    sector_df = None
    try:
        if progress_callback:
            progress_callback(CalibrationProgress(
                ticker=ticker, horizon=0, stage='loading',
                progress=5, message='Loading sector ETF data for relative strength...'
            ))
        sector_df = await ensure_sector_etf_data(ticker, db)
        if sector_df is not None and len(sector_df) > 0:
            logger.info(f"[WFO] Loaded {len(sector_df)} days of sector ETF data for {ticker}")
        else:
            logger.warning(f"[WFO] No sector ETF data available for {ticker} - RS indicators will be disabled")
            if progress_callback:
                progress_callback(CalibrationProgress(
                    ticker=ticker, horizon=0, stage='loading',
                    progress=5, message='Warning: No sector ETF data - relative strength indicators disabled'
                ))
    except Exception as e:
        logger.warning(f"[WFO] Failed to load sector ETF data for {ticker}: {e}")
        if progress_callback:
            progress_callback(CalibrationProgress(
                ticker=ticker, horizon=0, stage='loading',
                progress=5, message=f'Warning: Sector ETF load failed - RS indicators disabled: {e}'
            ))
        # Continue without RS data - other indicators will still work

    # 2. Calibrate for each strategy class and horizon
    total_iterations = len(strategy_classes) * len(horizons)
    iteration = 0

    for s_idx, strategy_class in enumerate(strategy_classes):
        results[strategy_class] = {}

        for i, horizon in enumerate(horizons):
            iteration += 1
            base_progress = ((iteration - 1) / total_iterations) * 100
            iter_weight = 1 / total_iterations

            if use_rolling:
                # ============================================================
                # ROLLING WALK-FORWARD OPTIMIZATION
                # ============================================================
                windows = generate_rolling_windows(df)

                if not windows:
                    if progress_callback:
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='error',
                            progress=base_progress,
                            message=f'[{strategy_class}] Insufficient data for rolling windows'
                        ))
                    results[strategy_class][horizon] = CalibrationErrorResult(
                        ticker=ticker, horizon=horizon,
                        error_type='insufficient_data',
                        message='Not enough data for rolling WFO (need 7+ months)',
                        trades_found=0, window_days=len(df)
                    )
                    continue

                if progress_callback:
                    progress_callback(CalibrationProgress(
                        ticker=ticker, horizon=horizon, stage='optimizing',
                        progress=base_progress,
                        message=f'[{strategy_class}] Rolling WFO: {len(windows)} windows for {horizon}d horizon...'
                    ))

                # Process each rolling window
                window_results: List[WindowResult] = []
                total_windows = len(windows)
                failed_windows = 0

                for w_idx, window in enumerate(windows):
                    window_progress = base_progress + (w_idx / total_windows) * iter_weight * 100

                    if progress_callback:
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='optimizing',
                            progress=window_progress,
                            message=f'[{strategy_class}] Window {w_idx + 1}/{total_windows}: {window["train_start"]} to {window["test_end"]}'
                        ))

                    # Create a closure to capture variables correctly
                    def make_indicator_callback(w_idx_local, base_prog, iter_wt, strat_cls):
                        def indicator_callback(indicator: str, pct: float):
                            if progress_callback:
                                combined = base_prog + ((w_idx_local + pct / 100) / total_windows) * iter_wt * 100
                                progress_callback(CalibrationProgress(
                                    ticker=ticker, horizon=horizon, stage='optimizing',
                                    progress=combined, current_indicator=indicator,
                                    message=f'[{strat_cls}] Window {w_idx_local + 1}: {indicator}...'
                                ))
                        return indicator_callback

                    try:
                        # Run CPU-bound optimization in a thread to keep event loop responsive
                        wr = await asyncio.to_thread(
                            optimize_single_window,
                            df=df,
                            train_start_idx=window['train_start_idx'],
                            train_end_idx=window['train_end_idx'],
                            test_start_idx=window['test_start_idx'],
                            test_end_idx=window['test_end_idx'],
                            ticker=ticker,
                            horizon=horizon,
                            progress_callback=make_indicator_callback(w_idx, base_progress, iter_weight, strategy_class),
                            strategy_class=strategy_class,
                            sector_df=sector_df
                        )
                        window_results.append(wr)
                    except Exception as e:
                        # Skip this window if it fails (e.g., insufficient trades in window)
                        logger.warning(f"[WFO] Window {w_idx + 1} failed for {ticker} horizon={horizon} strategy={strategy_class}: {e}")
                        failed_windows += 1
                        continue

                if not window_results:
                    results[strategy_class][horizon] = CalibrationErrorResult(
                        ticker=ticker, horizon=horizon,
                        error_type='insufficient_trades',
                        message=f'No valid rolling windows produced (all {total_windows} windows failed)',
                        trades_found=0, window_days=len(df)
                    )
                    continue

                # Aggregate weights from all windows using exponential decay
                aggregated_weights = aggregate_weights_exponential_decay(window_results)

                # Calculate aggregate metrics
                total_trades = sum(wr.trades_count for wr in window_results)
                avg_train_sqn_val = np.mean([wr.train_sqn for wr in window_results if not np.isnan(wr.train_sqn)])
                avg_test_sqn_val = np.mean([wr.test_sqn for wr in window_results if not np.isnan(wr.test_sqn)])
                avg_gross_sqn_val = np.mean([wr.test_gross_sqn for wr in window_results if not np.isnan(wr.test_gross_sqn)])

                # Overfit detection: flag if out-of-sample performance is less than 50% of in-sample
                # This indicates the model may be overfitting to training data
                overfit_warning = False
                if not np.isnan(avg_train_sqn_val) and not np.isnan(avg_test_sqn_val) and avg_train_sqn_val > 0:
                    overfit_warning = avg_test_sqn_val < (0.5 * avg_train_sqn_val)
                    if overfit_warning:
                        logger.warning(
                            f"[WFO] Overfit warning for {ticker} horizon={horizon}: "
                            f"avg_test_sqn={avg_test_sqn_val:.2f} < 0.5 * avg_train_sqn={avg_train_sqn_val:.2f}"
                        )

                # Build per-indicator results from aggregated weights
                # Use avg_test_sqn as the per-indicator score since rolling windows
                # optimize all indicators together (not per-indicator SQN available)
                per_indicator_results = {}
                for indicator, weight in aggregated_weights.items():
                    per_indicator_results[indicator] = OptimizationResult(
                        indicator=indicator,
                        optimal_weight=weight,
                        sqn_score=float(avg_test_sqn_val) if not np.isnan(avg_test_sqn_val) else 0.0,
                        stability_passed=True,  # Stability is validated per-window
                        coarse_results={weight: avg_test_sqn_val if not np.isnan(avg_test_sqn_val) else 0.0},
                        fine_results={weight: avg_test_sqn_val if not np.isnan(avg_test_sqn_val) else 0.0}
                    )

                # Build FullOptimizationResult with rolling window results attached
                result = FullOptimizationResult(
                    ticker=ticker,
                    horizon=horizon,
                    weights=aggregated_weights,
                    train_sqn=float(avg_test_sqn_val) if not np.isnan(avg_test_sqn_val) else 0.0,
                    total_trades=total_trades,
                    per_indicator=per_indicator_results,
                    optimized_at=datetime.utcnow(),
                    reduced_confidence=total_trades < 30,
                    window_results=window_results,  # Attach for database persistence
                    overfit_warning=overfit_warning,
                    avg_train_sqn=float(avg_train_sqn_val) if not np.isnan(avg_train_sqn_val) else None,
                    avg_test_sqn=float(avg_test_sqn_val) if not np.isnan(avg_test_sqn_val) else None,
                    avg_gross_sqn=float(avg_gross_sqn_val) if not np.isnan(avg_gross_sqn_val) else None
                )
                results[strategy_class][horizon] = result

                if progress_callback:
                    overfit_msg = ' [OVERFIT WARNING]' if overfit_warning else ''
                    gross_sqn_str = f'{avg_gross_sqn_val:.2f}' if not np.isnan(avg_gross_sqn_val) else 'N/A'
                    net_sqn_str = f'{avg_test_sqn_val:.2f}' if not np.isnan(avg_test_sqn_val) else 'N/A'
                    window_msg = f'{len(window_results)}/{total_windows} windows'
                    if failed_windows > 0:
                        window_msg += f' ({failed_windows} failed)'
                    progress_callback(CalibrationProgress(
                        ticker=ticker, horizon=horizon, stage='complete',
                        progress=(iteration / total_iterations) * 100,
                        weights=result.weights,
                        train_sqn=result.train_sqn,
                        test_sqn=result.avg_test_sqn,
                        gross_sqn=result.avg_gross_sqn,
                        message=f'[{strategy_class}] Horizon {horizon}d: {window_msg} | Gross SQN={gross_sqn_str} | Net SQN={net_sqn_str}{overfit_msg}'
                    ))

            else:
                # ============================================================
                # SINGLE-WINDOW OPTIMIZATION (Legacy mode)
                # ============================================================
                if progress_callback:
                    progress_callback(CalibrationProgress(
                        ticker=ticker, horizon=horizon, stage='optimizing',
                        progress=base_progress,
                        message=f'[{strategy_class}] Optimizing {horizon}-day horizon...'
                    ))

                def indicator_callback(indicator: str, pct: float):
                    if progress_callback:
                        combined_progress = base_progress + (pct / 100) * iter_weight * 100
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='optimizing',
                            progress=combined_progress,
                            current_indicator=indicator,
                            message=f'[{strategy_class}] Optimizing {indicator}...'
                        ))

                try:
                    # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
                    # Note: progress_callback can't cross process boundaries, so we
                    # only get high-level progress updates (before/after optimization)
                    loop = asyncio.get_running_loop()
                    pool = get_process_pool()

                    # Create args for process pool (can't use kwargs directly)
                    optimize_func = partial(
                        optimize_for_ticker,
                        df, ticker, horizon,
                        progress_callback=None,  # Can't pickle callbacks across processes
                        optimizer=optimizer,
                        strategy_class=strategy_class,
                        sector_df=sector_df
                    )

                    result = await loop.run_in_executor(pool, optimize_func)
                    results[strategy_class][horizon] = result

                    if progress_callback:
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='complete',
                            progress=(iteration / total_iterations) * 100,
                            weights=result.weights,
                            train_sqn=result.train_sqn,
                            message=f'[{strategy_class}] Horizon {horizon} complete: SQN={result.train_sqn:.2f}'
                        ))

                except InsufficientVolatilityError as e:
                    if progress_callback:
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='error',
                            progress=(iteration / total_iterations) * 100,
                            message=str(e)
                        ))
                    results[strategy_class][horizon] = CalibrationErrorResult(
                        ticker=ticker,
                        horizon=horizon,
                        error_type='insufficient_trades',
                        message=str(e),
                        trades_found=getattr(e, 'trades', 0),
                        window_days=getattr(e, 'window_days', 0)
                    )
                except Exception as e:
                    # Catch any other errors (including subprocess errors)
                    logger.error(f"[WFO] Optimization failed for {ticker} {strategy_class} h={horizon}: {e}")
                    if progress_callback:
                        progress_callback(CalibrationProgress(
                            ticker=ticker, horizon=horizon, stage='error',
                            progress=(iteration / total_iterations) * 100,
                            message=f'Optimization error: {str(e)}'
                        ))
                    results[strategy_class][horizon] = CalibrationErrorResult(
                        ticker=ticker,
                        horizon=horizon,
                        error_type='optimization_error',
                        message=str(e),
                        trades_found=0,
                        window_days=0
                    )

    return results


async def calibrate_ticker_streaming(
    db: AsyncSession,
    ticker: str,
    horizons: Optional[List[int]] = None,
    strategy_classes: List[str] = DEFAULT_STRATEGY_CLASSES
) -> AsyncGenerator[str, None]:
    """
    Calibrate ticker with SSE streaming progress.

    Args:
        db: Database session
        ticker: Stock ticker
        horizons: List of horizons to calibrate. If None, uses HORIZONS default.
        strategy_classes: Strategy classes to calibrate

    Yields SSE-formatted progress updates.
    """
    # Use default horizons if none provided
    actual_horizons = horizons if horizons else HORIZONS
    logger.info(f"[SSE] calibrate_ticker_streaming for {ticker} with horizons: {actual_horizons}")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_callback(update: CalibrationProgress):
        # Ensure thread-safety for queue operations when called from worker threads
        loop.call_soon_threadsafe(queue.put_nowait, update.to_sse())

    # Start calibration in background
    async def run_calibration():
        try:
            await calibrate_ticker(
                db=db,
                ticker=ticker,
                horizons=actual_horizons,
                strategy_classes=strategy_classes,
                progress_callback=progress_callback
            )
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                CalibrationProgress(
                    ticker=ticker, horizon=0, stage='error',
                    progress=0, message=str(e)
                ).to_sse()
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # Signal completion

    task = asyncio.create_task(run_calibration())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        task.cancel()


# ============================================================================
# Result Storage
# ============================================================================

async def save_calibration_result(
    db: AsyncSession,
    ticker: str,
    result: FullOptimizationResult,
    window_results: Optional[List[WindowResult]] = None,
    strategy_class: str = 'all'
) -> None:
    """
    Save calibration results to database.

    Args:
        db: Database session
        ticker: Stock ticker
        result: Optimization result
        window_results: Optional rolling window results
        strategy_class: Strategy class ('all', 'directional', 'premium_sell', 'premium_buy')
    """
    # 1. Save/update weights using atomic upsert (thread-safe for parallel calibration)
    for indicator, weight in result.weights.items():
        per_ind = result.per_indicator.get(indicator)

        # Build the row data
        row_data = {
            'ticker': ticker,
            'indicator': indicator,
            'action': 'all',
            'horizon': result.horizon,
            'strategy_class': strategy_class,
            'weight': weight,
            'sqn_score': per_ind.sqn_score if per_ind else None,
            'stability_passed': per_ind.stability_passed if per_ind else True,
            'overfit_warning': result.overfit_warning,
            'avg_train_sqn': result.avg_train_sqn,
            'avg_test_sqn': result.avg_test_sqn,
            'avg_gross_sqn': result.avg_gross_sqn,
            'updated_at': datetime.utcnow()
        }

        # Atomic upsert: INSERT or UPDATE on conflict
        # This is thread-safe and eliminates race conditions
        stmt = sqlite_insert(CalibrationWeights).values(**row_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['ticker', 'indicator', 'action', 'horizon', 'strategy_class'],
            set_={
                'weight': stmt.excluded.weight,
                'sqn_score': stmt.excluded.sqn_score,
                'stability_passed': stmt.excluded.stability_passed,
                'overfit_warning': stmt.excluded.overfit_warning,
                'avg_train_sqn': stmt.excluded.avg_train_sqn,
                'avg_test_sqn': stmt.excluded.avg_test_sqn,
                'avg_gross_sqn': stmt.excluded.avg_gross_sqn,
                'updated_at': stmt.excluded.updated_at
            }
        )
        await db.execute(stmt)
    
    # 2. Save window results and trades if provided (batched for performance)
    if window_results:
        # First pass: create all windows and collect trades
        windows_to_flush = []
        trades_by_window_idx = {}

        for idx, wr in enumerate(window_results):
            window = CalibrationWindow(
                ticker=ticker,
                horizon=result.horizon,
                train_start=wr.train_start,
                train_end=wr.train_end,
                test_start=wr.test_start,
                test_end=wr.test_end,
                window_days=wr.window_days,
                weights_json=json.dumps(wr.weights),
                train_sqn=wr.train_sqn if not pd.isna(wr.train_sqn) else 0.0,
                test_sqn=wr.test_sqn if not pd.isna(wr.test_sqn) else 0.0,
                expectancy=wr.expectancy if not pd.isna(wr.expectancy) else 0.0,
                trades_count=wr.trades_count
            )
            db.add(window)
            windows_to_flush.append(window)

            # Collect trades for later
            if wr.trades is not None and not wr.trades.empty:
                trades_by_window_idx[idx] = wr.trades

        # Single flush to get all window IDs
        await db.flush()

        # Second pass: create trades with proper window_ids
        for idx, trades_df in trades_by_window_idx.items():
            window_id = windows_to_flush[idx].id
            for _, trade in trades_df.iterrows():
                cal_trade = CalibrationTrade(
                    window_id=window_id,
                    ticker=ticker,
                    signal_date=str(trade.get('signal_date', '')) if pd.notna(trade.get('signal_date')) else None,
                    entry_date=str(trade.get('entry_date', '')),
                    exit_date=str(trade.get('exit_date', '')),
                    horizon=result.horizon,
                    direction=str(trade.get('direction', 'long')),
                    entry_price=float(trade.get('entry_price', 0)),
                    exit_price=float(trade.get('exit_price', 0)),
                    pnl_pct=float(trade.get('pnl_pct', 0)),
                    transaction_cost=float(trade.get('costs', 0.001)),
                    entry_gap_pct=float(trade.get('entry_gap_pct', 0)) if pd.notna(trade.get('entry_gap_pct')) else None,
                    slippage_applied=float(trade.get('slippage_applied', 0)) if pd.notna(trade.get('slippage_applied')) else None,
                    market_regime=str(trade.get('market_regime', '')) if pd.notna(trade.get('market_regime')) else None
                )
                db.add(cal_trade)

    await db.commit()


async def load_calibrated_weights(
    db: AsyncSession,
    ticker: str,
    horizon: int,
    strategy_class: str = 'all'
) -> Optional[Dict[str, float]]:
    """
    Load calibrated weights from database.

    Args:
        db: Database session
        ticker: Stock ticker
        horizon: Trading horizon (3 or 15)
        strategy_class: Strategy class to load weights for

    Returns None if no calibration exists (will use defaults).
    """
    result = await db.execute(
        select(CalibrationWeights).where(
            CalibrationWeights.ticker == ticker,
            CalibrationWeights.horizon == horizon,
            CalibrationWeights.strategy_class == strategy_class
        )
    )

    rows = result.scalars().all()

    if not rows:
        # Fall back to 'all' strategy if specific strategy not found
        if strategy_class != 'all':
            return await load_calibrated_weights(db, ticker, horizon, 'all')
        return None

    return {row.indicator: row.weight for row in rows}


# ============================================================================
# Batch Calibration
# ============================================================================

async def calibrate_portfolio(
    db: AsyncSession,
    tickers: List[str],
    horizons: List[int] = HORIZONS,
    strategy_classes: List[str] = DEFAULT_STRATEGY_CLASSES,
    progress_callback: Optional[Callable[[str, CalibrationProgress], None]] = None
) -> Dict[str, Dict[str, Dict[int, FullOptimizationResult]]]:
    """
    Calibrate all tickers in a portfolio.

    Args:
        db: Database session
        tickers: List of stock tickers
        horizons: Horizons to calibrate
        strategy_classes: Strategy classes to calibrate
        progress_callback: Optional callback(ticker, progress)

    Returns:
        Dict of {ticker: {strategy_class: {horizon: result}}}
    """
    results = {}

    for i, ticker in enumerate(tickers):
        def ticker_callback(update: CalibrationProgress):
            if progress_callback:
                progress_callback(ticker, update)

        try:
            results[ticker] = await calibrate_ticker(
                db=db,
                ticker=ticker,
                horizons=horizons,
                strategy_classes=strategy_classes,
                progress_callback=ticker_callback
            )
        except Exception as e:
            results[ticker] = {'error': str(e)}

    return results

