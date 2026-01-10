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

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

logger = logging.getLogger(__name__)

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
    FullOptimizationResult
)


# ============================================================================
# Configuration
# ============================================================================

# Rolling window configuration
TRAIN_WINDOW_MONTHS = 6       # Initial training window (4:1 ratio)
TEST_WINDOW_MONTHS = 1        # Out-of-sample test period
ROLL_STEP_MONTHS = 1          # Slide forward by 1 month

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
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


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
    
    logger.info(f"[WFO] Price history for {ticker}: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
    
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
# Main Calibration Engine
# ============================================================================

async def calibrate_ticker(
    db: AsyncSession,
    ticker: str,
    horizons: List[int] = HORIZONS,
    progress_callback: Optional[Callable[[CalibrationProgress], None]] = None
) -> Dict[int, FullOptimizationResult]:
    """
    Run full calibration for a ticker across all horizons.
    
    Args:
        db: Database session
        ticker: Stock ticker
        horizons: List of holding periods to calibrate
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict of {horizon: FullOptimizationResult}
    """
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
    
    # 2. Calibrate for each horizon
    for i, horizon in enumerate(horizons):
        horizon_progress = (i / len(horizons)) * 100
        
        if progress_callback:
            progress_callback(CalibrationProgress(
                ticker=ticker, horizon=horizon, stage='optimizing',
                progress=horizon_progress,
                message=f'Optimizing {horizon}-day horizon...'
            ))
        
        def indicator_callback(indicator: str, pct: float):
            if progress_callback:
                combined_progress = horizon_progress + (pct / len(horizons))
                progress_callback(CalibrationProgress(
                    ticker=ticker, horizon=horizon, stage='optimizing',
                    progress=combined_progress,
                    current_indicator=indicator,
                    message=f'Optimizing {indicator}...'
                ))
        
        try:
            result = optimize_for_ticker(
                df, ticker, horizon,
                progress_callback=indicator_callback
            )
            results[horizon] = result
            
            if progress_callback:
                progress_callback(CalibrationProgress(
                    ticker=ticker, horizon=horizon, stage='complete',
                    progress=((i + 1) / len(horizons)) * 100,
                    weights=result.weights,
                    train_sqn=result.train_sqn,
                    message=f'Horizon {horizon} complete: SQN={result.train_sqn:.2f}'
                ))
                
        except InsufficientVolatilityError as e:
            if progress_callback:
                progress_callback(CalibrationProgress(
                    ticker=ticker, horizon=horizon, stage='error',
                    progress=((i + 1) / len(horizons)) * 100,
                    message=str(e)
                ))
            results[horizon] = None
    
    return results


async def calibrate_ticker_streaming(
    db: AsyncSession,
    ticker: str,
    horizons: List[int] = HORIZONS
) -> AsyncGenerator[str, None]:
    """
    Calibrate ticker with SSE streaming progress.
    
    Yields SSE-formatted progress updates.
    """
    queue: asyncio.Queue = asyncio.Queue()
    
    def progress_callback(update: CalibrationProgress):
        queue.put_nowait(update.to_sse())
    
    # Start calibration in background
    async def run_calibration():
        try:
            await calibrate_ticker(db, ticker, horizons, progress_callback)
        except Exception as e:
            queue.put_nowait(CalibrationProgress(
                ticker=ticker, horizon=0, stage='error',
                progress=0, message=str(e)
            ).to_sse())
        finally:
            queue.put_nowait(None)  # Signal completion
    
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
    # 1. Save/update weights
    for indicator, weight in result.weights.items():
        per_ind = result.per_indicator.get(indicator)

        # Upsert weight (including strategy_class in query)
        existing = await db.execute(
            select(CalibrationWeights).where(
                CalibrationWeights.ticker == ticker,
                CalibrationWeights.indicator == indicator,
                CalibrationWeights.horizon == result.horizon,
                CalibrationWeights.action == 'all',
                CalibrationWeights.strategy_class == strategy_class
            )
        )
        row = existing.scalar_one_or_none()

        if row:
            row.weight = weight
            row.sqn_score = per_ind.sqn_score if per_ind else None
            row.stability_passed = per_ind.stability_passed if per_ind else True
            row.updated_at = datetime.utcnow()
        else:
            db.add(CalibrationWeights(
                ticker=ticker,
                indicator=indicator,
                action='all',
                horizon=result.horizon,
                strategy_class=strategy_class,
                weight=weight,
                sqn_score=per_ind.sqn_score if per_ind else None,
                stability_passed=per_ind.stability_passed if per_ind else True
            ))
    
    # 2. Save window results if provided
    if window_results:
        for wr in window_results:
            window = CalibrationWindow(
                ticker=ticker,
                horizon=result.horizon,
                train_start=wr.train_start,
                train_end=wr.train_end,
                test_start=wr.test_start,
                test_end=wr.test_end,
                window_days=wr.window_days,
                weights_json=json.dumps(wr.weights),
                train_sqn=wr.train_sqn,
                test_sqn=wr.test_sqn,
                expectancy=wr.expectancy,
                trades_count=wr.trades_count
            )
            db.add(window)
    
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
    progress_callback: Optional[Callable[[str, CalibrationProgress], None]] = None
) -> Dict[str, Dict[int, FullOptimizationResult]]:
    """
    Calibrate all tickers in a portfolio.
    
    Args:
        db: Database session
        tickers: List of stock tickers
        horizons: Horizons to calibrate
        progress_callback: Optional callback(ticker, progress)
        
    Returns:
        Dict of {ticker: {horizon: result}}
    """
    results = {}
    
    for i, ticker in enumerate(tickers):
        def ticker_callback(update: CalibrationProgress):
            if progress_callback:
                progress_callback(ticker, update)
        
        try:
            results[ticker] = await calibrate_ticker(
                db, ticker, horizons, ticker_callback
            )
        except Exception as e:
            results[ticker] = {'error': str(e)}
    
    return results

