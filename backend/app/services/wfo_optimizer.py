"""
Walk-Forward Optimization Optimizer

Two-pass Coordinate Descent optimizer with stability validation.
Optimizes indicator weights while avoiding curve fitting.

CONSTRAINTS (Pre-Flight Checklist):
1. Only optimize WEIGHTS (0.0-2.5), NOT indicator periods
2. Transaction cost >= 0.1% (realistic friction)
3. Force cash in un-tradeable regimes (BEAR_VOLATILE)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

from .wfo_simulator import (
    fast_simulate,
    SimulationResult,
    simulate_weight_grid,
    MIN_TRANSACTION_COST,
    MIN_TRADES_FOR_SIGNIFICANCE
)
from .regime import MarketRegime, get_regime_weight_adjustments
from .strategy_models import (
    get_strategy_model,
    BaseStrategyModel,
    STRATEGY_MODELS,
)


# ============================================================================
# Configuration
# ============================================================================

# Coarse grid for first pass
COARSE_GRID = [0.0, 1.0, 2.0]

# Fine grid offsets for second pass (relative to coarse winner)
FINE_OFFSETS = [-0.4, -0.2, 0.2, 0.4]  # Skip 0.0 (center already tested)

# Stability validation: neighbor must be >= this fraction of peak
# Higher threshold = more strict validation, reduces overfitting risk
STABILITY_THRESHOLD = 0.8

# Multiple Testing Correction Parameters
# With 9 indicators × 3 coarse values = 27 tests, we need to guard against false positives

# Minimum improvement over default to accept a non-default weight
# This acts like a significance threshold - small improvements are likely noise
MIN_IMPROVEMENT_THRESHOLD = 0.10  # 10% improvement required

# Shrinkage factor: how much to pull extreme weights toward default (1.0)
# Higher = more conservative, lower = trust optimization more
# Formula: final_weight = shrinkage * default + (1 - shrinkage) * optimal
SHRINKAGE_FACTOR = 0.3  # 30% shrinkage toward default

# Bonferroni-style penalty: reduce confidence when testing many hypotheses
# penalty = score * (1 - penalty_per_test * num_tests)
PENALTY_PER_TEST = 0.005  # 0.5% penalty per test

# Default weights for all indicators
DEFAULT_WEIGHTS = {
    'rsi': 1.0,
    'macd': 1.0,
    'bollinger': 1.0,
    'adx': 1.0,
    'cmf': 1.0,
    'momentum': 1.0,
    'volume': 1.0,
    'rvol': 1.0,
    'sma': 1.0,
    'position': 1.0,
    'squeeze': 1.0,
}

# Indicators to optimize
INDICATORS_TO_OPTIMIZE = [
    'rsi', 'macd', 'bollinger', 'adx', 'cmf', 
    'momentum', 'volume', 'sma', 'position'
]


# ============================================================================
# Custom Errors
# ============================================================================

class CalibrationError(Exception):
    """Base class for calibration errors."""
    pass


class InsufficientVolatilityError(CalibrationError):
    """Raised when stock doesn't generate enough trades even with max window."""
    
    def __init__(self, ticker: str, trades: int, window_days: int):
        self.ticker = ticker
        self.trades = trades
        self.window_days = window_days
        super().__init__(
            f"{ticker}: Only {trades} trades in {window_days} days "
            f"(need {MIN_TRADES_FOR_SIGNIFICANCE}). Strategy may not apply."
        )


class UnstableWeightError(CalibrationError):
    """Raised when optimal weight fails stability validation."""
    
    def __init__(self, indicator: str, weight: float, neighbor_scores: List[float]):
        self.indicator = indicator
        self.weight = weight
        self.neighbor_scores = neighbor_scores
        super().__init__(
            f"{indicator}={weight} is unstable. Neighbors: {neighbor_scores}"
        )


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OptimizationResult:
    """Result of optimizing a single indicator."""
    indicator: str
    optimal_weight: float
    sqn_score: float
    stability_passed: bool
    coarse_results: Dict[float, float]  # weight -> SQN
    fine_results: Dict[float, float]    # weight -> SQN


@dataclass
class FullOptimizationResult:
    """Result of optimizing all indicators."""
    ticker: str
    horizon: int
    weights: Dict[str, float]
    train_sqn: float
    total_trades: int
    per_indicator: Dict[str, OptimizationResult]
    optimized_at: datetime


# ============================================================================
# Two-Pass Coordinate Descent Optimizer
# ============================================================================

def two_pass_coordinate_descent(
    df: pd.DataFrame,
    horizon: int = 3,
    initial_weights: Optional[Dict[str, float]] = None,
    indicators: Optional[List[str]] = None,
    coarse_grid: List[float] = COARSE_GRID,
    fine_offsets: List[float] = FINE_OFFSETS,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    strategy_class: str = 'all'
) -> Tuple[Dict[str, float], Dict[str, OptimizationResult]]:
    """
    Two-pass Coordinate Descent optimizer.

    Pass 1 (Coarse): Test [0.0, 1.0, 2.0] for each indicator
    Pass 2 (Fine): Test [±0.2, ±0.4] around coarse winner

    Args:
        df: Price data DataFrame
        horizon: Holding period in days
        initial_weights: Starting weights (defaults to 1.0 for all)
        indicators: Which indicators to optimize
        coarse_grid: Values to test in coarse pass
        fine_offsets: Offsets to test in fine pass (relative to winner)
        progress_callback: Optional callback(indicator, progress) for updates
        strategy_class: Strategy class for objective function ('all', 'directional',
                       'premium_sell', 'premium_buy')

    Returns:
        Tuple of (optimized_weights, per_indicator_results)
    """
    # Get strategy model for objective function
    strategy_model = get_strategy_model(strategy_class)

    # Use strategy-specific defaults if no initial weights provided
    weights = (initial_weights or strategy_model.get_default_weights()).copy()

    # Use strategy-specific indicators if not specified
    indicators = indicators or strategy_model.get_indicators_to_optimize()
    
    results: Dict[str, OptimizationResult] = {}
    total_indicators = len(indicators)

    def get_objective_score(sim_result: SimulationResult) -> float:
        """Get objective score from strategy model."""
        objective = strategy_model.objective(sim_result)
        return objective.score if objective.is_valid else -np.inf

    # Calculate total tests for multiple testing correction
    num_tests = calculate_total_tests(coarse_grid, fine_offsets)

    for i, indicator in enumerate(indicators):
        if progress_callback:
            progress = (i / total_indicators) * 100
            progress_callback(indicator, progress)

        # Get default weight for this indicator
        default_weight = strategy_model.get_default_weights().get(indicator, 1.0)

        # Pass 1: Coarse grid search
        coarse_results = {}
        best_coarse_weight = weights.get(indicator, default_weight)
        best_coarse_score = -np.inf
        default_score = -np.inf  # Score at default weight

        for test_weight in coarse_grid:
            test_weights = weights.copy()
            test_weights[indicator] = test_weight

            result = fast_simulate(df, test_weights, horizon=horizon)
            score = get_objective_score(result)

            coarse_results[test_weight] = score

            # Track default score for multiple testing correction
            if abs(test_weight - default_weight) < 0.01:
                default_score = score

            if score > best_coarse_score:
                best_coarse_score = score
                best_coarse_weight = test_weight

        # If default wasn't in coarse grid, compute it
        if default_score == -np.inf:
            test_weights = weights.copy()
            test_weights[indicator] = default_weight
            result = fast_simulate(df, test_weights, horizon=horizon)
            default_score = get_objective_score(result)

        # Pass 2: Fine grid around winner (skip center - already tested)
        fine_results = {best_coarse_weight: best_coarse_score}  # Include coarse winner
        best_fine_weight = best_coarse_weight
        best_fine_score = best_coarse_score

        for offset in fine_offsets:
            test_weight = best_coarse_weight + offset

            # Clamp to valid range
            if test_weight < 0.0 or test_weight > 2.5:
                continue

            test_weights = weights.copy()
            test_weights[indicator] = test_weight

            result = fast_simulate(df, test_weights, horizon=horizon)
            score = get_objective_score(result)

            fine_results[test_weight] = score

            if score > best_fine_score:
                best_fine_score = score
                best_fine_weight = test_weight

        # Stability check
        stability_passed = validate_stability(
            indicator, best_fine_weight, fine_results
        )

        # Apply multiple testing correction
        corrected_weight, corrected_score, was_shrunk = apply_multiple_testing_correction(
            indicator=indicator,
            optimal_weight=best_fine_weight,
            optimal_score=best_fine_score,
            default_score=default_score,
            num_tests=num_tests,
            default_weight=default_weight
        )

        # If weight was shrunk, stability is questionable
        if was_shrunk:
            stability_passed = False

        # Lock in the corrected winner
        weights[indicator] = corrected_weight

        results[indicator] = OptimizationResult(
            indicator=indicator,
            optimal_weight=corrected_weight,
            sqn_score=corrected_score,  # Now stores corrected objective score
            stability_passed=stability_passed,
            coarse_results=coarse_results,
            fine_results=fine_results
        )
    
    if progress_callback:
        progress_callback("complete", 100.0)
    
    return weights, results


def validate_stability(
    indicator: str,
    best_weight: float,
    results: Dict[float, float],
    threshold: float = STABILITY_THRESHOLD
) -> bool:
    """
    Validate that optimal weight is in a stable region, not a lonely peak.
    
    A "stable" weight means neighboring weights also perform reasonably well.
    This prevents overfitting to noise.
    
    Args:
        indicator: Indicator name
        best_weight: The proposed optimal weight
        results: Dict of weight -> SQN from optimization
        threshold: Neighbors must achieve >= threshold * peak_sqn
        
    Returns:
        True if stable, False if the weight is a "lonely peak"
    """
    peak_sqn = results.get(best_weight, 0)
    
    if peak_sqn <= 0:
        return False
    
    # Find neighbors (within ±0.3 of best weight)
    neighbors = []
    for weight, sqn in results.items():
        if weight != best_weight and abs(weight - best_weight) <= 0.3:
            neighbors.append(sqn)
    
    if not neighbors:
        # No neighbors tested - can't validate stability
        return True
    
    # Check if neighbors are at least threshold% of peak
    for neighbor_sqn in neighbors:
        if neighbor_sqn < peak_sqn * threshold:
            # This neighbor is too weak - peak might be noise
            return False
    
    return True


# ============================================================================
# Multiple Testing Correction
# ============================================================================

def apply_multiple_testing_correction(
    indicator: str,
    optimal_weight: float,
    optimal_score: float,
    default_score: float,
    num_tests: int,
    default_weight: float = 1.0
) -> Tuple[float, float, bool]:
    """
    Apply multiple testing correction to reduce overfitting risk.

    When testing many weight combinations, we're likely to find some that
    perform well by chance. This function applies three corrections:

    1. Minimum improvement threshold: Only accept non-default weights if
       they show meaningful improvement (guards against noise)

    2. Shrinkage toward default: Pull extreme weights back toward the
       prior (default=1.0) based on uncertainty

    3. Bonferroni-style penalty: Reduce the effective score based on
       the number of tests performed

    Args:
        indicator: Name of indicator being optimized
        optimal_weight: The proposed optimal weight from optimization
        optimal_score: Score at optimal weight
        default_score: Score at default weight (1.0)
        num_tests: Total number of weight values tested
        default_weight: The default/prior weight (usually 1.0)

    Returns:
        Tuple of (corrected_weight, corrected_score, was_shrunk)
    """
    was_shrunk = False

    # Guard against invalid scores
    if optimal_score <= 0 or not np.isfinite(optimal_score):
        return default_weight, default_score, True

    if default_score <= 0 or not np.isfinite(default_score):
        default_score = 0.01  # Small positive value

    # 1. Check minimum improvement threshold
    improvement = (optimal_score - default_score) / max(abs(default_score), 0.01)

    if improvement < MIN_IMPROVEMENT_THRESHOLD:
        # Improvement too small - likely noise, revert to default
        logger.debug(
            f"[MTC] {indicator}: improvement {improvement:.1%} < threshold "
            f"{MIN_IMPROVEMENT_THRESHOLD:.1%}, reverting to default"
        )
        return default_weight, default_score, True

    # 2. Apply shrinkage toward default
    # The further from default, the more we shrink
    shrunk_weight = SHRINKAGE_FACTOR * default_weight + (1 - SHRINKAGE_FACTOR) * optimal_weight

    # Round to nearest 0.1 for cleaner weights
    shrunk_weight = round(shrunk_weight, 1)

    # Clamp to valid range
    shrunk_weight = max(0.0, min(2.5, shrunk_weight))

    if abs(shrunk_weight - optimal_weight) > 0.05:
        was_shrunk = True
        logger.debug(
            f"[MTC] {indicator}: shrunk {optimal_weight:.2f} -> {shrunk_weight:.2f}"
        )

    # 3. Apply Bonferroni-style penalty to score
    penalty = 1 - (PENALTY_PER_TEST * num_tests)
    penalty = max(0.5, penalty)  # Cap penalty at 50%
    corrected_score = optimal_score * penalty

    logger.debug(
        f"[MTC] {indicator}: score {optimal_score:.3f} -> {corrected_score:.3f} "
        f"(penalty {1-penalty:.1%} for {num_tests} tests)"
    )

    return shrunk_weight, corrected_score, was_shrunk


def calculate_total_tests(coarse_grid: List[float], fine_offsets: List[float]) -> int:
    """Calculate total number of weight values tested per indicator."""
    # Coarse pass tests all grid values
    coarse_tests = len(coarse_grid)
    # Fine pass tests offsets around winner (plus the winner itself is already counted)
    fine_tests = len(fine_offsets)
    return coarse_tests + fine_tests


# ============================================================================
# Regime-Aware Weight Adjustment
# ============================================================================

def apply_regime_filter(
    weights: Dict[str, float],
    regime: str
) -> Dict[str, float]:
    """
    Apply regime-specific weight adjustments.
    
    In un-tradeable regimes (BEAR_VOLATILE), zero out mean reversion
    indicators to force cash position.
    
    Args:
        weights: Base weights
        regime: Current market regime
        
    Returns:
        Adjusted weights
    """
    adjustments = get_regime_weight_adjustments(regime)
    
    if not adjustments:
        return weights
    
    adjusted = weights.copy()
    for indicator, multiplier in adjustments.items():
        if indicator in adjusted:
            adjusted[indicator] = adjusted[indicator] * multiplier
    
    return adjusted


# ============================================================================
# Adaptive Window Sizing
# ============================================================================

def get_adaptive_window(
    df: pd.DataFrame,
    weights: Dict[str, float],
    horizon: int,
    min_trades: Optional[int] = None,
    start_days: int = 126,  # 6 months
    max_days: int = 504,    # 2 years
    step_days: int = 21,    # 1 month
    strategy_class: str = 'all'
) -> Tuple[int, int]:
    """
    Adaptively expand training window until minimum trade count is reached.

    This solves the "Small Sample Size Trap" where a 6-month window
    might only generate 18 trades (need >= 30 for statistical significance).

    Args:
        df: Full price data
        weights: Current weight configuration
        horizon: Holding period
        min_trades: Minimum trades required (uses strategy default if None)
        start_days: Initial window size (default 6 months)
        max_days: Maximum window size (default 2 years)
        step_days: Window expansion step (default 1 month)
        strategy_class: Strategy class for min_trades default

    Returns:
        Tuple of (window_days, trade_count)

    Raises:
        InsufficientVolatilityError: If max window still has < min_trades
    """
    # Use strategy-specific min_trades if not specified
    if min_trades is None:
        strategy_model = get_strategy_model(strategy_class)
        min_trades = strategy_model.min_trades
    window_days = min(start_days, len(df))
    
    while window_days <= min(max_days, len(df)):
        # Use most recent window
        window_df = df.tail(window_days).reset_index(drop=True)
        
        result = fast_simulate(window_df, weights, horizon=horizon)
        
        if result.total_trades >= min_trades:
            return window_days, result.total_trades
        
        # Expand window
        window_days += step_days
    
    # Max window still insufficient
    final_window = min(max_days, len(df))
    window_df = df.tail(final_window).reset_index(drop=True)
    result = fast_simulate(window_df, weights, horizon=horizon)
    
    if result.total_trades < min_trades:
        raise InsufficientVolatilityError(
            ticker="Unknown",  # Will be set by caller
            trades=result.total_trades,
            window_days=final_window
        )
    
    return final_window, result.total_trades


# ============================================================================
# Full Optimization Pipeline
# ============================================================================

def optimize_for_ticker(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 3,
    initial_weights: Optional[Dict[str, float]] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    strategy_class: str = 'all'
) -> FullOptimizationResult:
    """
    Run full optimization pipeline for a single ticker.

    Args:
        df: Price data
        ticker: Stock ticker symbol
        horizon: Holding period (3 for swing, 15 for trend)
        initial_weights: Starting weights
        progress_callback: Optional progress updates
        strategy_class: Strategy class for optimization ('all', 'directional',
                       'premium_sell', 'premium_buy')

    Returns:
        FullOptimizationResult with optimized weights and metrics
    """
    logger.info(f"[WFO] optimize_for_ticker: {ticker}, horizon={horizon}, strategy={strategy_class}, data_rows={len(df)}")

    # Get strategy model for defaults
    strategy_model = get_strategy_model(strategy_class)
    weights = initial_weights or strategy_model.get_default_weights()
    logger.info(f"[WFO] Initial weights ({strategy_class}): {weights}")

    # 1. Get adaptive window size (uses strategy-specific min_trades)
    try:
        logger.info(f"[WFO] Getting adaptive window for {ticker}...")
        window_days, trade_count = get_adaptive_window(
            df, weights, horizon, strategy_class=strategy_class
        )
        logger.info(f"[WFO] Adaptive window: {window_days} days, {trade_count} trades")
    except InsufficientVolatilityError as e:
        logger.warning(f"[WFO] Insufficient volatility for {ticker}: {e}")
        e.ticker = ticker
        raise

    # 2. Use the adaptive window for optimization
    train_df = df.tail(window_days).reset_index(drop=True)
    logger.info(f"[WFO] Training data: {len(train_df)} rows")

    # 3. Run two-pass coordinate descent with strategy-specific objective
    logger.info(f"[WFO] Running two-pass coordinate descent for {ticker} ({strategy_class})...")
    optimized_weights, per_indicator = two_pass_coordinate_descent(
        train_df,
        horizon=horizon,
        initial_weights=weights,
        progress_callback=progress_callback,
        strategy_class=strategy_class
    )
    logger.info(f"[WFO] Optimized weights: {optimized_weights}")

    # 4. Calculate final training score with optimized weights
    logger.info(f"[WFO] Calculating final score for {ticker}...")
    final_result = fast_simulate(train_df, optimized_weights, horizon=horizon)
    final_objective = strategy_model.objective(final_result)
    logger.info(f"[WFO] Final result: score={final_objective.score:.3f}, trades={final_result.total_trades}")

    return FullOptimizationResult(
        ticker=ticker,
        horizon=horizon,
        weights=optimized_weights,
        train_sqn=final_objective.score,  # Now stores objective score
        total_trades=final_result.total_trades,
        per_indicator=per_indicator,
        optimized_at=datetime.utcnow()
    )


# ============================================================================
# Weight Drift Detection
# ============================================================================

def calculate_weight_drift(
    old_weights: Dict[str, float],
    new_weights: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate drift between old and new weights.
    
    Large drifts indicate noisy signals (random walk).
    Small drifts indicate stable, predictive indicators.
    
    Args:
        old_weights: Previous weight configuration
        new_weights: New weight configuration
        
    Returns:
        Dict of {indicator: drift_magnitude}
    """
    drift = {}
    
    for indicator in set(old_weights.keys()) | set(new_weights.keys()):
        old_val = old_weights.get(indicator, 1.0)
        new_val = new_weights.get(indicator, 1.0)
        drift[indicator] = abs(new_val - old_val)
    
    return drift


def classify_weight_stability(
    drift: Dict[str, float]
) -> Dict[str, str]:
    """
    Classify each indicator's stability based on weight drift.
    
    - stable: drift < 0.3 (reliable signal)
    - moderate: drift 0.3-0.7 (some signal)
    - noisy: drift > 0.7 (likely noise)
    
    Args:
        drift: Dict of {indicator: drift_magnitude}
        
    Returns:
        Dict of {indicator: stability_classification}
    """
    classifications = {}
    
    for indicator, d in drift.items():
        if d < 0.3:
            classifications[indicator] = 'stable'
        elif d < 0.7:
            classifications[indicator] = 'moderate'
        else:
            classifications[indicator] = 'noisy'
    
    return classifications

