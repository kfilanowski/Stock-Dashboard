"""
Walk-Forward Optimization Optimizer

Two-pass Coordinate Descent optimizer with stability validation.
Optimizes indicator weights while avoiding curve fitting.

Supports multiple optimizer backends:
- COORDINATE_DESCENT: Fast, greedy two-pass optimization
- DIFFERENTIAL_EVOLUTION: Global optimizer using scipy.optimize
- HYBRID (default): DE global search + CD local refinement

CONSTRAINTS (Pre-Flight Checklist):
1. Optimize WEIGHTS (0.0-2.5), optionally periods (via DE)
2. Transaction cost >= 0.1% (realistic friction)
3. Force cash in un-tradeable regimes (BEAR_VOLATILE)

ENSEMBLE CONSISTENCY NOTE:
-------------------------
There is an intentional asymmetry between backend and frontend scoring:

Backend (WFO):
- Optimizes raw indicator weights via simulation
- Uses fast_simulate() which applies weights directly to signals
- Includes regime filtering (blocks BEAR_VOLATILE) and learned regime multipliers
- Does NOT include frontend ensemble layers (consensus, regime rules)

Frontend (Stock Scoring):
- Applies calibrated weights from WFO
- THEN adds ensemble layers:
  - Regime rules (specific indicator blocking per regime)
  - Consensus scoring (agreement across indicators)
  - Additional safety layers

This asymmetry is INTENTIONAL:
1. WFO captures the core signal quality per indicator
2. Frontend ensemble adds a safety/confirmation layer
3. Trying to optimize the full ensemble would be:
   - Much slower (more complex simulation)
   - Risk overfitting to ensemble interactions
   - Harder to interpret which component adds value

The current approach lets WFO find "signal strength" while the
frontend ensemble provides "signal confirmation" as a separate layer.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

from .wfo_simulator import (
    fast_simulate,
    SimulationResult,
    simulate_weight_grid,
    MIN_TRANSACTION_COST,
    MIN_TRADES_FOR_SIGNIFICANCE,
    FULL_CONFIDENCE_TRADES
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

# Bayesian Shrinkage Parameters
# Uses sample-size-aware posterior instead of fixed shrinkage.
# Formula: posterior_weight = (prior_strength * default + n_trades * optimal) / (prior_strength + n_trades)
# This automatically shrinks more when you have fewer trades (more uncertainty).
PRIOR_STRENGTH = 30  # Equivalent of 30 trades worth of confidence in prior (default)
# The old fixed shrinkage is retained as fallback when trade count is unavailable
FALLBACK_SHRINKAGE_FACTOR = 0.3  # 30% shrinkage toward default

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
    'vwap': 1.0,  # Volume-weighted average price
    # Relative strength indicators (vs sector)
    'rel_momentum': 1.0,
    'rs_ratio': 1.0,
}

# Default signal directions for mean-reversion indicators
# +1.0 = mean reversion (oversold = bullish, overbought = bearish)
# -1.0 = momentum (oversold = bearish continuation, overbought = bullish continuation)
# 0.0 = neutral/disabled
# The optimizer can learn values in between (-1 to +1)
DEFAULT_DIRECTIONS = {
    'rsi': 1.0,        # Default: mean reversion (oversold = buy)
    'bollinger': 1.0,  # Default: mean reversion (below lower band = buy)
    'position': 1.0,   # Default: mean reversion (near 52w low = buy)
    'vwap': 1.0,       # Default: mean reversion (below VWAP = buy)
}

# Indicators to optimize (weights 0-2.5)
INDICATORS_TO_OPTIMIZE = [
    'rsi', 'macd', 'bollinger', 'adx', 'cmf',
    'momentum', 'volume', 'sma', 'position', 'vwap',
    'rel_momentum', 'rs_ratio'  # Relative strength vs sector
]

# Indicators with learnable direction (-1 to +1)
# These are mean-reversion indicators that might work better as momentum signals
DIRECTION_INDICATORS = ['rsi', 'bollinger', 'position', 'vwap']

# Direction optimization grid (separate from weight grid)
DIRECTION_COARSE_GRID = [-1.0, 0.0, 1.0]  # Momentum, neutral, mean-reversion
DIRECTION_FINE_OFFSETS = [-0.3, 0.3]  # Fine-tune around winner


# ============================================================================
# Optimizer Type Enum
# ============================================================================

class OptimizerType(str, Enum):
    """Available optimization algorithms."""

    COORDINATE_DESCENT = 'coordinate_descent'
    """
    Fast, greedy two-pass optimizer. Default choice.

    Pros: Fast, interpretable, works with small datasets
    Cons: May miss global optimum, doesn't capture indicator interactions
    """

    DIFFERENTIAL_EVOLUTION = 'differential_evolution'
    """
    Global optimizer using scipy.optimize.differential_evolution.

    Pros: Finds global optimum, captures indicator interactions
    Cons: Slower (5-10x), requires scipy, may overfit
    """

    HYBRID = 'hybrid'
    """
    DE for global search, then coordinate descent for local refinement.

    Pros: Best of both worlds
    Cons: Slowest option
    """


# Default optimizer
# HYBRID captures indicator interactions via DE global search,
# then refines locally with coordinate descent. Best for weekly/monthly calibration.
DEFAULT_OPTIMIZER = OptimizerType.HYBRID


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
            f"(need {MIN_TRADES_FOR_SIGNIFICANCE}+). Stock may be too stable."
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
    reduced_confidence: bool = False  # True if trades < FULL_CONFIDENCE_TRADES (30)
    window_results: Optional[List] = None  # Rolling window results for persistence
    overfit_warning: bool = False  # True if test SQN < 0.5 * train SQN (significant performance drop)
    avg_train_sqn: Optional[float] = None  # Average training SQN across windows
    avg_test_sqn: Optional[float] = None  # Average test (out-of-sample) SQN across windows
    avg_gross_sqn: Optional[float] = None  # Average gross SQN (before costs) - shows signal quality


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
    strategy_class: str = 'all',
    optimize_directions: bool = True,
    sector_df: pd.DataFrame = None
) -> Tuple[Dict[str, float], Dict[str, OptimizationResult]]:
    """
    Two-pass Coordinate Descent optimizer with optional direction learning.

    Pass 1 (Coarse): Test [0.0, 1.0, 2.0] for each indicator weight
    Pass 2 (Fine): Test [±0.2, ±0.4] around coarse winner
    Pass 3 (Direction): For mean-reversion indicators, learn optimal direction
                       (-1 = momentum, +1 = mean reversion)

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
        optimize_directions: Whether to optimize signal directions (default True)
        sector_df: Optional sector ETF DataFrame for relative strength calculations

    Returns:
        Tuple of (optimized_weights, per_indicator_results)
        Weights dict includes '{indicator}_dir' keys for direction parameters
    """
    # Get strategy model for objective function
    strategy_model = get_strategy_model(strategy_class)

    # Use strategy-specific defaults if no initial weights provided
    weights = (initial_weights or strategy_model.get_default_weights()).copy()

    # Initialize direction parameters with defaults
    for indicator in DIRECTION_INDICATORS:
        dir_key = f"{indicator}_dir"
        if dir_key not in weights:
            weights[dir_key] = DEFAULT_DIRECTIONS.get(indicator, 1.0)

    # Use strategy-specific indicators if not specified
    indicators = indicators or strategy_model.get_indicators_to_optimize()

    results: Dict[str, OptimizationResult] = {}

    # Calculate total items to optimize (weights + directions if enabled)
    total_items = len(indicators)
    if optimize_directions:
        total_items += len(DIRECTION_INDICATORS)

    def get_objective_score(sim_result: SimulationResult) -> float:
        """Get objective score from strategy model."""
        objective = strategy_model.objective(sim_result)
        return objective.score if objective.is_valid else -np.inf

    # Calculate total tests for multiple testing correction
    num_tests = calculate_total_tests(coarse_grid, fine_offsets)
    current_item = 0

    # =========================================================================
    # PHASE 1: Optimize indicator weights (0 to 2.5)
    # =========================================================================
    for i, indicator in enumerate(indicators):
        if progress_callback:
            progress = (current_item / total_items) * 100
            progress_callback(indicator, progress)
        current_item += 1

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

            result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
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
            result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
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

            result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
            score = get_objective_score(result)

            fine_results[test_weight] = score

            if score > best_fine_score:
                best_fine_score = score
                best_fine_weight = test_weight

        # Stability check
        stability_passed = validate_stability(
            indicator, best_fine_weight, fine_results
        )

        # Get trade count for Bayesian shrinkage
        # Re-simulate with best fine weight to get accurate trade count
        test_weights = weights.copy()
        test_weights[indicator] = best_fine_weight
        best_result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
        n_trades = best_result.total_trades

        # Apply multiple testing correction with Bayesian shrinkage
        corrected_weight, corrected_score, was_shrunk = apply_multiple_testing_correction(
            indicator=indicator,
            optimal_weight=best_fine_weight,
            optimal_score=best_fine_score,
            default_score=default_score,
            num_tests=num_tests,
            default_weight=default_weight,
            n_trades=n_trades  # Pass trade count for Bayesian shrinkage
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

    # =========================================================================
    # PHASE 2: Optimize signal directions (-1 to +1) for mean-reversion indicators
    # This allows the optimizer to FLIP signals for momentum stocks
    # =========================================================================
    if optimize_directions:
        dir_num_tests = calculate_total_tests(DIRECTION_COARSE_GRID, DIRECTION_FINE_OFFSETS)

        for dir_indicator in DIRECTION_INDICATORS:
            dir_key = f"{dir_indicator}_dir"

            if progress_callback:
                progress = (current_item / total_items) * 100
                progress_callback(f"{dir_indicator} direction", progress)
            current_item += 1

            # Skip if base indicator weight is zero (disabled)
            if weights.get(dir_indicator, 1.0) < 0.1:
                logger.debug(f"[DIR] Skipping {dir_key} - base weight is near zero")
                continue

            # Default direction is mean-reversion (+1.0)
            default_dir = DEFAULT_DIRECTIONS.get(dir_indicator, 1.0)

            # Pass 1: Coarse direction grid
            coarse_dir_results = {}
            best_coarse_dir = weights.get(dir_key, default_dir)
            best_coarse_dir_score = -np.inf
            default_dir_score = -np.inf

            for test_dir in DIRECTION_COARSE_GRID:
                test_weights = weights.copy()
                test_weights[dir_key] = test_dir

                result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
                score = get_objective_score(result)

                coarse_dir_results[test_dir] = score

                if abs(test_dir - default_dir) < 0.01:
                    default_dir_score = score

                if score > best_coarse_dir_score:
                    best_coarse_dir_score = score
                    best_coarse_dir = test_dir

            # If default wasn't tested, compute it
            if default_dir_score == -np.inf:
                test_weights = weights.copy()
                test_weights[dir_key] = default_dir
                result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
                default_dir_score = get_objective_score(result)

            # Pass 2: Fine direction grid
            fine_dir_results = {best_coarse_dir: best_coarse_dir_score}
            best_fine_dir = best_coarse_dir
            best_fine_dir_score = best_coarse_dir_score

            for offset in DIRECTION_FINE_OFFSETS:
                test_dir = best_coarse_dir + offset

                # Clamp to valid range [-1, 1]
                if test_dir < -1.0 or test_dir > 1.0:
                    continue

                test_weights = weights.copy()
                test_weights[dir_key] = test_dir

                result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
                score = get_objective_score(result)

                fine_dir_results[test_dir] = score

                if score > best_fine_dir_score:
                    best_fine_dir_score = score
                    best_fine_dir = test_dir

            # Stability check for direction
            dir_stability = validate_stability(dir_key, best_fine_dir, fine_dir_results)

            # Get trade count for correction
            test_weights = weights.copy()
            test_weights[dir_key] = best_fine_dir
            dir_result = fast_simulate(df, test_weights, horizon=horizon, sector_df=sector_df)
            dir_n_trades = dir_result.total_trades

            # Apply correction (with tighter threshold for directions)
            corrected_dir, corrected_dir_score, dir_was_shrunk = apply_multiple_testing_correction(
                indicator=dir_key,
                optimal_weight=best_fine_dir,
                optimal_score=best_fine_dir_score,
                default_score=default_dir_score,
                num_tests=dir_num_tests,
                default_weight=default_dir,
                n_trades=dir_n_trades
            )

            # Round direction to nearest 0.1
            corrected_dir = round(corrected_dir, 1)

            # Lock in direction
            weights[dir_key] = corrected_dir

            # Log significant direction changes
            if abs(corrected_dir - default_dir) > 0.3:
                logger.info(
                    f"[DIR] {dir_key}: {default_dir:.1f} -> {corrected_dir:.1f} "
                    f"({'MOMENTUM' if corrected_dir < 0 else 'mean-rev' if corrected_dir > 0 else 'neutral'})"
                )

            results[dir_key] = OptimizationResult(
                indicator=dir_key,
                optimal_weight=corrected_dir,
                sqn_score=corrected_dir_score,
                stability_passed=dir_stability and not dir_was_shrunk,
                coarse_results=coarse_dir_results,
                fine_results=fine_dir_results
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
    default_weight: float = 1.0,
    n_trades: Optional[int] = None
) -> Tuple[float, float, bool]:
    """
    Apply multiple testing correction to reduce overfitting risk.

    When testing many weight combinations, we're likely to find some that
    perform well by chance. This function applies three corrections:

    1. Minimum improvement threshold: Only accept non-default weights if
       they show meaningful improvement (guards against noise)

    2. Bayesian shrinkage toward default: Pull weights toward the prior
       (default=1.0) based on sample size. With more trades, we trust
       the optimization more; with fewer, we rely more on the prior.
       Formula: posterior = (prior_strength * default + n_trades * optimal) / (prior_strength + n_trades)

    3. Bonferroni-style penalty: Reduce the effective score based on
       the number of tests performed

    Args:
        indicator: Name of indicator being optimized
        optimal_weight: The proposed optimal weight from optimization
        optimal_score: Score at optimal weight
        default_score: Score at default weight (1.0)
        num_tests: Total number of weight values tested
        default_weight: The default/prior weight (usually 1.0)
        n_trades: Number of trades in the optimization (for Bayesian shrinkage)

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

    # 2. Apply Bayesian shrinkage toward default
    # With more trades, we trust the optimal weight more
    # With fewer trades, we rely more on the prior (default)
    if n_trades is not None and n_trades > 0:
        # Bayesian posterior: weighted average where weight is proportional to sample size
        shrunk_weight = (PRIOR_STRENGTH * default_weight + n_trades * optimal_weight) / (PRIOR_STRENGTH + n_trades)
        shrinkage_pct = PRIOR_STRENGTH / (PRIOR_STRENGTH + n_trades) * 100
        logger.debug(
            f"[MTC] {indicator}: Bayesian shrinkage with n_trades={n_trades}, "
            f"prior_strength={PRIOR_STRENGTH}, shrinkage={shrinkage_pct:.1f}%"
        )
    else:
        # Fallback to fixed linear shrinkage if trade count unavailable
        shrunk_weight = FALLBACK_SHRINKAGE_FACTOR * default_weight + (1 - FALLBACK_SHRINKAGE_FACTOR) * optimal_weight

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
    strategy_class: str = 'all',
    sector_df: pd.DataFrame = None
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
        sector_df: Optional sector ETF DataFrame for relative strength calculations

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

        result = fast_simulate(window_df, weights, horizon=horizon, sector_df=sector_df)

        if result.total_trades >= min_trades:
            return window_days, result.total_trades

        # Expand window
        window_days += step_days

    # Max window still insufficient
    final_window = min(max_days, len(df))
    window_df = df.tail(final_window).reset_index(drop=True)
    result = fast_simulate(window_df, weights, horizon=horizon, sector_df=sector_df)

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
    strategy_class: str = 'all',
    optimizer: OptimizerType = DEFAULT_OPTIMIZER,
    sector_df: pd.DataFrame = None
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
        optimizer: Optimization algorithm to use (default: COORDINATE_DESCENT)
        sector_df: Optional sector ETF DataFrame for relative strength calculations

    Returns:
        FullOptimizationResult with optimized weights and metrics
    """
    # Route to appropriate optimizer
    if optimizer == OptimizerType.DIFFERENTIAL_EVOLUTION:
        from .wfo_optimizer_de import differential_evolution_optimize
        return differential_evolution_optimize(
            df, ticker, horizon,
            progress_callback=progress_callback,
            strategy_class=strategy_class
        )
    elif optimizer == OptimizerType.HYBRID:
        from .wfo_optimizer_de import hybrid_optimize
        return hybrid_optimize(
            df, ticker, horizon,
            progress_callback=progress_callback,
            strategy_class=strategy_class
        )

    # Default: Coordinate Descent
    logger.info(f"[WFO] optimize_for_ticker: {ticker}, horizon={horizon}, strategy={strategy_class}, data_rows={len(df)}")

    # Get strategy model for defaults
    strategy_model = get_strategy_model(strategy_class)
    weights = initial_weights or strategy_model.get_default_weights()
    logger.info(f"[WFO] Initial weights ({strategy_class}): {weights}")

    # 1. Get adaptive window size (uses strategy-specific min_trades)
    try:
        logger.info(f"[WFO] Getting adaptive window for {ticker}...")
        window_days, trade_count = get_adaptive_window(
            df, weights, horizon, strategy_class=strategy_class, sector_df=sector_df
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
        strategy_class=strategy_class,
        sector_df=sector_df
    )
    logger.info(f"[WFO] Optimized weights: {optimized_weights}")

    # 4. Calculate final training score with optimized weights
    logger.info(f"[WFO] Calculating final score for {ticker}...")
    final_result = fast_simulate(train_df, optimized_weights, horizon=horizon, sector_df=sector_df)
    final_objective = strategy_model.objective(final_result)
    logger.info(f"[WFO] Final result: score={final_objective.score:.3f}, trades={final_result.total_trades}")

    # Flag as reduced confidence if fewer than 30 trades
    is_reduced_confidence = final_result.total_trades < FULL_CONFIDENCE_TRADES
    if is_reduced_confidence:
        logger.info(f"[WFO] {ticker} horizon={horizon}: Reduced confidence ({final_result.total_trades} < {FULL_CONFIDENCE_TRADES} trades)")

    return FullOptimizationResult(
        ticker=ticker,
        horizon=horizon,
        weights=optimized_weights,
        train_sqn=final_objective.score,  # Now stores objective score
        total_trades=final_result.total_trades,
        per_indicator=per_indicator,
        optimized_at=datetime.utcnow(),
        reduced_confidence=is_reduced_confidence
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

