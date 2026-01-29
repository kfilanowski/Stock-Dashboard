"""
Differential Evolution Optimizer for Walk-Forward Optimization

A global optimizer that finds better optima than coordinate descent by
searching the entire weight + parameter space simultaneously. This captures
indicator interactions that greedy optimization misses.

Usage:
    from .wfo_optimizer_de import differential_evolution_optimize

    result = differential_evolution_optimize(
        df, horizon=3, progress_callback=callback
    )
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Import scipy's differential evolution
try:
    from scipy.optimize import differential_evolution, OptimizeResult
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("[DE] scipy not available - differential evolution disabled")

from .wfo_simulator import (
    fast_simulate,
    SimulationResult,
    MIN_TRADES_FOR_SIGNIFICANCE,
    FULL_CONFIDENCE_TRADES
)
from .wfo_optimizer import (
    DEFAULT_WEIGHTS,
    DEFAULT_DIRECTIONS,
    INDICATORS_TO_OPTIMIZE,
    DIRECTION_INDICATORS,
    validate_stability,
    InsufficientVolatilityError,
    FullOptimizationResult,
    OptimizationResult
)
from .optimization_params import OptimizationParams, DEFAULT_PARAMS
from .strategy_models import get_strategy_model


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DEConfig:
    """Configuration for Differential Evolution optimizer."""
    # DE strategy: 'best1bin' is good balance of exploitation/exploration
    strategy: str = 'best1bin'

    # Max iterations (generations)
    maxiter: int = 100

    # Population size multiplier (popsize * n_dimensions)
    popsize: int = 15

    # Convergence tolerance
    tol: float = 0.01

    # Mutation factor (F): [0, 2], higher = more exploration
    mutation: Tuple[float, float] = (0.5, 1.0)

    # Crossover probability (CR): [0, 1], higher = more exploration
    recombination: float = 0.7

    # Use Latin Hypercube for initial population
    init: str = 'latinhypercube'

    # Number of parallel workers (scipy default: -1 = all cores)
    workers: int = 1  # Keep at 1 to avoid issues with DataFrame copying

    # Whether to also optimize params (Phase 1) in addition to weights
    optimize_params: bool = False

    # Whether to optimize regime multipliers (Phase 2)
    # This adds 24 parameters (4 indicators × 6 regimes)
    optimize_regime: bool = False

    # Whether to optimize indicator periods (Phase 3)
    # This adds 6 parameters (RSI, MACD fast/slow/signal, ADX, BB periods)
    # WARNING: Period optimization significantly increases dimensionality
    # and can lead to overfitting. Use with caution.
    optimize_periods: bool = False

    # Seed for reproducibility (None = random)
    seed: Optional[int] = None


# Default configuration
DEFAULT_DE_CONFIG = DEConfig()

# Weight bounds for each indicator [min, max]
WEIGHT_BOUNDS = (0.0, 2.5)

# Direction bounds [-1, +1] for mean-reversion indicators
DIRECTION_BOUNDS = (-1.0, 1.0)


# ============================================================================
# Bounds Building
# ============================================================================

def build_weight_bounds(
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_directions: bool = True
) -> List[Tuple[float, float]]:
    """
    Build bounds for weight optimization vector.

    Args:
        indicators: List of indicator names to optimize
        include_directions: Whether to include direction parameters

    Returns:
        List of (min, max) tuples for each weight (and direction if enabled)
    """
    bounds = [WEIGHT_BOUNDS for _ in indicators]

    # Add direction bounds for mean-reversion indicators
    if include_directions:
        bounds.extend([DIRECTION_BOUNDS for _ in DIRECTION_INDICATORS])

    return bounds


def build_full_bounds(
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_params: bool = False,
    include_regime: bool = False,
    include_periods: bool = False,
    include_directions: bool = True
) -> List[Tuple[float, float]]:
    """
    Build bounds for full optimization (weights + directions + optional params).

    Args:
        indicators: Indicators to optimize
        include_params: Whether to include OptimizationParams (10 params)
        include_regime: Whether to include regime multipliers (24 params)
        include_periods: Whether to include indicator periods (6 params)
        include_directions: Whether to include direction parameters (4 params)

    Returns:
        Combined bounds list
    """
    bounds = build_weight_bounds(indicators, include_directions)

    if include_params:
        # Use the full bounds method that supports all combinations
        bounds.extend(OptimizationParams.get_bounds_full(
            include_regime=include_regime,
            include_periods=include_periods
        ))
    else:
        # Handle individual optional components
        if include_regime:
            from .optimization_params import RegimeAdjustments
            bounds.extend(RegimeAdjustments.get_learnable_bounds())
        if include_periods:
            from .optimization_params import IndicatorPeriods
            bounds.extend(IndicatorPeriods.get_learnable_bounds())

    return bounds


# ============================================================================
# Vector Conversion
# ============================================================================

def vector_to_weights(
    vec: np.ndarray,
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_directions: bool = True
) -> Dict[str, float]:
    """
    Convert optimization vector to weights dict.

    Args:
        vec: Numpy array of weight values
        indicators: Indicator names (in vector order)
        include_directions: Whether vector includes direction parameters

    Returns:
        Dict mapping indicator -> weight (and direction params if included)
    """
    weights = DEFAULT_WEIGHTS.copy()

    # Initialize direction defaults
    for dir_ind in DIRECTION_INDICATORS:
        dir_key = f"{dir_ind}_dir"
        weights[dir_key] = DEFAULT_DIRECTIONS.get(dir_ind, 1.0)

    # Set indicator weights
    for i, indicator in enumerate(indicators):
        weights[indicator] = float(vec[i])

    # Set direction parameters if included
    if include_directions:
        n_weights = len(indicators)
        for j, dir_ind in enumerate(DIRECTION_INDICATORS):
            dir_key = f"{dir_ind}_dir"
            if n_weights + j < len(vec):
                weights[dir_key] = float(vec[n_weights + j])

    return weights


def weights_to_vector(
    weights: Dict[str, float],
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_directions: bool = True
) -> np.ndarray:
    """
    Convert weights dict to optimization vector.

    Args:
        weights: Weights dictionary
        indicators: Indicator names (in vector order)
        include_directions: Whether to include direction parameters

    Returns:
        Numpy array of weight values (and directions if included)
    """
    vec = [weights.get(ind, 1.0) for ind in indicators]

    if include_directions:
        for dir_ind in DIRECTION_INDICATORS:
            dir_key = f"{dir_ind}_dir"
            vec.append(weights.get(dir_key, DEFAULT_DIRECTIONS.get(dir_ind, 1.0)))

    return np.array(vec)


def vector_to_config(
    vec: np.ndarray,
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_params: bool = False,
    include_regime: bool = False,
    include_periods: bool = False,
    include_directions: bool = True
) -> Tuple[Dict[str, float], OptimizationParams]:
    """
    Convert full optimization vector to weights and params.

    Args:
        vec: Full optimization vector
        indicators: Indicator names
        include_params: Whether base params (10) are in vector
        include_regime: Whether regime multipliers (24) are in vector
        include_periods: Whether indicator periods (6) are in vector
        include_directions: Whether direction params are in vector

    Returns:
        Tuple of (weights_dict, params)
    """
    n_weights = len(indicators)
    n_directions = len(DIRECTION_INDICATORS) if include_directions else 0
    n_weights_and_dirs = n_weights + n_directions

    # Extract weights and directions together
    weights = vector_to_weights(vec[:n_weights_and_dirs], indicators, include_directions)

    if include_params and len(vec) > n_weights_and_dirs:
        # Use the full reconstruction method that handles all combinations
        params = OptimizationParams.from_vector_full(
            list(vec[n_weights_and_dirs:]),
            include_regime=include_regime,
            include_periods=include_periods
        )
    else:
        # Handle individual optional components without base params
        import copy
        params = copy.deepcopy(DEFAULT_PARAMS)
        idx = n_weights_and_dirs

        if include_regime and len(vec) > idx:
            params.regime_adjustments.update_from_learnable_vector(list(vec[idx:idx + 24]))
            idx += 24

        if include_periods and len(vec) > idx:
            params.indicator_periods.update_from_vector(list(vec[idx:idx + 6]))

    return weights, params


# ============================================================================
# Objective Function
# ============================================================================

def create_objective_function(
    df: pd.DataFrame,
    horizon: int,
    indicators: List[str] = INDICATORS_TO_OPTIMIZE,
    include_params: bool = False,
    include_regime: bool = False,
    include_periods: bool = False,
    include_directions: bool = True,
    strategy_class: str = 'all',
    min_trades: int = 5
) -> Callable[[np.ndarray], float]:
    """
    Create objective function for differential_evolution.

    DE minimizes, so we return negative SQN (we want to maximize SQN).
    Invalid configurations return a large positive value (penalized).

    Args:
        df: Price data with indicators
        horizon: Holding period
        indicators: Indicator names
        include_params: Whether to optimize base params (10)
        include_regime: Whether to optimize regime multipliers (24)
        include_periods: Whether to optimize indicator periods (6)
        include_directions: Whether to optimize signal directions (4)
        strategy_class: Strategy class for objective
        min_trades: Minimum trades required for valid result

    Returns:
        Objective function(vector) -> float
    """
    strategy_model = get_strategy_model(strategy_class)

    def objective(vec: np.ndarray) -> float:
        weights, params = vector_to_config(
            vec, indicators, include_params, include_regime, include_periods, include_directions
        )

        try:
            result = fast_simulate(df, weights, horizon=horizon, params=params)

            # Penalize insufficient trades
            if result.total_trades < min_trades:
                return 100.0  # High positive = bad (DE minimizes)

            # Use strategy-specific objective
            obj = strategy_model.objective(result)

            if not obj.is_valid or not np.isfinite(obj.score):
                return 100.0

            # Return negative score (DE minimizes, we want to maximize)
            return -obj.score

        except Exception as e:
            logger.debug(f"[DE] Objective exception: {e}")
            return 100.0

    return objective


# ============================================================================
# DE Optimizer Result
# ============================================================================

@dataclass
class DEOptimizationResult:
    """Result from Differential Evolution optimization."""
    weights: Dict[str, float]
    params: OptimizationParams
    score: float              # Best objective score (positive = good)
    n_iterations: int         # Number of generations
    n_evaluations: int        # Total function evaluations
    converged: bool           # Whether DE converged
    convergence_message: str  # DE termination message


# ============================================================================
# Main DE Optimizer
# ============================================================================

def differential_evolution_optimize(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 3,
    config: DEConfig = DEFAULT_DE_CONFIG,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    strategy_class: str = 'all'
) -> FullOptimizationResult:
    """
    Run Differential Evolution optimization for a ticker.

    This is a global optimizer that explores the full weight space,
    unlike coordinate descent which is greedy and may get stuck in
    local optima.

    Advantages over coordinate descent:
    - Captures indicator interactions (e.g., RSI + MACD synergy)
    - More likely to find global optimum
    - Naturally handles multimodal objective landscapes

    Disadvantages:
    - Slower (5-10x more function evaluations)
    - May overfit if not regularized
    - Requires scipy

    Args:
        df: Price data with indicators calculated
        ticker: Stock ticker
        horizon: Holding period (3 or 15)
        config: DE configuration
        progress_callback: Optional callback(indicator, progress)
        strategy_class: Strategy class for optimization

    Returns:
        FullOptimizationResult with optimized weights

    Raises:
        ImportError: If scipy is not installed
        InsufficientVolatilityError: If not enough trades
    """
    if not SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for differential evolution. "
            "Install with: pip install scipy"
        )

    logger.info(f"[DE] Starting optimization for {ticker}, horizon={horizon}, strategy={strategy_class}")

    # Get strategy model for defaults and indicators
    strategy_model = get_strategy_model(strategy_class)
    indicators = strategy_model.get_indicators_to_optimize()

    # Build bounds (weights + directions + optional params + optional regime + optional periods)
    bounds = build_full_bounds(
        indicators,
        include_params=config.optimize_params,
        include_regime=config.optimize_regime,
        include_periods=config.optimize_periods,
        include_directions=True  # Always optimize directions
    )
    n_weights = len(indicators)
    n_directions = len(DIRECTION_INDICATORS)
    n_params = len(OptimizationParams.get_bounds()) if config.optimize_params else 0
    n_regime = 24 if config.optimize_regime else 0  # 4 indicators × 6 regimes
    n_periods = 6 if config.optimize_periods else 0  # RSI, MACD×3, ADX, BB
    logger.info(
        f"[DE] Optimizing {len(bounds)} dimensions "
        f"({n_weights} weights + {n_directions} directions + {n_params} params + {n_regime} regime + {n_periods} periods)"
    )

    # Create objective function
    objective = create_objective_function(
        df, horizon, indicators,
        include_params=config.optimize_params,
        include_regime=config.optimize_regime,
        include_periods=config.optimize_periods,
        include_directions=True,  # Always optimize directions
        strategy_class=strategy_class
    )

    # Track progress
    n_evals = [0]
    best_score = [float('inf')]

    def callback(xk, convergence):
        """Callback for DE progress updates."""
        n_evals[0] += 1
        current_score = objective(xk)
        if current_score < best_score[0]:
            best_score[0] = current_score
            if progress_callback:
                # Convert to percentage (rough estimate based on maxiter)
                pct = min(100, (n_evals[0] / (config.maxiter * config.popsize * len(bounds))) * 100)
                progress_callback("evolving", pct)

    # Run differential evolution
    logger.info(f"[DE] Running DE with maxiter={config.maxiter}, popsize={config.popsize}")

    result: OptimizeResult = differential_evolution(
        objective,
        bounds,
        strategy=config.strategy,
        maxiter=config.maxiter,
        popsize=config.popsize,
        tol=config.tol,
        mutation=config.mutation,
        recombination=config.recombination,
        init=config.init,
        seed=config.seed,
        workers=config.workers,
        updating='deferred' if config.workers != 1 else 'immediate',
        callback=callback
    )

    logger.info(f"[DE] Optimization complete: success={result.success}, nit={result.nit}, nfev={result.nfev}")

    # Extract best weights and params (and regime/periods if optimized)
    weights, params = vector_to_config(
        result.x, indicators,
        include_params=config.optimize_params,
        include_regime=config.optimize_regime,
        include_periods=config.optimize_periods,
        include_directions=True
    )

    # Log direction parameters learned
    for dir_ind in DIRECTION_INDICATORS:
        dir_key = f"{dir_ind}_dir"
        dir_val = weights.get(dir_key, 1.0)
        if abs(dir_val - 1.0) > 0.3:
            mode = 'MOMENTUM' if dir_val < 0 else 'mean-rev' if dir_val > 0 else 'neutral'
            logger.info(f"[DE] {dir_key}: {dir_val:.2f} ({mode})")

    # Calculate final result with best weights
    final_sim = fast_simulate(df, weights, horizon=horizon, params=params)
    final_obj = strategy_model.objective(final_sim)

    logger.info(f"[DE] Final: score={final_obj.score:.3f}, trades={final_sim.total_trades}")

    # Check minimum trades
    if final_sim.total_trades < MIN_TRADES_FOR_SIGNIFICANCE:
        raise InsufficientVolatilityError(
            ticker=ticker,
            trades=final_sim.total_trades,
            window_days=len(df)
        )

    # Build per-indicator results (DE doesn't have per-indicator breakdown)
    per_indicator = {}
    for ind in indicators:
        per_indicator[ind] = OptimizationResult(
            indicator=ind,
            optimal_weight=weights.get(ind, 1.0),
            sqn_score=final_obj.score,  # Same for all (from combined optimization)
            stability_passed=True,  # DE doesn't do per-indicator stability
            coarse_results={},
            fine_results={}
        )

    # Add direction results
    for dir_ind in DIRECTION_INDICATORS:
        dir_key = f"{dir_ind}_dir"
        per_indicator[dir_key] = OptimizationResult(
            indicator=dir_key,
            optimal_weight=weights.get(dir_key, 1.0),
            sqn_score=final_obj.score,
            stability_passed=True,
            coarse_results={},
            fine_results={}
        )

    if progress_callback:
        progress_callback("complete", 100.0)

    return FullOptimizationResult(
        ticker=ticker,
        horizon=horizon,
        weights=weights,
        train_sqn=final_obj.score,
        total_trades=final_sim.total_trades,
        per_indicator=per_indicator,
        optimized_at=datetime.utcnow(),
        reduced_confidence=final_sim.total_trades < FULL_CONFIDENCE_TRADES
    )


# ============================================================================
# Hybrid Optimizer (DE + Local Refinement)
# ============================================================================

def hybrid_optimize(
    df: pd.DataFrame,
    ticker: str,
    horizon: int = 3,
    config: DEConfig = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    strategy_class: str = 'all'
) -> FullOptimizationResult:
    """
    Hybrid optimization: DE for global search, then coordinate descent for refinement.

    This combines the best of both approaches:
    1. DE finds the approximate global optimum
    2. Coordinate descent refines the solution locally

    Use this when you want the most accurate optimization and have time.

    Args:
        df: Price data
        ticker: Stock ticker
        horizon: Holding period
        config: DE configuration (uses a lighter version by default)
        progress_callback: Progress updates
        strategy_class: Strategy class

    Returns:
        FullOptimizationResult with refined weights
    """
    # Use lighter DE config for first phase
    if config is None:
        config = DEConfig(
            maxiter=50,    # Fewer iterations
            popsize=10,    # Smaller population
            tol=0.02       # Looser convergence
        )

    # Phase 1: Global search with DE
    if progress_callback:
        progress_callback("de_phase", 0)

    de_result = differential_evolution_optimize(
        df, ticker, horizon, config,
        progress_callback=lambda ind, pct: progress_callback("de_phase", pct * 0.7) if progress_callback else None,
        strategy_class=strategy_class
    )

    # Phase 2: Local refinement with coordinate descent
    if progress_callback:
        progress_callback("cd_phase", 70)

    from .wfo_optimizer import two_pass_coordinate_descent

    refined_weights, per_indicator = two_pass_coordinate_descent(
        df,
        horizon=horizon,
        initial_weights=de_result.weights,  # Start from DE solution
        progress_callback=lambda ind, pct: progress_callback("cd_phase", 70 + pct * 0.3) if progress_callback else None,
        strategy_class=strategy_class
    )

    # Calculate final score
    strategy_model = get_strategy_model(strategy_class)
    final_sim = fast_simulate(df, refined_weights, horizon=horizon)
    final_obj = strategy_model.objective(final_sim)

    if progress_callback:
        progress_callback("complete", 100)

    return FullOptimizationResult(
        ticker=ticker,
        horizon=horizon,
        weights=refined_weights,
        train_sqn=final_obj.score,
        total_trades=final_sim.total_trades,
        per_indicator=per_indicator,
        optimized_at=datetime.utcnow(),
        reduced_confidence=final_sim.total_trades < FULL_CONFIDENCE_TRADES
    )
