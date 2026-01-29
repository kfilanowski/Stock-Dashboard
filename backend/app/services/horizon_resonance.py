"""
Horizon Resonance Discovery

Instead of hard-coding trading horizons (3d, 15d), this module discovers
the optimal "resonant frequency" for each stock by testing multiple horizons
and measuring signal-return correlation (Information Coefficient).

Key Insight: Different stocks have different natural trading cycles.
- TSLA might resonate at 5-day volatility cycles
- KO might resonate at 20-day trend cycles
- Forcing all stocks to 3d/15d is suboptimal

Methodology:
1. Calculate composite signal using default weights
2. Test horizons from 2 to 30 days (configurable)
3. Measure Information Coefficient (IC) for each horizon
4. Identify horizons with statistically significant IC
5. Return the top resonant horizons for WFO calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

from .indicators import calculate_indicator_signals
from .wfo_simulator import apply_weights_to_signals
from .wfo_optimizer import DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Horizon search range
MIN_HORIZON = 2
MAX_HORIZON = 30
HORIZON_STEP = 1  # Test every day for granularity

# Minimum IC to consider a horizon "significant"
# IC > 0.02 is generally considered meaningful in quant finance
MIN_SIGNIFICANT_IC = 0.02

# Minimum t-statistic for statistical significance (95% confidence)
MIN_T_STAT = 1.96

# Number of top horizons to return
TOP_HORIZONS_COUNT = 3


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HorizonResult:
    """Result for a single horizon test."""
    horizon: int
    ic: float                    # Information Coefficient (correlation)
    ic_t_stat: float            # T-statistic for IC significance
    avg_return: float           # Average forward return
    hit_rate: float             # % of correct direction predictions
    signal_coverage: float      # % of days with actionable signals
    is_significant: bool        # IC statistically significant?

    def to_dict(self) -> dict:
        return {
            'horizon': int(self.horizon),
            'ic': float(round(self.ic, 4)),
            'ic_t_stat': float(round(self.ic_t_stat, 2)),
            'avg_return': float(round(self.avg_return, 4)),
            'hit_rate': float(round(self.hit_rate, 4)),
            'signal_coverage': float(round(self.signal_coverage, 4)),
            'is_significant': bool(self.is_significant)
        }


@dataclass
class ResonanceResult:
    """Complete resonance analysis result."""
    ticker: str
    horizons_tested: int
    heatmap: Dict[int, HorizonResult]  # horizon -> result
    top_horizons: List[int]            # Best horizons sorted by IC
    recommended_short: Optional[int]   # Best short-term horizon (2-7 days)
    recommended_medium: Optional[int]  # Best medium-term horizon (8-15 days)
    recommended_long: Optional[int]    # Best long-term horizon (16-30 days)

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker,
            'horizons_tested': self.horizons_tested,
            'heatmap': {h: r.to_dict() for h, r in self.heatmap.items()},
            'top_horizons': self.top_horizons,
            'recommended': {
                'short': self.recommended_short,
                'medium': self.recommended_medium,
                'long': self.recommended_long
            }
        }


# ============================================================================
# Information Coefficient Calculation
# ============================================================================

def calculate_ic(
    signals: pd.Series,
    forward_returns: pd.Series
) -> Tuple[float, float]:
    """
    Calculate Information Coefficient and its t-statistic.

    IC = Pearson correlation between signal and forward returns

    Args:
        signals: Predicted signal values
        forward_returns: Actual forward returns

    Returns:
        Tuple of (IC, t-statistic)
    """
    # Align and drop NaN
    valid_mask = ~(signals.isna() | forward_returns.isna())
    sig = signals[valid_mask].values
    ret = forward_returns[valid_mask].values

    n = len(sig)
    if n < 30:  # Need enough samples
        return 0.0, 0.0

    # Pearson correlation
    ic = np.corrcoef(sig, ret)[0, 1]

    if np.isnan(ic):
        return 0.0, 0.0

    # T-statistic: t = IC * sqrt(n-2) / sqrt(1 - IC^2)
    if abs(ic) >= 1.0:
        t_stat = np.inf if ic > 0 else -np.inf
    else:
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)

    return ic, t_stat


def calculate_hit_rate(
    signals: pd.Series,
    forward_returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Calculate directional hit rate.

    Hit rate = % of times signal correctly predicted return direction.

    Args:
        signals: Predicted signals
        forward_returns: Actual returns
        threshold: Signal threshold for "actionable" signals

    Returns:
        Hit rate as decimal (0-1)
    """
    valid_mask = ~(signals.isna() | forward_returns.isna())
    sig = signals[valid_mask]
    ret = forward_returns[valid_mask]

    # Only count "actionable" signals (above threshold)
    actionable_mask = sig.abs() >= threshold
    if actionable_mask.sum() == 0:
        return 0.0

    sig_actionable = sig[actionable_mask]
    ret_actionable = ret[actionable_mask]

    # Correct if sign matches
    correct = (np.sign(sig_actionable) == np.sign(ret_actionable)).sum()

    return correct / len(sig_actionable)


# ============================================================================
# Main Resonance Discovery
# ============================================================================

def discover_resonance(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    min_horizon: int = MIN_HORIZON,
    max_horizon: int = MAX_HORIZON,
    step: int = HORIZON_STEP
) -> Dict[int, HorizonResult]:
    """
    Discover resonant horizons by testing IC across multiple timeframes.

    Args:
        df: Price DataFrame with OHLCV data
        weights: Indicator weights (uses defaults if None)
        min_horizon: Minimum horizon to test
        max_horizon: Maximum horizon to test
        step: Step size between horizons

    Returns:
        Dict of {horizon: HorizonResult}
    """
    weights = weights or DEFAULT_WEIGHTS

    # Calculate signals once (expensive operation)
    logger.info(f"[RESONANCE] Calculating signals for {len(df)} days...")
    signals_df = calculate_indicator_signals(df)
    composite_signal = apply_weights_to_signals(signals_df, weights)

    results = {}

    for horizon in range(min_horizon, max_horizon + 1, step):
        # Calculate forward returns for this horizon
        forward_returns = df['close'].pct_change(horizon).shift(-horizon)

        # Calculate IC
        ic, t_stat = calculate_ic(composite_signal, forward_returns)

        # Calculate hit rate (using 0.1 threshold for "actionable" signals)
        hit_rate = calculate_hit_rate(composite_signal, forward_returns, threshold=0.1)

        # Average return when signal is positive
        valid_mask = ~(composite_signal.isna() | forward_returns.isna())
        positive_signal_mask = (composite_signal > 0.1) & valid_mask
        avg_return = forward_returns[positive_signal_mask].mean() if positive_signal_mask.sum() > 0 else 0.0

        # Signal coverage (% of days with actionable signals)
        actionable = (composite_signal.abs() >= 0.1).sum()
        signal_coverage = actionable / len(composite_signal)

        # Is this horizon statistically significant?
        is_significant = abs(ic) >= MIN_SIGNIFICANT_IC and abs(t_stat) >= MIN_T_STAT

        results[horizon] = HorizonResult(
            horizon=horizon,
            ic=ic,
            ic_t_stat=t_stat,
            avg_return=avg_return if not np.isnan(avg_return) else 0.0,
            hit_rate=hit_rate,
            signal_coverage=signal_coverage,
            is_significant=is_significant
        )

        logger.debug(f"[RESONANCE] Horizon {horizon}d: IC={ic:.4f}, t={t_stat:.2f}, hit={hit_rate:.2%}")

    return results


def find_optimal_horizons(
    heatmap: Dict[int, HorizonResult],
    top_n: int = TOP_HORIZONS_COUNT
) -> List[int]:
    """
    Find the top N horizons by IC.

    Args:
        heatmap: Results from discover_resonance()
        top_n: Number of top horizons to return

    Returns:
        List of horizon values sorted by IC (descending)
    """
    # Sort by absolute IC (we care about predictive power, not direction)
    sorted_horizons = sorted(
        heatmap.items(),
        key=lambda x: abs(x[1].ic),
        reverse=True
    )

    # Filter to significant only
    significant = [h for h, r in sorted_horizons if r.is_significant]

    if len(significant) >= top_n:
        return significant[:top_n]

    # If not enough significant, return top by IC anyway
    return [h for h, _ in sorted_horizons[:top_n]]


def find_recommended_by_range(
    heatmap: Dict[int, HorizonResult]
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Find the best horizon in each range (short/medium/long).

    This ensures diversity in recommended horizons.

    Returns:
        Tuple of (short_horizon, medium_horizon, long_horizon)
    """
    def best_in_range(min_d: int, max_d: int) -> Optional[int]:
        candidates = {h: r for h, r in heatmap.items() if min_d <= h <= max_d}
        if not candidates:
            return None

        # Prefer significant horizons
        significant = {h: r for h, r in candidates.items() if r.is_significant}
        if significant:
            candidates = significant

        # Return the one with highest IC
        return max(candidates.items(), key=lambda x: abs(x[1].ic))[0]

    short = best_in_range(2, 7)      # 2-7 days: swing trades
    medium = best_in_range(8, 15)    # 8-15 days: short-term trend
    long = best_in_range(16, 30)     # 16-30 days: trend following

    return short, medium, long


def analyze_resonance(
    df: pd.DataFrame,
    ticker: str,
    weights: Optional[Dict[str, float]] = None,
    min_horizon: int = MIN_HORIZON,
    max_horizon: int = MAX_HORIZON
) -> ResonanceResult:
    """
    Complete resonance analysis for a stock.

    Args:
        df: Price DataFrame
        ticker: Stock symbol
        weights: Optional custom weights
        min_horizon: Min horizon to test
        max_horizon: Max horizon to test

    Returns:
        ResonanceResult with heatmap and recommendations
    """
    logger.info(f"[RESONANCE] Analyzing {ticker} for horizons {min_horizon}-{max_horizon} days...")

    # Discover resonance across all horizons
    heatmap = discover_resonance(df, weights, min_horizon, max_horizon)

    # Find top horizons
    top_horizons = find_optimal_horizons(heatmap)

    # Find best in each range
    short, medium, long = find_recommended_by_range(heatmap)

    result = ResonanceResult(
        ticker=ticker,
        horizons_tested=len(heatmap),
        heatmap=heatmap,
        top_horizons=top_horizons,
        recommended_short=short,
        recommended_medium=medium,
        recommended_long=long
    )

    logger.info(f"[RESONANCE] {ticker}: Top horizons = {top_horizons}, "
                f"Recommended = short:{short}, medium:{medium}, long:{long}")

    return result


# ============================================================================
# Visualization Helper (for API response)
# ============================================================================

def format_heatmap_for_display(result: ResonanceResult) -> List[Dict]:
    """
    Format heatmap data for frontend visualization.

    Returns list sorted by horizon for easy charting.
    """
    rows = []
    for horizon in sorted(result.heatmap.keys()):
        hr = result.heatmap[horizon]
        rows.append({
            'horizon': int(horizon),
            'ic': float(hr.ic),
            'ic_t_stat': float(hr.ic_t_stat),
            'hit_rate': float(hr.hit_rate),
            'is_significant': bool(hr.is_significant),
            'is_recommended': bool(horizon in result.top_horizons)
        })
    return rows
