"""
Optimization Parameters Configuration

Centralizes all optimizable parameters that were previously magic numbers.
This module provides:
1. Dataclasses for each parameter category
2. Bounds for differential evolution optimization
3. Serialization for database storage
4. Default values matching current hardcoded behavior

Usage:
    from .optimization_params import OptimizationParams, DEFAULT_PARAMS

    # Use defaults (backward compatible)
    result = fast_simulate(df, weights, params=DEFAULT_PARAMS)

    # Use custom params
    custom = OptimizationParams()
    custom.context_multipliers.mr_trend = 0.5
    result = fast_simulate(df, weights, params=custom)
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import json
import copy


# ============================================================================
# Context Multipliers
# ============================================================================

# ============================================================================
# Indicator Period Configuration
# ============================================================================

@dataclass
class IndicatorPeriods:
    """
    Indicator period configuration for adaptive optimization.

    These control the lookback periods for key indicators.
    Optimizing periods can capture stock-specific rhythms but
    significantly increases optimization dimensionality.

    WARNING: Period optimization is experimental and can lead
    to overfitting if not properly constrained.
    """
    # RSI period (standard: 14, range: 10-20)
    rsi_period: int = 14

    # MACD periods (standard: 12/26/9)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX period (standard: 14, range: 10-20)
    adx_period: int = 14

    # Bollinger Bands period (standard: 20, range: 15-30)
    bb_period: int = 20

    @classmethod
    def bounds(cls) -> Dict[str, Tuple[int, int]]:
        """Return optimization bounds for learnable periods."""
        return {
            'rsi_period': (10, 20),
            'macd_fast': (8, 15),
            'macd_slow': (18, 30),
            'macd_signal': (5, 12),
            'adx_period': (10, 20),
            'bb_period': (15, 30),
        }

    @classmethod
    def get_learnable_bounds(cls) -> List[Tuple[float, float]]:
        """Get bounds as list of tuples for DE optimization."""
        b = cls.bounds()
        return [
            (float(b['rsi_period'][0]), float(b['rsi_period'][1])),
            (float(b['macd_fast'][0]), float(b['macd_fast'][1])),
            (float(b['macd_slow'][0]), float(b['macd_slow'][1])),
            (float(b['macd_signal'][0]), float(b['macd_signal'][1])),
            (float(b['adx_period'][0]), float(b['adx_period'][1])),
            (float(b['bb_period'][0]), float(b['bb_period'][1])),
        ]

    @classmethod
    def get_learnable_param_names(cls) -> List[str]:
        """Get parameter names for learnable periods."""
        return ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 'adx_period', 'bb_period']

    def to_vector(self) -> List[float]:
        """Convert to optimization vector."""
        return [
            float(self.rsi_period),
            float(self.macd_fast),
            float(self.macd_slow),
            float(self.macd_signal),
            float(self.adx_period),
            float(self.bb_period),
        ]

    def update_from_vector(self, vec: List[float]) -> None:
        """Update periods from optimization vector (rounds to integers)."""
        self.rsi_period = int(round(vec[0]))
        self.macd_fast = int(round(vec[1]))
        self.macd_slow = int(round(vec[2]))
        self.macd_signal = int(round(vec[3]))
        self.adx_period = int(round(vec[4]))
        self.bb_period = int(round(vec[5]))

        # Enforce MACD constraint: slow > fast
        if self.macd_slow <= self.macd_fast:
            self.macd_slow = self.macd_fast + 10

    def to_dict(self) -> Dict[str, int]:
        return {
            'rsi_period': self.rsi_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            'adx_period': self.adx_period,
            'bb_period': self.bb_period,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> 'IndicatorPeriods':
        obj = cls()
        obj.rsi_period = d.get('rsi_period', 14)
        obj.macd_fast = d.get('macd_fast', 12)
        obj.macd_slow = d.get('macd_slow', 26)
        obj.macd_signal = d.get('macd_signal', 9)
        obj.adx_period = d.get('adx_period', 14)
        obj.bb_period = d.get('bb_period', 20)
        return obj


@dataclass
class ContextMultipliers:
    """
    Weight adjustments based on market context (trending vs value).

    These multiply indicator weights based on detected context:
    - In trending markets: reduce mean-reversion, boost trend-following
    - In value contexts: boost mean-reversion signals

    Previously hardcoded in wfo_simulator.py lines 136-141.
    """
    # Mean-reversion weight in trending markets (reduce noise)
    mr_trend: float = 0.7

    # Mean-reversion weight in value/oversold contexts (boost)
    mr_value: float = 1.5

    # Trend-following weight in trending markets (boost)
    tf_trend: float = 1.3

    @classmethod
    def bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds for each parameter."""
        return {
            'mr_trend': (0.3, 1.0),   # Don't completely zero out MR
            'mr_value': (1.0, 2.5),   # At least neutral in value contexts
            'tf_trend': (1.0, 2.0),   # At least neutral when trending
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            'mr_trend': self.mr_trend,
            'mr_value': self.mr_value,
            'tf_trend': self.tf_trend,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ContextMultipliers':
        return cls(
            mr_trend=d.get('mr_trend', 0.7),
            mr_value=d.get('mr_value', 1.5),
            tf_trend=d.get('tf_trend', 1.3),
        )


# ============================================================================
# Trade Thresholds
# ============================================================================

@dataclass
class TradeThresholds:
    """
    Entry/exit thresholds for trade signals.

    The composite score must exceed these thresholds to trigger a trade:
    - score >= buy_threshold: Long entry
    - score <= sell_threshold: Short entry

    Previously hardcoded in wfo_simulator.py lines 166-167.
    """
    # Composite score threshold for long entry
    buy_threshold: float = 0.2

    # Composite score threshold for short entry
    sell_threshold: float = -0.2

    @classmethod
    def bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds."""
        return {
            'buy_threshold': (0.1, 0.4),    # Not too sensitive, not too strict
            'sell_threshold': (-0.4, -0.1), # Symmetric with buy
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeThresholds':
        return cls(
            buy_threshold=d.get('buy_threshold', 0.2),
            sell_threshold=d.get('sell_threshold', -0.2),
        )


# ============================================================================
# Signal Breakpoints
# ============================================================================

@dataclass
class SignalBreakpoints:
    """
    Indicator signal interpretation breakpoints.

    These define how raw indicator values map to signals (-1 to +1).

    Previously hardcoded in indicators.py lines 741-781.
    """
    # RSI thresholds
    rsi_oversold: float = 30.0      # Below = oversold (bullish)
    rsi_overbought: float = 70.0    # Above = overbought (bearish)

    # ADX thresholds for trend strength
    adx_no_trend: float = 20.0      # Below = no trend
    adx_trending: float = 25.0      # Above = trending
    adx_strong_trend: float = 40.0  # Above = strong trend

    # Momentum thresholds (percentage)
    momentum_strong_bull: float = 10.0
    momentum_bull: float = 5.0
    momentum_bear: float = -5.0
    momentum_strong_bear: float = -10.0

    # Volume ratio thresholds
    volume_very_high: float = 2.0
    volume_high: float = 1.5
    volume_above_normal: float = 1.0

    @classmethod
    def bounds(cls) -> Dict[str, Tuple[float, float]]:
        """Get optimization bounds for key thresholds."""
        return {
            'rsi_oversold': (20.0, 40.0),
            'rsi_overbought': (60.0, 80.0),
            'adx_no_trend': (15.0, 25.0),
            'adx_trending': (20.0, 35.0),
            'adx_strong_trend': (30.0, 50.0),
        }

    def to_dict(self) -> Dict[str, float]:
        return {
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'adx_no_trend': self.adx_no_trend,
            'adx_trending': self.adx_trending,
            'adx_strong_trend': self.adx_strong_trend,
            'momentum_strong_bull': self.momentum_strong_bull,
            'momentum_bull': self.momentum_bull,
            'momentum_bear': self.momentum_bear,
            'momentum_strong_bear': self.momentum_strong_bear,
            'volume_very_high': self.volume_very_high,
            'volume_high': self.volume_high,
            'volume_above_normal': self.volume_above_normal,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SignalBreakpoints':
        return cls(
            rsi_oversold=d.get('rsi_oversold', 30.0),
            rsi_overbought=d.get('rsi_overbought', 70.0),
            adx_no_trend=d.get('adx_no_trend', 20.0),
            adx_trending=d.get('adx_trending', 25.0),
            adx_strong_trend=d.get('adx_strong_trend', 40.0),
            momentum_strong_bull=d.get('momentum_strong_bull', 10.0),
            momentum_bull=d.get('momentum_bull', 5.0),
            momentum_bear=d.get('momentum_bear', -5.0),
            momentum_strong_bear=d.get('momentum_strong_bear', -10.0),
            volume_very_high=d.get('volume_very_high', 2.0),
            volume_high=d.get('volume_high', 1.5),
            volume_above_normal=d.get('volume_above_normal', 1.0),
        )


# ============================================================================
# Regime Adjustments
# ============================================================================

# Key indicators for regime-specific learning
# These are the most impacted by regime changes
REGIME_LEARNABLE_INDICATORS = ['rsi', 'bollinger', 'macd', 'adx']

# All 6 market regimes
REGIME_NAMES = [
    'bear_volatile', 'bear_quiet', 'neutral_chop',
    'neutral_volatile', 'bull_quiet', 'bull_volatile'
]


@dataclass
class RegimeAdjustments:
    """
    Per-regime indicator weight multipliers.

    Different market regimes require different indicator weighting:
    - BEAR_VOLATILE: Disable most signals (cash is king)
    - NEUTRAL_CHOP: Boost mean-reversion (range trading)
    - BULL_*: Normal operation with slight adjustments

    The learnable subset (4 indicators × 6 regimes = 24 params) can be
    optimized via differential evolution when optimize_regime=True.

    Previously hardcoded in regime.py lines 254-300.
    """
    bear_volatile: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 0.0,
        'bollinger': 0.0,
        'cmf': 0.5,
        'position': 0.0,
        'macd': 1.0,
        'adx': 1.0,
        'momentum': 1.0,
        'sma': 1.0,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    bear_quiet: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 0.3,
        'bollinger': 0.3,
        'macd': 1.2,
        'adx': 1.0,
        'momentum': 1.0,
        'cmf': 1.0,
        'position': 0.5,
        'sma': 1.0,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    neutral_chop: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 1.5,
        'bollinger': 1.5,
        'cmf': 1.2,
        'macd': 0.5,
        'adx': 0.5,
        'momentum': 0.5,
        'position': 1.2,
        'sma': 0.5,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    neutral_volatile: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 1.0,
        'bollinger': 1.0,
        'cmf': 1.0,
        'macd': 1.0,
        'adx': 1.0,
        'momentum': 1.0,
        'position': 1.0,
        'sma': 1.0,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    bull_quiet: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 0.8,
        'bollinger': 0.8,
        'macd': 1.2,
        'adx': 1.0,
        'momentum': 1.0,
        'cmf': 1.0,
        'position': 0.8,
        'sma': 1.0,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    bull_volatile: Dict[str, float] = field(default_factory=lambda: {
        'rsi': 1.2,
        'bollinger': 1.2,
        'macd': 1.2,
        'adx': 1.0,
        'momentum': 1.0,
        'cmf': 1.0,
        'position': 1.0,
        'sma': 1.0,
        'volume': 1.0,
        'rvol': 1.0,
        'squeeze': 1.0,
    })

    def get_adjustments(self, regime: str) -> Dict[str, float]:
        """Get adjustments for a specific regime."""
        regime_map = {
            'BEAR_VOLATILE': self.bear_volatile,
            'BEAR_QUIET': self.bear_quiet,
            'NEUTRAL_CHOP': self.neutral_chop,
            'NEUTRAL_VOLATILE': self.neutral_volatile,
            'BULL_QUIET': self.bull_quiet,
            'BULL_VOLATILE': self.bull_volatile,
        }
        return regime_map.get(regime, {})

    @classmethod
    def get_learnable_bounds(cls) -> List[Tuple[float, float]]:
        """
        Get optimization bounds for learnable regime multipliers.

        Returns bounds for 4 indicators × 6 regimes = 24 parameters.
        Order: [bear_volatile_rsi, bear_volatile_bollinger, ..., bull_volatile_adx]
        """
        bounds = []
        for regime in REGIME_NAMES:
            for indicator in REGIME_LEARNABLE_INDICATORS:
                # Bounds depend on indicator role:
                # - Mean reversion (rsi, bollinger): Allow 0 (disable) to 2.0 (boost)
                # - Trend following (macd, adx): Allow 0.3 to 2.0 (don't fully disable)
                if indicator in ('rsi', 'bollinger'):
                    bounds.append((0.0, 2.0))
                else:
                    bounds.append((0.3, 2.0))
        return bounds

    @classmethod
    def get_learnable_param_names(cls) -> List[str]:
        """Get parameter names for learnable regime multipliers."""
        names = []
        for regime in REGIME_NAMES:
            for indicator in REGIME_LEARNABLE_INDICATORS:
                names.append(f'{regime}_{indicator}')
        return names

    def to_learnable_vector(self) -> List[float]:
        """
        Extract learnable regime parameters to optimization vector.

        Order: [bear_volatile_rsi, bear_volatile_bollinger, ..., bull_volatile_adx]
        """
        regime_map = {
            'bear_volatile': self.bear_volatile,
            'bear_quiet': self.bear_quiet,
            'neutral_chop': self.neutral_chop,
            'neutral_volatile': self.neutral_volatile,
            'bull_quiet': self.bull_quiet,
            'bull_volatile': self.bull_volatile,
        }
        vec = []
        for regime in REGIME_NAMES:
            adjustments = regime_map[regime]
            for indicator in REGIME_LEARNABLE_INDICATORS:
                vec.append(adjustments.get(indicator, 1.0))
        return vec

    def update_from_learnable_vector(self, vec: List[float]) -> None:
        """
        Update regime adjustments from optimization vector.

        Args:
            vec: List of 24 floats (4 indicators × 6 regimes)
        """
        regime_map = {
            'bear_volatile': self.bear_volatile,
            'bear_quiet': self.bear_quiet,
            'neutral_chop': self.neutral_chop,
            'neutral_volatile': self.neutral_volatile,
            'bull_quiet': self.bull_quiet,
            'bull_volatile': self.bull_volatile,
        }
        idx = 0
        for regime in REGIME_NAMES:
            for indicator in REGIME_LEARNABLE_INDICATORS:
                regime_map[regime][indicator] = vec[idx]
                idx += 1

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            'bear_volatile': self.bear_volatile,
            'bear_quiet': self.bear_quiet,
            'neutral_chop': self.neutral_chop,
            'neutral_volatile': self.neutral_volatile,
            'bull_quiet': self.bull_quiet,
            'bull_volatile': self.bull_volatile,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RegimeAdjustments':
        obj = cls()
        if 'bear_volatile' in d:
            obj.bear_volatile = d['bear_volatile']
        if 'bear_quiet' in d:
            obj.bear_quiet = d['bear_quiet']
        if 'neutral_chop' in d:
            obj.neutral_chop = d['neutral_chop']
        if 'neutral_volatile' in d:
            obj.neutral_volatile = d['neutral_volatile']
        if 'bull_quiet' in d:
            obj.bull_quiet = d['bull_quiet']
        if 'bull_volatile' in d:
            obj.bull_volatile = d['bull_volatile']
        return obj


# ============================================================================
# Main OptimizationParams Class
# ============================================================================

@dataclass
class OptimizationParams:
    """
    Complete set of optimizable parameters.

    This is the main class that aggregates all parameter categories.
    Use DEFAULT_PARAMS for backward-compatible default behavior.

    Example:
        # Load defaults
        params = OptimizationParams()

        # Customize
        params.context_multipliers.mr_trend = 0.5
        params.trade_thresholds.buy_threshold = 0.15

        # Use in simulation
        result = fast_simulate(df, weights, params=params)

        # Serialize for database
        json_str = params.to_json()

        # Deserialize
        loaded = OptimizationParams.from_json(json_str)
    """
    context_multipliers: ContextMultipliers = field(default_factory=ContextMultipliers)
    trade_thresholds: TradeThresholds = field(default_factory=TradeThresholds)
    signal_breakpoints: SignalBreakpoints = field(default_factory=SignalBreakpoints)
    regime_adjustments: RegimeAdjustments = field(default_factory=RegimeAdjustments)
    indicator_periods: IndicatorPeriods = field(default_factory=IndicatorPeriods)

    def to_vector(self) -> List[float]:
        """
        Flatten key optimizable params to a vector for differential_evolution.

        Order: [context_multipliers..., trade_thresholds..., signal_breakpoints...]
        """
        return [
            # Context multipliers (3)
            self.context_multipliers.mr_trend,
            self.context_multipliers.mr_value,
            self.context_multipliers.tf_trend,
            # Trade thresholds (2)
            self.trade_thresholds.buy_threshold,
            self.trade_thresholds.sell_threshold,
            # Signal breakpoints - key ones only (5)
            self.signal_breakpoints.rsi_oversold,
            self.signal_breakpoints.rsi_overbought,
            self.signal_breakpoints.adx_no_trend,
            self.signal_breakpoints.adx_trending,
            self.signal_breakpoints.adx_strong_trend,
        ]

    @classmethod
    def from_vector(cls, vec: List[float]) -> 'OptimizationParams':
        """Reconstruct params from optimization vector."""
        params = cls()

        # Context multipliers
        params.context_multipliers.mr_trend = vec[0]
        params.context_multipliers.mr_value = vec[1]
        params.context_multipliers.tf_trend = vec[2]

        # Trade thresholds
        params.trade_thresholds.buy_threshold = vec[3]
        params.trade_thresholds.sell_threshold = vec[4]

        # Signal breakpoints
        params.signal_breakpoints.rsi_oversold = vec[5]
        params.signal_breakpoints.rsi_overbought = vec[6]
        params.signal_breakpoints.adx_no_trend = vec[7]
        params.signal_breakpoints.adx_trending = vec[8]
        params.signal_breakpoints.adx_strong_trend = vec[9]

        return params

    @classmethod
    def get_bounds(cls) -> List[Tuple[float, float]]:
        """Get bounds for all parameters in vector order."""
        return [
            # Context multipliers
            *ContextMultipliers.bounds().values(),
            # Trade thresholds
            *TradeThresholds.bounds().values(),
            # Signal breakpoints (key ones)
            SignalBreakpoints.bounds()['rsi_oversold'],
            SignalBreakpoints.bounds()['rsi_overbought'],
            SignalBreakpoints.bounds()['adx_no_trend'],
            SignalBreakpoints.bounds()['adx_trending'],
            SignalBreakpoints.bounds()['adx_strong_trend'],
        ]

    # ========================================================================
    # Extended methods for regime multiplier optimization
    # ========================================================================

    def to_vector_with_regime(self) -> List[float]:
        """
        Flatten params INCLUDING learnable regime multipliers to optimization vector.

        Order: [base_params (10)..., regime_multipliers (24)...]
        Total: 34 parameters
        """
        return self.to_vector() + self.regime_adjustments.to_learnable_vector()

    @classmethod
    def from_vector_with_regime(cls, vec: List[float]) -> 'OptimizationParams':
        """
        Reconstruct params from extended optimization vector (with regime multipliers).

        Args:
            vec: List of 34 floats (10 base + 24 regime)
        """
        base_vec = vec[:10]
        regime_vec = vec[10:]

        params = cls.from_vector(base_vec)
        params.regime_adjustments.update_from_learnable_vector(regime_vec)

        return params

    @classmethod
    def get_bounds_with_regime(cls) -> List[Tuple[float, float]]:
        """
        Get bounds for all parameters INCLUDING regime multipliers.

        Returns 34 bounds (10 base + 24 regime).
        """
        return cls.get_bounds() + RegimeAdjustments.get_learnable_bounds()

    @classmethod
    def get_param_names_with_regime(cls) -> List[str]:
        """Get parameter names including regime multipliers."""
        return cls.get_param_names() + RegimeAdjustments.get_learnable_param_names()

    # ========================================================================
    # Extended methods for indicator period optimization
    # ========================================================================

    def to_vector_with_periods(self) -> List[float]:
        """
        Flatten params INCLUDING indicator periods to optimization vector.

        Order: [base_params (10)..., indicator_periods (6)...]
        Total: 16 parameters
        """
        return self.to_vector() + self.indicator_periods.to_vector()

    @classmethod
    def from_vector_with_periods(cls, vec: List[float]) -> 'OptimizationParams':
        """
        Reconstruct params from extended optimization vector (with indicator periods).

        Args:
            vec: List of 16 floats (10 base + 6 periods)
        """
        base_vec = vec[:10]
        period_vec = vec[10:]

        params = cls.from_vector(base_vec)
        params.indicator_periods.update_from_vector(period_vec)

        return params

    @classmethod
    def get_bounds_with_periods(cls) -> List[Tuple[float, float]]:
        """
        Get bounds for all parameters INCLUDING indicator periods.

        Returns 16 bounds (10 base + 6 periods).
        """
        return cls.get_bounds() + IndicatorPeriods.get_learnable_bounds()

    def to_vector_full(self, include_regime: bool = True, include_periods: bool = True) -> List[float]:
        """
        Flatten all params to optimization vector.

        Order: [base_params (10)..., regime_multipliers (24)..., indicator_periods (6)...]
        Total: up to 40 parameters
        """
        vec = self.to_vector()
        if include_regime:
            vec += self.regime_adjustments.to_learnable_vector()
        if include_periods:
            vec += self.indicator_periods.to_vector()
        return vec

    @classmethod
    def from_vector_full(
        cls,
        vec: List[float],
        include_regime: bool = True,
        include_periods: bool = True
    ) -> 'OptimizationParams':
        """
        Reconstruct params from full optimization vector.

        Args:
            vec: Full optimization vector
            include_regime: Whether regime multipliers are included (24 params)
            include_periods: Whether indicator periods are included (6 params)
        """
        idx = 10  # Base params
        params = cls.from_vector(vec[:idx])

        if include_regime:
            regime_end = idx + 24
            params.regime_adjustments.update_from_learnable_vector(vec[idx:regime_end])
            idx = regime_end

        if include_periods:
            params.indicator_periods.update_from_vector(vec[idx:idx + 6])

        return params

    @classmethod
    def get_bounds_full(
        cls,
        include_regime: bool = True,
        include_periods: bool = True
    ) -> List[Tuple[float, float]]:
        """
        Get bounds for full optimization vector.

        Returns up to 40 bounds (10 base + 24 regime + 6 periods).
        """
        bounds = cls.get_bounds()
        if include_regime:
            bounds += RegimeAdjustments.get_learnable_bounds()
        if include_periods:
            bounds += IndicatorPeriods.get_learnable_bounds()
        return bounds

    @classmethod
    def get_param_names(cls) -> List[str]:
        """Get parameter names in vector order (for logging/debugging)."""
        return [
            'mr_trend', 'mr_value', 'tf_trend',
            'buy_threshold', 'sell_threshold',
            'rsi_oversold', 'rsi_overbought',
            'adx_no_trend', 'adx_trending', 'adx_strong_trend',
        ]

    def to_json(self) -> str:
        """Serialize to JSON for database storage."""
        return json.dumps({
            'context_multipliers': self.context_multipliers.to_dict(),
            'trade_thresholds': self.trade_thresholds.to_dict(),
            'signal_breakpoints': self.signal_breakpoints.to_dict(),
            'regime_adjustments': self.regime_adjustments.to_dict(),
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationParams':
        """Deserialize from JSON."""
        d = json.loads(json_str)
        return cls(
            context_multipliers=ContextMultipliers.from_dict(d.get('context_multipliers', {})),
            trade_thresholds=TradeThresholds.from_dict(d.get('trade_thresholds', {})),
            signal_breakpoints=SignalBreakpoints.from_dict(d.get('signal_breakpoints', {})),
            regime_adjustments=RegimeAdjustments.from_dict(d.get('regime_adjustments', {})),
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'context_multipliers': self.context_multipliers.to_dict(),
            'trade_thresholds': self.trade_thresholds.to_dict(),
            'signal_breakpoints': self.signal_breakpoints.to_dict(),
            'regime_adjustments': self.regime_adjustments.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'OptimizationParams':
        """Create from dictionary."""
        return cls(
            context_multipliers=ContextMultipliers.from_dict(d.get('context_multipliers', {})),
            trade_thresholds=TradeThresholds.from_dict(d.get('trade_thresholds', {})),
            signal_breakpoints=SignalBreakpoints.from_dict(d.get('signal_breakpoints', {})),
            regime_adjustments=RegimeAdjustments.from_dict(d.get('regime_adjustments', {})),
        )

    def copy(self) -> 'OptimizationParams':
        """Create a deep copy."""
        return OptimizationParams.from_json(self.to_json())


# ============================================================================
# Global Default Instance
# ============================================================================

# This instance uses all the hardcoded defaults for backward compatibility.
# All existing code should continue to work unchanged by using DEFAULT_PARAMS.
DEFAULT_PARAMS = OptimizationParams()
