"""
Directional Strategy Model

Optimizes for price direction prediction (buyShares, sellShares).
Uses SQN (System Quality Number) as the primary objective.

SQN = (Average R-multiple) / (StdDev of R-multiples) * sqrt(N)

Where R-multiple = (P&L per trade) / (Risk per trade)

Good SQN scores:
- 1.6-1.9: Below average
- 2.0-2.4: Average
- 2.5-2.9: Good
- 3.0-5.0: Excellent
- 5.0+: Superb (rare)
"""

from typing import Dict, List
from .base import BaseStrategyModel, StrategyObjective
from ..wfo_simulator import SimulationResult


class DirectionalModel(BaseStrategyModel):
    """
    Directional trading model for buyShares/sellShares actions.

    Objective: Maximize SQN (risk-adjusted returns)
    Best for: Trend following, momentum trading, mean reversion

    This is the default/legacy model that optimizes for overall
    prediction accuracy of price direction.
    """

    strategy_class = 'directional'
    min_trades = 30

    # Indicators most relevant for directional trading
    priority_indicators: List[str] = [
        'rsi',       # Momentum/mean-reversion
        'macd',      # Trend following
        'sma',       # Trend alignment
        'adx',       # Trend strength
        'momentum',  # Price momentum
        'cmf',       # Money flow confirmation
        'position',  # Price position in range
        'volume',    # Volume confirmation
        'bollinger', # Volatility bands
    ]

    # Weights tuned for directional trading
    default_weights: Dict[str, float] = {
        'rsi': 1.0,
        'macd': 1.2,      # Slightly favor trend indicators
        'bollinger': 0.8,
        'adx': 1.2,       # Trend strength important
        'cmf': 1.0,
        'momentum': 1.0,
        'volume': 0.8,
        'rvol': 0.6,
        'sma': 1.3,       # SMA alignment important for direction
        'position': 0.9,
        'squeeze': 0.5,   # Less relevant for directional
    }

    def objective(self, result: SimulationResult) -> StrategyObjective:
        """
        Calculate SQN-based objective score.

        SQN is the primary metric for directional trading because it
        measures risk-adjusted performance across many trades.
        """
        if result.total_trades < self.min_trades:
            return StrategyObjective(
                score=-float('inf'),
                is_valid=False,
                details={
                    'reason': 'insufficient_trades',
                    'trades': result.total_trades,
                    'required': self.min_trades
                }
            )

        if result.expectancy <= 0:
            return StrategyObjective(
                score=-float('inf'),
                is_valid=False,
                details={
                    'reason': 'negative_expectancy',
                    'expectancy': result.expectancy
                }
            )

        # Primary score is SQN
        score = result.sqn

        # Penalize extreme drawdowns
        if result.max_drawdown > 0.25:  # >25% drawdown
            score *= 0.8
        elif result.max_drawdown > 0.15:  # >15% drawdown
            score *= 0.9

        # Bonus for high profit factor (gross profit / gross loss)
        if result.profit_factor > 2.0:
            score *= 1.1
        elif result.profit_factor > 1.5:
            score *= 1.05

        return StrategyObjective(
            score=score,
            is_valid=True,
            details={
                'raw_sqn': result.sqn,
                'expectancy': result.expectancy,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
            }
        )
