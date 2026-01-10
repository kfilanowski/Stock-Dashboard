"""
Premium Selling Strategy Model

Optimizes for premium selling success (openCSP, openCC).
Uses Win Rate × Average Premium Capture as the primary objective.

Premium selling strategies (CSP = Cash Secured Put, CC = Covered Call)
are fundamentally different from directional trading:
- High win rate is critical (aim for 70%+)
- Average wins are small (premium received)
- Average losses can be large (assignment risk)
- Theta decay is on your side

Objective: Maximize (win_rate × avg_win) - ((1 - win_rate) × avg_loss × penalty)

The penalty factor accounts for the asymmetric risk profile:
- A 5% loss wipes out many 1% wins
- High win rate reduces frequency of these losses
"""

from typing import Dict, List
from .base import BaseStrategyModel, StrategyObjective
from ..wfo_simulator import SimulationResult


class PremiumSellModel(BaseStrategyModel):
    """
    Premium selling model for openCSP/openCC actions.

    Objective: Maximize win rate × average premium capture
    Best for: Range-bound markets, high IV environments, income generation

    Premium selling is about:
    1. Selling when IV is high (rich premiums)
    2. Picking range-bound stocks (no assignment)
    3. Timing entries at extremes (support for CSP, resistance for CC)
    """

    strategy_class = 'premium_sell'
    min_trades = 20  # Premium selling trades less frequently

    # Indicators most relevant for premium selling
    priority_indicators: List[str] = [
        'rsi',       # Extremes = good entry points
        'adx',       # Low ADX = range-bound = good
        'bollinger', # Band position for timing
        'position',  # Price position in range
        'squeeze',   # Volatility expansion = rich premiums
        'volume',    # Volume confirmation
    ]

    # Weights tuned for premium selling
    default_weights: Dict[str, float] = {
        'rsi': 1.5,       # RSI extremes very important
        'macd': 0.5,      # Less important for range-bound
        'bollinger': 1.3,
        'adx': 1.8,       # ADX crucial - low = good for selling
        'cmf': 0.8,
        'momentum': 0.4,  # Momentum less relevant
        'volume': 0.7,
        'rvol': 0.8,
        'sma': 0.6,       # Trend less important
        'position': 1.5,  # Position in range very important
        'squeeze': 1.6,   # Volatility expansion = rich premiums
    }

    # Target win rate for premium selling
    TARGET_WIN_RATE = 0.70

    # Loss penalty multiplier (accounts for asymmetric risk)
    LOSS_PENALTY = 3.0

    def objective(self, result: SimulationResult) -> StrategyObjective:
        """
        Calculate premium selling objective score.

        For premium selling, we want:
        - High win rate (70%+ ideal)
        - Consistent small wins
        - Avoiding large losses (assignment events)

        Score = (win_rate × avg_win) - ((1 - win_rate) × avg_loss × penalty)

        This rewards high win rates while penalizing large losses heavily.
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

        win_rate = result.win_rate
        avg_win = abs(result.avg_win) if result.avg_win else 0.01
        avg_loss = abs(result.avg_loss) if result.avg_loss else 0.01

        # Expected value per trade accounting for loss asymmetry
        expected_win = win_rate * avg_win
        expected_loss = (1 - win_rate) * avg_loss * self.LOSS_PENALTY
        raw_score = expected_win - expected_loss

        # Bonus for hitting target win rate
        if win_rate >= self.TARGET_WIN_RATE:
            raw_score *= 1.2
        elif win_rate >= 0.60:
            raw_score *= 1.0
        else:
            # Penalize low win rates heavily for premium selling
            raw_score *= 0.5

        # Scale to be comparable with SQN range (roughly 0-5)
        score = raw_score * 100

        # Penalize extreme drawdowns even more for premium selling
        # (one big loss shouldn't wipe out months of premium)
        if result.max_drawdown > 0.20:
            score *= 0.6
        elif result.max_drawdown > 0.10:
            score *= 0.8

        # Require positive expectancy
        is_valid = result.expectancy > 0 and win_rate >= 0.50

        return StrategyObjective(
            score=score if is_valid else -float('inf'),
            is_valid=is_valid,
            details={
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expected_win': expected_win,
                'expected_loss': expected_loss,
                'raw_score': raw_score,
                'total_trades': result.total_trades,
                'max_drawdown': result.max_drawdown,
            }
        )
