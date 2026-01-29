"""
Premium Buying Strategy Model

Optimizes for breakout trade expectancy (buyCall, buyPut).
Uses Expectancy × Momentum as the primary objective.

Premium buying (long calls/puts) is fundamentally different:
- Win rate is typically low (30-40%)
- Average wins need to be large (2-3x the premium paid)
- Time decay works against you
- Best in low IV environments before breakouts

Objective: Maximize expectancy weighted by breakout momentum

Expectancy = (win_rate × avg_win) - ((1 - win_rate) × avg_loss)

For profitable options buying:
- avg_win / avg_loss ratio must be > 2:1
- Entry on squeezes (low IV, compressed range)
- Exit on expansion
"""

from typing import Dict, List
from .base import BaseStrategyModel, StrategyObjective
from ..wfo_simulator import SimulationResult


class PremiumBuyModel(BaseStrategyModel):
    """
    Premium buying model for buyCall/buyPut actions.

    Objective: Maximize breakout trade expectancy
    Best for: Squeeze setups, momentum breakouts, trend initiations

    Premium buying succeeds when:
    1. IV is low (cheap options)
    2. Range is compressed (squeeze)
    3. Momentum confirms direction
    4. Volume confirms breakout
    """

    strategy_class = 'premium_buy'
    min_trades = 25  # Need enough trades to measure expectancy

    # Indicators most relevant for premium buying
    priority_indicators: List[str] = [
        'squeeze',    # Squeeze = cheap premiums + pending breakout
        'momentum',   # Momentum confirmation critical
        'volume',     # Volume breakout confirmation
        'rvol',       # Relative volume for breakout strength
        'adx',        # Trend strength for continuation
        'macd',       # Momentum divergence
        'cmf',        # Money flow for direction
        'rel_momentum',  # Relative strength vs sector
        'rs_ratio',      # RRG-style relative strength
    ]

    # Weights tuned for premium buying (breakout trading)
    default_weights: Dict[str, float] = {
        'rsi': 0.7,       # RSI less important for breakouts
        'macd': 1.3,      # MACD divergence important
        'bollinger': 1.0,
        'adx': 1.4,       # Trend strength for continuation
        'cmf': 1.2,       # Money flow confirmation
        'momentum': 1.5,  # Momentum very important
        'volume': 1.6,    # Volume breakout critical
        'rvol': 1.4,      # Relative volume important
        'sma': 0.8,
        'position': 0.6,  # Less important for breakouts
        'squeeze': 2.0,   # Squeeze is the primary setup
        'rel_momentum': 1.2,  # Sector outperformance = breakout confirmation
        'rs_ratio': 1.2,      # RRG relative strength
    }

    # Minimum win/loss ratio for profitable options buying
    MIN_WIN_LOSS_RATIO = 1.5

    def objective(self, result: SimulationResult) -> StrategyObjective:
        """
        Calculate premium buying objective score.

        For premium buying, we accept lower win rates but need
        larger average wins to compensate:
        - 35% win rate with 3:1 win/loss ratio = profitable
        - 40% win rate with 2:1 win/loss ratio = profitable

        Score emphasizes expectancy and win/loss ratio.
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

        # Win/loss ratio is critical for options buying
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Expectancy calculation
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Base score is expectancy scaled up
        raw_score = expectancy * 100

        # Bonus for good win/loss ratio (compensates for low win rate)
        if win_loss_ratio >= 3.0:
            raw_score *= 1.3
        elif win_loss_ratio >= 2.0:
            raw_score *= 1.15
        elif win_loss_ratio >= self.MIN_WIN_LOSS_RATIO:
            raw_score *= 1.0
        else:
            # Poor win/loss ratio = not viable for options buying
            raw_score *= 0.5

        # Options buying can have large drawdowns, but penalize extreme ones
        if result.max_drawdown > 0.40:
            raw_score *= 0.7
        elif result.max_drawdown > 0.30:
            raw_score *= 0.85

        # Bonus for total return (options buying should have convex returns)
        if result.total_return > 0.50:  # >50% total return
            raw_score *= 1.1

        # Validity requires positive expectancy and adequate win/loss ratio
        is_valid = (
            expectancy > 0 and
            win_loss_ratio >= self.MIN_WIN_LOSS_RATIO
        )

        return StrategyObjective(
            score=raw_score if is_valid else -float('inf'),
            is_valid=is_valid,
            details={
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': win_loss_ratio,
                'expectancy': expectancy,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
            }
        )
