"""
Base class for strategy-specific WFO models.

Each strategy model defines:
1. An objective function for optimization
2. Which indicators are most relevant
3. How to interpret simulation results
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from ..wfo_simulator import SimulationResult


@dataclass
class StrategyObjective:
    """Result of evaluating a strategy's objective function."""
    score: float              # Primary optimization target (higher is better)
    is_valid: bool           # Whether result meets minimum requirements
    details: Dict[str, float] # Additional metrics for logging


class BaseStrategyModel(ABC):
    """
    Base class for strategy-specific optimization models.

    Subclasses must implement:
    - objective(): The scoring function to maximize
    - priority_indicators: Which indicators matter most for this strategy
    - min_trades: Minimum trades required for valid results
    """

    # Strategy identifier
    strategy_class: str = 'base'

    # Minimum trades required for statistical validity (lowered from 30 to allow stable stocks)
    min_trades: int = 20

    # Indicators to prioritize in optimization (others get default weight)
    priority_indicators: List[str] = []

    # Default indicator weights for this strategy
    default_weights: Dict[str, float] = {
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

    @abstractmethod
    def objective(self, result: SimulationResult) -> StrategyObjective:
        """
        Calculate the objective score for optimization.

        This is what the optimizer tries to maximize. Different strategies
        have different goals:
        - Directional: Maximize SQN (risk-adjusted returns)
        - Premium sell: Maximize win rate × average premium
        - Premium buy: Maximize expectancy on breakout moves

        Args:
            result: Simulation result to evaluate

        Returns:
            StrategyObjective with score and validity flag
        """
        pass

    def get_indicators_to_optimize(self) -> List[str]:
        """
        Get the list of indicators to optimize for this strategy.

        Returns priority indicators if defined, otherwise all indicators.
        """
        if self.priority_indicators:
            return self.priority_indicators
        return list(self.default_weights.keys())

    def get_default_weights(self) -> Dict[str, float]:
        """Get default indicator weights for this strategy."""
        return self.default_weights.copy()

    def adjust_result_for_strategy(
        self,
        result: SimulationResult,
        regime: Optional[str] = None
    ) -> SimulationResult:
        """
        Optional: Adjust simulation results based on strategy context.

        Override in subclasses if needed (e.g., to penalize certain regimes).
        """
        return result

    def is_result_valid(self, result: SimulationResult) -> bool:
        """Check if a simulation result meets minimum requirements."""
        return (
            result.total_trades >= self.min_trades and
            result.expectancy > 0 and
            not np.isnan(result.sqn)
        )
