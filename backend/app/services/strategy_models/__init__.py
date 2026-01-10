"""
Strategy-specific WFO models.

Each strategy class optimizes indicator weights using different objective functions:
- DirectionalModel: Optimizes for price direction (SQN)
- PremiumSellModel: Optimizes for premium selling success (win rate × premium)
- PremiumBuyModel: Optimizes for breakout trade expectancy
"""

from .base import BaseStrategyModel, StrategyObjective
from .directional_model import DirectionalModel
from .premium_sell_model import PremiumSellModel
from .premium_buy_model import PremiumBuyModel

# Strategy class to model mapping
STRATEGY_MODELS = {
    'all': DirectionalModel,  # Legacy behavior
    'directional': DirectionalModel,
    'premium_sell': PremiumSellModel,
    'premium_buy': PremiumBuyModel,
}

# Action to strategy class mapping
ACTION_TO_STRATEGY = {
    'buyShares': 'directional',
    'sellShares': 'directional',
    'openCSP': 'premium_sell',
    'openCC': 'premium_sell',
    'buyCall': 'premium_buy',
    'buyPut': 'premium_buy',
}


def get_strategy_model(strategy_class: str) -> BaseStrategyModel:
    """Get the appropriate strategy model for a given strategy class."""
    model_class = STRATEGY_MODELS.get(strategy_class, DirectionalModel)
    return model_class()


def get_strategy_for_action(action: str) -> str:
    """Get the strategy class for a given action."""
    return ACTION_TO_STRATEGY.get(action, 'directional')


__all__ = [
    'BaseStrategyModel',
    'StrategyObjective',
    'DirectionalModel',
    'PremiumSellModel',
    'PremiumBuyModel',
    'STRATEGY_MODELS',
    'ACTION_TO_STRATEGY',
    'get_strategy_model',
    'get_strategy_for_action',
]
