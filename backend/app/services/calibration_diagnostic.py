"""
Calibration Diagnostic Tool

Establishes ground truth by running controlled experiments:
1. Buy-and-hold baseline (what's the market doing?)
2. Random signal baseline (are our signals better than random?)
3. Gross vs Net comparison (are costs killing us?)
4. Sample trades inspection (are prices realistic?)

Usage:
    from .calibration_diagnostic import run_diagnostic
    report = await run_diagnostic(db, "AAPL", horizon=5)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

from .wfo_simulator import (
    fast_simulate, calculate_next_day_open_returns,
    MIN_TRANSACTION_COST, MIN_SLIPPAGE, get_adaptive_slippage,
    calculate_atr_series
)
from .indicators import calculate_indicator_signals
from .calibration_service import load_price_history, DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report."""
    ticker: str
    horizon: int
    data_points: int
    date_range: str

    # Buy and hold baseline
    buy_hold_return: float
    buy_hold_annualized: float

    # Strategy results
    strategy_trades: int
    strategy_gross_sqn: float  # Before costs
    strategy_net_sqn: float    # After costs
    strategy_expectancy_gross: float
    strategy_expectancy_net: float
    strategy_win_rate: float

    # Cost analysis
    avg_transaction_cost: float
    avg_slippage: float
    avg_total_cost_per_trade: float
    cost_drag_on_sqn: float

    # Signal quality
    long_signals_count: int
    short_signals_count: int
    avg_long_return: float  # Before costs
    avg_short_return: float

    # Random baseline (for comparison)
    random_sqn: float
    random_expectancy: float

    # Sample trades (for inspection)
    sample_trades: list

    # Verdict
    verdict: str
    recommendations: list


async def run_diagnostic(
    db,
    ticker: str,
    horizon: int = 5,
    weights: Optional[Dict[str, float]] = None
) -> DiagnosticReport:
    """
    Run comprehensive diagnostic to establish ground truth.

    Args:
        db: Database session
        ticker: Stock ticker
        horizon: Trading horizon (days)
        weights: Weights to test (uses defaults if None)

    Returns:
        DiagnosticReport with all findings
    """
    ticker = ticker.upper()
    weights = weights or DEFAULT_WEIGHTS.copy()

    logger.info(f"[DIAGNOSTIC] Starting diagnostic for {ticker}, horizon={horizon}")

    # 1. Load data
    df = await load_price_history(db, ticker, min_days=500)
    n = len(df)

    if n < 200:
        raise ValueError(f"Insufficient data: {n} days (need 200+)")

    date_range = f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}"
    logger.info(f"[DIAGNOSTIC] Loaded {n} days: {date_range}")

    # 2. Buy-and-hold baseline
    first_price = df['close'].iloc[0]
    last_price = df['close'].iloc[-1]
    buy_hold_return = (last_price / first_price - 1) * 100
    years = n / 252
    buy_hold_annualized = ((1 + buy_hold_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

    logger.info(f"[DIAGNOSTIC] Buy & Hold: {buy_hold_return:.1f}% total, {buy_hold_annualized:.1f}% annualized")

    # 3. Run strategy simulation
    result = fast_simulate(
        df, weights, horizon=horizon,
        transaction_cost=MIN_TRANSACTION_COST,
        apply_regime_filter=True
    )

    # 4. Calculate GROSS metrics (before costs)
    if result.trades is not None and len(result.trades) > 0:
        trades_df = result.trades

        # Gross returns (add back costs)
        gross_returns = trades_df['net_return'] + trades_df['costs']
        gross_expectancy = gross_returns.mean() * 100
        gross_std = gross_returns.std()
        n_trades = len(trades_df)
        gross_sqn = np.sqrt(n_trades) * (gross_returns.mean() / gross_std) if gross_std > 0 and n_trades >= 20 else 0

        # Cost analysis
        avg_cost = trades_df['costs'].mean() * 100
        avg_slip = trades_df['slippage_applied'].mean() if 'slippage_applied' in trades_df.columns else 0

        # Signal analysis
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']

        # Use forward_return for gross analysis
        avg_long_return = long_trades['forward_return'].mean() * 100 if len(long_trades) > 0 else 0
        avg_short_return = -short_trades['forward_return'].mean() * 100 if len(short_trades) > 0 else 0

        # Sample trades for inspection
        sample_trades = trades_df.head(10).to_dict('records')

    else:
        gross_sqn = 0
        gross_expectancy = 0
        avg_cost = 0
        avg_slip = 0
        avg_long_return = 0
        avg_short_return = 0
        n_trades = 0
        sample_trades = []

    # 5. Random baseline - generate random signals and compare
    random_sqn, random_exp = _run_random_baseline(df, horizon, n_trades)

    # 6. Calculate cost drag
    cost_drag = gross_sqn - result.sqn if gross_sqn > 0 else 0

    # 7. Generate verdict and recommendations
    verdict, recommendations = _generate_verdict(
        buy_hold_annualized=buy_hold_annualized,
        gross_sqn=gross_sqn,
        net_sqn=result.sqn,
        gross_expectancy=gross_expectancy,
        net_expectancy=result.expectancy,
        n_trades=n_trades,
        random_sqn=random_sqn,
        avg_cost=avg_cost,
        avg_long_return=avg_long_return,
        avg_short_return=avg_short_return,
        win_rate=result.win_rate
    )

    return DiagnosticReport(
        ticker=ticker,
        horizon=horizon,
        data_points=n,
        date_range=date_range,

        buy_hold_return=round(buy_hold_return, 2),
        buy_hold_annualized=round(buy_hold_annualized, 2),

        strategy_trades=n_trades,
        strategy_gross_sqn=round(gross_sqn, 3),
        strategy_net_sqn=round(result.sqn, 3),
        strategy_expectancy_gross=round(gross_expectancy, 3),
        strategy_expectancy_net=round(result.expectancy, 3),
        strategy_win_rate=round(result.win_rate, 1),

        avg_transaction_cost=round(MIN_TRANSACTION_COST * 100, 2),
        avg_slippage=round(avg_slip, 2),
        avg_total_cost_per_trade=round(avg_cost, 3),
        cost_drag_on_sqn=round(cost_drag, 3),

        long_signals_count=len(long_trades) if 'long_trades' in dir() else 0,
        short_signals_count=len(short_trades) if 'short_trades' in dir() else 0,
        avg_long_return=round(avg_long_return, 3),
        avg_short_return=round(avg_short_return, 3),

        random_sqn=round(random_sqn, 3),
        random_expectancy=round(random_exp, 3),

        sample_trades=sample_trades,

        verdict=verdict,
        recommendations=recommendations
    )


def _run_random_baseline(df: pd.DataFrame, horizon: int, target_trades: int) -> tuple:
    """
    Run random signal baseline for comparison.

    Generates random entry signals with same frequency as strategy
    to see if our signals are better than random.
    """
    if target_trades < 10:
        return 0.0, 0.0

    n = len(df)
    forward_returns, _, atr_series = calculate_next_day_open_returns(df, horizon)

    # Generate random signals with similar frequency
    signal_prob = min(0.1, target_trades / n)  # Cap at 10% of days

    np.random.seed(42)  # Reproducible
    random_signals = np.random.random(n) < signal_prob

    # Get returns for random signals
    random_returns = []
    for i in np.where(random_signals)[0]:
        if i < len(forward_returns) and not np.isnan(forward_returns.iloc[i]):
            # Random long/short
            direction = np.random.choice([1, -1])
            ret = direction * forward_returns.iloc[i]

            # Apply costs
            atr = atr_series.iloc[i] if i < len(atr_series) else 0.01
            slip = get_adaptive_slippage(atr)
            cost = 2 * (MIN_TRANSACTION_COST + slip)

            random_returns.append(ret - cost)

    if len(random_returns) < 10:
        return 0.0, 0.0

    random_returns = np.array(random_returns)
    expectancy = np.mean(random_returns) * 100
    std = np.std(random_returns)
    sqn = np.sqrt(len(random_returns)) * (np.mean(random_returns) / std) if std > 0 else 0

    return sqn, expectancy


def _generate_verdict(
    buy_hold_annualized: float,
    gross_sqn: float,
    net_sqn: float,
    gross_expectancy: float,
    net_expectancy: float,
    n_trades: int,
    random_sqn: float,
    avg_cost: float,
    avg_long_return: float,
    avg_short_return: float,
    win_rate: float
) -> tuple:
    """Generate human-readable verdict and recommendations."""

    recommendations = []

    # Check signal predictiveness
    signals_predictive = gross_sqn > random_sqn + 0.5
    costs_killing = gross_sqn > 1.5 and net_sqn < 1.0
    insufficient_trades = n_trades < 20

    if insufficient_trades:
        verdict = "INSUFFICIENT DATA"
        recommendations.append(f"Only {n_trades} trades generated. Need 20+ for statistical significance.")
        recommendations.append("Try a lower threshold or longer date range.")

    elif gross_sqn < 0.5:
        verdict = "SIGNALS NOT PREDICTIVE"
        recommendations.append(f"Gross SQN ({gross_sqn:.2f}) is very low - signals aren't predicting returns.")
        recommendations.append("The indicators may not work for this stock's price behavior.")
        if avg_long_return < 0:
            recommendations.append(f"Long signals avg return: {avg_long_return:.2f}% (negative = wrong direction)")
        if avg_short_return < 0:
            recommendations.append(f"Short signals avg return: {avg_short_return:.2f}% (negative = wrong direction)")

    elif gross_sqn > random_sqn + 0.5 and costs_killing:
        verdict = "COSTS KILLING EDGE"
        recommendations.append(f"Gross SQN ({gross_sqn:.2f}) shows edge, but costs reduce to {net_sqn:.2f}")
        recommendations.append(f"Avg cost per trade: {avg_cost:.2f}% is eating the {gross_expectancy:.2f}% gross expectancy")
        recommendations.append("Consider: longer horizons (bigger moves), or lower cost threshold for high-vol stocks")

    elif net_sqn >= 2.0:
        verdict = "STRATEGY WORKS"
        recommendations.append(f"Net SQN of {net_sqn:.2f} indicates a tradeable edge.")
        recommendations.append(f"Win rate: {win_rate:.1f}%, Expectancy: {net_expectancy:.2f}% per trade")

    elif net_sqn >= 1.0:
        verdict = "MARGINAL EDGE"
        recommendations.append(f"Net SQN of {net_sqn:.2f} shows weak but positive edge.")
        recommendations.append("Consider if this is worth trading given real-world slippage.")

    elif not signals_predictive:
        verdict = "NO BETTER THAN RANDOM"
        recommendations.append(f"Strategy SQN ({net_sqn:.2f}) similar to random ({random_sqn:.2f})")
        recommendations.append("The indicators aren't providing predictive value for this stock.")

    else:
        verdict = "UNPROFITABLE"
        recommendations.append(f"Negative expectancy ({net_expectancy:.2f}%) - strategy loses money.")

    # Add buy-and-hold comparison
    if buy_hold_annualized > 15:
        recommendations.append(f"Note: Buy & hold returned {buy_hold_annualized:.1f}%/year. "
                             f"Strong uptrend may make timing unnecessary.")
    elif buy_hold_annualized < -10:
        recommendations.append(f"Note: Buy & hold lost {buy_hold_annualized:.1f}%/year. "
                             f"Bear market conditions make long signals harder.")

    return verdict, recommendations


def diagnostic_to_dict(report: DiagnosticReport) -> Dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    return {
        'ticker': report.ticker,
        'horizon': report.horizon,
        'data_points': report.data_points,
        'date_range': report.date_range,

        'baseline': {
            'buy_hold_return': report.buy_hold_return,
            'buy_hold_annualized': report.buy_hold_annualized,
        },

        'strategy': {
            'trades': report.strategy_trades,
            'gross_sqn': report.strategy_gross_sqn,
            'net_sqn': report.strategy_net_sqn,
            'expectancy_gross': report.strategy_expectancy_gross,
            'expectancy_net': report.strategy_expectancy_net,
            'win_rate': report.strategy_win_rate,
        },

        'costs': {
            'transaction_cost_pct': report.avg_transaction_cost,
            'avg_slippage_pct': report.avg_slippage,
            'total_cost_per_trade_pct': report.avg_total_cost_per_trade,
            'sqn_drag': report.cost_drag_on_sqn,
        },

        'signals': {
            'long_count': report.long_signals_count,
            'short_count': report.short_signals_count,
            'avg_long_return_pct': report.avg_long_return,
            'avg_short_return_pct': report.avg_short_return,
        },

        'random_baseline': {
            'sqn': report.random_sqn,
            'expectancy': report.random_expectancy,
        },

        'sample_trades': report.sample_trades[:5],  # Limit for JSON

        'verdict': report.verdict,
        'recommendations': report.recommendations,
    }
