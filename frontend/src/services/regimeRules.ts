/**
 * Regime Rules Module
 *
 * Fixed rules for each market regime that provide baseline action scores
 * independent of WFO calibration. These rules are based on established
 * trading wisdom for each market condition.
 */

import type { ActionType } from '../types';
import type { MarketRegime } from '../types/calibration';

/**
 * Actions that should be BLOCKED (hard filter) in certain regimes.
 * These match the backend's behavior during WFO optimization.
 * When backend zeros out signals, frontend should block the action entirely.
 */
export const BLOCKED_ACTIONS: Partial<Record<MarketRegime, ActionType[]>> = {
  // In crash mode, backend zeros out ALL long signals
  // Mean reversion indicators are disabled - catching falling knives kills
  BEAR_VOLATILE: ['buyShares', 'openCSP', 'buyCall'],
};

/**
 * Check if an action is blocked by the current market regime.
 * This implements a hard filter matching the backend's WFO behavior.
 */
export function isActionBlockedByRegime(
  regime: MarketRegime | null | undefined,
  action: ActionType
): boolean {
  if (!regime) return false;
  const blockedActions = BLOCKED_ACTIONS[regime];
  return blockedActions?.includes(action) ?? false;
}

/**
 * Fixed action scores per regime.
 * These are baseline recommendations that don't change with calibration.
 * Score range: -1.0 (strongly avoid) to +1.0 (strongly favor)
 */
export const REGIME_ACTION_SCORES: Record<MarketRegime, Record<ActionType, number>> = {
  // Steady uptrend with low volatility - favor trend following
  BULL_QUIET: {
    buyShares: 0.8,    // Trend following works well
    sellShares: -0.5,  // Don't fight the trend
    openCSP: 0.6,      // Good for collecting premium in up market
    openCC: 0.3,       // May cap upside in strong trend
    buyCall: 0.9,      // Ideal for directional plays
    buyPut: -0.8,      // Fighting the trend
  },

  // Uptrend with shakeouts - favor premium selling on dips
  BULL_VOLATILE: {
    buyShares: 0.5,    // OK but wait for dips
    sellShares: -0.3,  // Still bullish overall
    openCSP: 0.9,      // Premium is elevated, dips are buyable
    openCC: 0.7,       // Good premium, less assignment risk
    buyCall: 0.4,      // Expensive due to IV
    buyPut: -0.5,      // Only for hedging
  },

  // Slow bleed downtrend - defensive positioning
  BEAR_QUIET: {
    buyShares: -0.3,   // Catching falling knives
    sellShares: 0.6,   // Reduce exposure
    openCSP: 0.2,      // Risky - may get assigned at losses
    openCC: 0.8,       // Generate income, reduce cost basis
    buyCall: -0.5,     // Fighting the trend
    buyPut: 0.4,       // Can work but IV may be low
  },

  // Crash mode - extreme caution required
  BEAR_VOLATILE: {
    buyShares: -0.9,   // Catching falling knives kills
    sellShares: 0.5,   // Reduce exposure if not already
    openCSP: -0.8,     // Extremely dangerous - assignment at worst time
    openCC: 0.3,       // Can work if already holding shares
    buyCall: -0.7,     // Fighting the trend, expensive
    buyPut: 0.9,       // Ideal environment for puts
  },

  // Range-bound, low volatility - mean reversion works
  NEUTRAL_CHOP: {
    buyShares: 0.3,    // Buy support levels
    sellShares: 0.2,   // Sell resistance levels
    openCSP: 0.8,      // Premium selling ideal in range
    openCC: 0.8,       // Premium selling ideal in range
    buyCall: 0.2,      // Limited profit potential
    buyPut: 0.2,       // Limited profit potential
  },

  // Flat but volatile - breakout imminent
  NEUTRAL_VOLATILE: {
    buyShares: 0.0,    // Wait for direction
    sellShares: 0.0,   // Wait for direction
    openCSP: 0.2,      // Risky - could break either way
    openCC: 0.2,       // Risky - could break either way
    buyCall: 0.5,      // Straddle-like opportunity
    buyPut: 0.5,       // Straddle-like opportunity
  },
};

/**
 * Get the regime-based score for an action.
 * Returns a score from -1.0 to +1.0.
 */
export function getRegimeScore(
  regime: MarketRegime | null | undefined,
  action: ActionType
): number {
  if (!regime || !(regime in REGIME_ACTION_SCORES)) {
    return 0; // Neutral if no regime data
  }
  return REGIME_ACTION_SCORES[regime][action] ?? 0;
}

/**
 * Get all regime scores for display purposes.
 */
export function getAllRegimeScores(
  regime: MarketRegime | null | undefined
): Record<ActionType, number> | null {
  if (!regime || !(regime in REGIME_ACTION_SCORES)) {
    return null;
  }
  return REGIME_ACTION_SCORES[regime];
}

/**
 * Get a human-readable description of the regime.
 */
export function getRegimeDescription(regime: MarketRegime | null | undefined): string {
  if (!regime) return 'Unknown';

  const descriptions: Record<MarketRegime, string> = {
    BULL_QUIET: 'Steady Uptrend',
    BULL_VOLATILE: 'Volatile Uptrend',
    BEAR_QUIET: 'Slow Decline',
    BEAR_VOLATILE: 'Crash Mode',
    NEUTRAL_CHOP: 'Range-Bound',
    NEUTRAL_VOLATILE: 'Breakout Pending',
  };

  return descriptions[regime] ?? regime;
}

/**
 * Check if mean reversion strategies are safe in this regime.
 */
export function isMeanReversionSafe(regime: MarketRegime | null | undefined): boolean {
  if (!regime) return false;

  // Only safe in BULL_VOLATILE (dips are buyable) and NEUTRAL_CHOP (range trading)
  return regime === 'BULL_VOLATILE' || regime === 'NEUTRAL_CHOP';
}

/**
 * Check if trend following strategies are preferred in this regime.
 */
export function isTrendFollowingPreferred(regime: MarketRegime | null | undefined): boolean {
  if (!regime) return false;

  // Trend following works in clear trends
  return regime === 'BULL_QUIET' || regime === 'BEAR_QUIET';
}

/**
 * Get recommended strategy type for the regime.
 */
export function getRecommendedStrategy(regime: MarketRegime | null | undefined): string {
  if (!regime) return 'Wait for clarity';

  const strategies: Record<MarketRegime, string> = {
    BULL_QUIET: 'Trend Following / Buy Calls',
    BULL_VOLATILE: 'Sell Puts on Dips',
    BEAR_QUIET: 'Reduce Exposure / Covered Calls',
    BEAR_VOLATILE: 'Cash / Long Puts',
    NEUTRAL_CHOP: 'Premium Selling / Mean Reversion',
    NEUTRAL_VOLATILE: 'Wait for Breakout',
  };

  return strategies[regime] ?? 'Unknown';
}
