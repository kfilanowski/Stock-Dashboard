/**
 * Consensus Scoring Module
 *
 * Provides multi-indicator consensus scoring to improve signal reliability.
 * When multiple indicators agree, the signal is stronger.
 *
 * WEIGHT-AWARE: Now respects WFO calibration weights. Indicators with low
 * calibrated weights are ignored in consensus to avoid overriding WFO's
 * learned signal reliability.
 */

import type { ActionType } from '../types';
import type { TechnicalIndicators } from './technicalIndicators';

/**
 * Indicator signal classification
 */
type SignalDirection = 'bullish' | 'bearish' | 'neutral';

interface IndicatorSignal {
  indicator: string;
  direction: SignalDirection;
  strength: number; // 0-1
  wfoWeight?: number; // WFO calibration weight (0-3)
}

/**
 * Map from consensus indicator keys to backend WFO keys.
 */
const CONSENSUS_TO_WFO_KEY: Record<string, string> = {
  'rsi': 'rsi',
  'macd': 'macd',
  'adx': 'adx',
  'sma': 'sma',
  'bb': 'bollinger',
  'cmf': 'cmf',
  'momentum': 'momentum',
};

/**
 * Weight threshold below which an indicator is ignored in consensus.
 * If WFO learned that an indicator is unreliable (weight < 0.5), consensus
 * should not override that by counting it equally.
 */
const WEIGHT_RELEVANCE_THRESHOLD = 0.5;

/**
 * Extract directional signals from technical indicators.
 * Now accepts optional WFO weights to filter out unreliable indicators.
 */
function extractSignals(
  indicators: TechnicalIndicators,
  wfoWeights?: Record<string, number>
): IndicatorSignal[] {
  const signals: IndicatorSignal[] = [];

  /**
   * Helper to get WFO weight for a consensus indicator.
   */
  function getWfoWeight(indicator: string): number | undefined {
    if (!wfoWeights) return undefined;
    const wfoKey = CONSENSUS_TO_WFO_KEY[indicator];
    return wfoKey ? wfoWeights[wfoKey] : undefined;
  }

  // RSI
  if (indicators.rsi) {
    const rsiValue = indicators.rsi.value;
    const wfoWeight = getWfoWeight('rsi');
    if (rsiValue < 30) {
      signals.push({ indicator: 'rsi', direction: 'bullish', strength: Math.min(1, (30 - rsiValue) / 20), wfoWeight });
    } else if (rsiValue > 70) {
      signals.push({ indicator: 'rsi', direction: 'bearish', strength: Math.min(1, (rsiValue - 70) / 20), wfoWeight });
    } else {
      signals.push({ indicator: 'rsi', direction: 'neutral', strength: 0.3, wfoWeight });
    }
  }

  // MACD
  if (indicators.macd) {
    const { histogram, isBullish } = indicators.macd;
    const wfoWeight = getWfoWeight('macd');
    if (isBullish && histogram > 0) {
      signals.push({ indicator: 'macd', direction: 'bullish', strength: Math.min(1, Math.abs(histogram) * 10), wfoWeight });
    } else if (!isBullish && histogram < 0) {
      signals.push({ indicator: 'macd', direction: 'bearish', strength: Math.min(1, Math.abs(histogram) * 10), wfoWeight });
    } else {
      signals.push({ indicator: 'macd', direction: 'neutral', strength: 0.2, wfoWeight });
    }
  }

  // ADX + DI
  if (indicators.adx) {
    const { adx, plusDI, minusDI, direction, isTrending } = indicators.adx;
    const wfoWeight = getWfoWeight('adx');
    if (isTrending && direction === 'bullish' && plusDI > minusDI) {
      signals.push({ indicator: 'adx', direction: 'bullish', strength: Math.min(1, adx / 50), wfoWeight });
    } else if (isTrending && direction === 'bearish' && minusDI > plusDI) {
      signals.push({ indicator: 'adx', direction: 'bearish', strength: Math.min(1, adx / 50), wfoWeight });
    } else {
      signals.push({ indicator: 'adx', direction: 'neutral', strength: 0.3, wfoWeight });
    }
  }

  // SMA Alignment
  if (indicators.sma50 && indicators.sma200) {
    const sma50 = indicators.sma50;
    const sma200 = indicators.sma200;
    const wfoWeight = getWfoWeight('sma');
    if (sma50 > sma200) {
      const diff = (sma50 - sma200) / sma200;
      signals.push({ indicator: 'sma', direction: 'bullish', strength: Math.min(1, diff * 10), wfoWeight });
    } else {
      const diff = (sma200 - sma50) / sma200;
      signals.push({ indicator: 'sma', direction: 'bearish', strength: Math.min(1, diff * 10), wfoWeight });
    }
  }

  // Bollinger Bands
  if (indicators.bollingerBands) {
    const { percentB } = indicators.bollingerBands;
    const wfoWeight = getWfoWeight('bb');
    if (percentB < 0.2) {
      signals.push({ indicator: 'bb', direction: 'bullish', strength: Math.min(1, (0.2 - percentB) * 2), wfoWeight });
    } else if (percentB > 0.8) {
      signals.push({ indicator: 'bb', direction: 'bearish', strength: Math.min(1, (percentB - 0.8) * 2), wfoWeight });
    } else {
      signals.push({ indicator: 'bb', direction: 'neutral', strength: 0.3, wfoWeight });
    }
  }

  // CMF (Chaikin Money Flow)
  if (indicators.cmf) {
    const cmfValue = indicators.cmf.value;
    const wfoWeight = getWfoWeight('cmf');
    if (cmfValue > 0.1) {
      signals.push({ indicator: 'cmf', direction: 'bullish', strength: Math.min(1, cmfValue * 2), wfoWeight });
    } else if (cmfValue < -0.1) {
      signals.push({ indicator: 'cmf', direction: 'bearish', strength: Math.min(1, Math.abs(cmfValue) * 2), wfoWeight });
    } else {
      signals.push({ indicator: 'cmf', direction: 'neutral', strength: 0.2, wfoWeight });
    }
  }

  // Momentum
  if (indicators.momentum) {
    const { shortTerm, trend } = indicators.momentum;
    const wfoWeight = getWfoWeight('momentum');
    if (trend === 'bullish' && shortTerm > 2) {
      signals.push({ indicator: 'momentum', direction: 'bullish', strength: Math.min(1, shortTerm / 10), wfoWeight });
    } else if (trend === 'bearish' && shortTerm < -2) {
      signals.push({ indicator: 'momentum', direction: 'bearish', strength: Math.min(1, Math.abs(shortTerm) / 10), wfoWeight });
    } else {
      signals.push({ indicator: 'momentum', direction: 'neutral', strength: 0.2, wfoWeight });
    }
  }

  return signals;
}

/**
 * Calculate consensus score based on indicator agreement.
 *
 * When multiple indicators agree on direction, confidence increases.
 * When indicators disagree, confidence decreases.
 *
 * WEIGHT-AWARE: If WFO weights are provided, indicators with low calibrated
 * weights (< WEIGHT_RELEVANCE_THRESHOLD) are excluded from consensus. This
 * prevents consensus from overriding WFO's learned indicator reliability.
 *
 * Returns a score from -1.0 (strong bearish consensus) to +1.0 (strong bullish consensus)
 */
export function calculateConsensusScore(
  indicators: TechnicalIndicators,
  action: ActionType,
  wfoWeights?: Record<string, number>
): number {
  const allSignals = extractSignals(indicators, wfoWeights);

  // Filter to only include signals from indicators with significant WFO weights
  // If no WFO weights provided, include all signals (backward compatible)
  const signals = wfoWeights
    ? allSignals.filter(s =>
        s.wfoWeight === undefined || s.wfoWeight >= WEIGHT_RELEVANCE_THRESHOLD
      )
    : allSignals;

  if (signals.length === 0) {
    return 0; // No data
  }

  // Count signals by direction, weighted by both signal strength AND WFO weight
  let bullishScore = 0;
  let bearishScore = 0;
  let totalWeight = 0;

  for (const signal of signals) {
    // Combine signal strength with WFO weight (default to 1.0 if not provided)
    const wfoMultiplier = signal.wfoWeight !== undefined ? signal.wfoWeight : 1.0;
    const weight = signal.strength * wfoMultiplier;
    totalWeight += weight;

    if (signal.direction === 'bullish') {
      bullishScore += weight;
    } else if (signal.direction === 'bearish') {
      bearishScore += weight;
    }
  }

  if (totalWeight === 0) {
    return 0;
  }

  // Calculate net direction (-1 to +1)
  const netDirection = (bullishScore - bearishScore) / totalWeight;

  // Calculate agreement level (0 to 1)
  // High agreement = most indicators point the same way
  const dominantScore = Math.max(bullishScore, bearishScore);
  const agreement = dominantScore / totalWeight;

  // Consensus score = direction * agreement
  // Strong agreement amplifies the signal, weak agreement dampens it
  let consensusScore = netDirection * agreement;

  // Flip for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) {
    consensusScore = -consensusScore;
  }

  return Math.max(-1, Math.min(1, consensusScore));
}

/**
 * Get a breakdown of consensus signals for debugging/display.
 */
export function getConsensusBreakdown(
  indicators: TechnicalIndicators
): {
  signals: IndicatorSignal[];
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  agreement: number;
} {
  const signals = extractSignals(indicators);

  const bullishCount = signals.filter((s) => s.direction === 'bullish').length;
  const bearishCount = signals.filter((s) => s.direction === 'bearish').length;
  const neutralCount = signals.filter((s) => s.direction === 'neutral').length;

  const total = signals.length || 1;
  const dominant = Math.max(bullishCount, bearishCount, neutralCount);
  const agreement = dominant / total;

  return {
    signals,
    bullishCount,
    bearishCount,
    neutralCount,
    agreement,
  };
}

/**
 * Check if there's strong consensus (most indicators agree).
 */
export function hasStrongConsensus(indicators: TechnicalIndicators): boolean {
  const { agreement, neutralCount, signals } = getConsensusBreakdown(indicators);

  // Need at least 4 signals and 70%+ agreement (excluding neutral)
  const nonNeutral = signals.length - neutralCount;
  return nonNeutral >= 4 && agreement >= 0.7;
}
