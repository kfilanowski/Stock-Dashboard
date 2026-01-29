/**
 * Stock Scoring Service
 * 
 * Calculates scores for various trading actions based on technical indicators.
 * Each indicator produces a signal (-1 to +1) that is weighted per action.
 */

import type { 
  HistoryPoint, 
  ActionType, 
  MetricType, 
  MetricSignal, 
  ActionScore, 
  StockAnalysis
} from '../types';
import {
  calculateAllIndicators,
  type TechnicalIndicators
} from './technicalIndicators';
import { getRegimeScore, isActionBlockedByRegime } from './regimeRules';
import { calculateConsensusScore } from './consensusScoring';
import type { MarketRegime } from '../types/calibration';

// ============================================================================
// Types
// ============================================================================

export interface AdditionalAnalysisData {
  hasOptions?: boolean;
  alpha?: number | null;
  daysToEarnings?: number | null;
  betaToSector?: number | null;
  sectorCorrelation?: number | null;
  roic?: number | null;
  callPutRatio?: number | null;
  ivPercentile?: number | null;
  roe?: number | null;
  sector?: string | null;
  industry?: string | null;
  beta?: number | null;
  callPutRatioOI?: number | null;
  callPutRatioVolume?: number | null;
  optionsSentiment?: string | null;
  avgImpliedVolatility?: number | null;
  marketRegime?: MarketRegime | null;
  // Relative strength indicators (vs sector)
  relMomentum?: number | null;
  rsRatio?: number | null;
  rsMomentum?: number | null;
}

// ============================================================================
// Constants
// ============================================================================

export const ALL_ACTIONS: ActionType[] = [
  'buyShares', 'sellShares', 'openCSP', 'openCC', 'buyCall', 'buyPut'
];

export const ALL_METRICS: MetricType[] = [
  'rsi', 'macd', 'bollingerBands', 'bollingerSqueeze', 'vwap', 'momentum',
  'volume', 'pricePosition', 'smaAlignment', 'rvol', 'adx', 'crossPattern',
  'cmf', 'divergence', 'alpha',
  'roic', 'callPutRatio', 'ivPercentile', 'sectorBeta', 'earningsProximity',
  'optionsSentiment', 'relMomentum', 'rsRatio',
  'avwap', 'obv', 'chandelier'
];

// ============================================================================
// Strategy Class Configuration
// ============================================================================

/**
 * Strategy classes for WFO optimization.
 * Each strategy class has a different objective function:
 * - 'directional': Optimized for price direction (SQN-based)
 * - 'premium_sell': Optimized for premium selling (win rate × premium)
 * - 'premium_buy': Optimized for breakout trades (expectancy)
 */
export type StrategyClass = 'all' | 'directional' | 'premium_sell' | 'premium_buy';

/**
 * Map actions to their strategy class.
 * Used to load the correct calibrated weights for each action type.
 */
export const ACTION_TO_STRATEGY: Record<ActionType, StrategyClass> = {
  buyShares: 'directional',
  sellShares: 'directional',
  openCSP: 'premium_sell',
  openCC: 'premium_sell',
  buyCall: 'premium_buy',
  buyPut: 'premium_buy',
};

/**
 * Get the strategy class for a given action.
 */
export function getStrategyForAction(action: ActionType): StrategyClass {
  return ACTION_TO_STRATEGY[action] || 'directional';
}

// ============================================================================
// Weight Matrix Configuration
// ============================================================================

/**
 * Weight matrix: [metric][action] = weight
 */
type WeightMatrix = Record<MetricType, Record<ActionType, number>>;

/**
 * DEFAULT WEIGHTS (Base)
 * Used if no horizon-specific or WFO weights are available.
 */
const BASE_WEIGHTS: WeightMatrix = {
  rsi: { buyShares: 0.8, sellShares: 1.0, openCSP: 1.8, openCC: 1.5, buyCall: 0.3, buyPut: 0.3 },
  macd: { buyShares: 1.2, sellShares: 1.2, openCSP: 0.6, openCC: 0.6, buyCall: 1.8, buyPut: 1.8 },
  bollingerBands: { buyShares: 0.8, sellShares: 0.8, openCSP: 1.5, openCC: 1.5, buyCall: 0.4, buyPut: 0.4 },
  bollingerSqueeze: { buyShares: 0.3, sellShares: 0.3, openCSP: 1.2, openCC: 1.2, buyCall: 2.0, buyPut: 2.0 },
  adx: { buyShares: 0.8, sellShares: 0.8, openCSP: 1.5, openCC: 1.5, buyCall: 2.2, buyPut: 2.2 },
  vwap: { buyShares: 0.8, sellShares: 0.8, openCSP: 1.0, openCC: 1.0, buyCall: 0.6, buyPut: 0.6 },
  momentum: { buyShares: 0.6, sellShares: 0.6, openCSP: 0.3, openCC: 0.3, buyCall: 1.0, buyPut: 1.0 },
  volume: { buyShares: 1.0, sellShares: 1.0, openCSP: 0.6, openCC: 0.6, buyCall: 1.8, buyPut: 1.8 },
  rvol: { buyShares: 0.8, sellShares: 0.8, openCSP: 0.5, openCC: 0.5, buyCall: 1.5, buyPut: 1.5 },
  pricePosition: { buyShares: 1.0, sellShares: 1.0, openCSP: 1.5, openCC: 1.5, buyCall: 0.4, buyPut: 0.4 },
  smaAlignment: { buyShares: 1.8, sellShares: 1.5, openCSP: 0.8, openCC: 0.8, buyCall: 1.0, buyPut: 1.0 },
  crossPattern: { buyShares: 1.5, sellShares: 1.5, openCSP: 0.6, openCC: 0.6, buyCall: 0.8, buyPut: 0.8 },
  roic: { buyShares: 1.5, sellShares: 0.5, openCSP: 1.2, openCC: 0.6, buyCall: 0.2, buyPut: 0.2 },
  callPutRatio: { buyShares: 0.6, sellShares: 0.6, openCSP: 1.0, openCC: 1.0, buyCall: 1.2, buyPut: 1.2 },
  ivPercentile: { buyShares: 0.2, sellShares: 0.2, openCSP: 2.0, openCC: 2.0, buyCall: 1.8, buyPut: 1.8 },
  sectorBeta: { buyShares: 1.0, sellShares: 0.8, openCSP: 0.6, openCC: 0.6, buyCall: 0.5, buyPut: 0.5 },
  cmf: { buyShares: 1.8, sellShares: 1.5, openCSP: 1.2, openCC: 1.0, buyCall: 2.0, buyPut: 2.0 },
  divergence: { buyShares: 1.5, sellShares: 1.5, openCSP: 1.0, openCC: 1.0, buyCall: 2.2, buyPut: 2.2 },
  alpha: { buyShares: 1.5, sellShares: 1.2, openCSP: 0.8, openCC: 0.8, buyCall: 1.8, buyPut: 1.8 },
  earningsProximity: { buyShares: 0.8, sellShares: 0.5, openCSP: 1.5, openCC: 1.5, buyCall: 2.0, buyPut: 2.0 },
  optionsSentiment: { buyShares: 0.8, sellShares: 0.8, openCSP: 1.2, openCC: 1.2, buyCall: 1.5, buyPut: 1.5 },
  // Relative strength indicators (vs sector)
  relMomentum: { buyShares: 1.2, sellShares: 1.0, openCSP: 0.8, openCC: 0.8, buyCall: 1.5, buyPut: 1.5 },
  rsRatio: { buyShares: 1.0, sellShares: 0.8, openCSP: 0.6, openCC: 0.6, buyCall: 1.2, buyPut: 1.2 },
  // Volume/trend health indicators (from VolatilityGuidance)
  avwap: { buyShares: 1.2, sellShares: 1.0, openCSP: 1.4, openCC: 1.2, buyCall: 0.8, buyPut: 0.8 },
  obv: { buyShares: 1.5, sellShares: 1.5, openCSP: 1.2, openCC: 1.0, buyCall: 1.8, buyPut: 1.8 },
  chandelier: { buyShares: 1.0, sellShares: 1.8, openCSP: 0.8, openCC: 1.5, buyCall: 0.6, buyPut: 1.2 }
};

/**
 * HORIZON-SPECIFIC ADJUSTMENTS (additive offsets)
 * Applied to BASE_WEIGHTS when no WFO calibration is available.
 * Additive approach prevents extreme weight ratios from multiplicative stacking.
 */
const SWING_ADJUSTMENTS: Record<string, number> = {
  momentum: 0.5,      // Was 1.5x, now +0.5
  rsi: 0.2,           // Was 1.2x, now +0.2
  vwap: 0.2,          // Was 1.2x, now +0.2
  bollingerBands: 0.3, // Was 1.3x, now +0.3
  smaAlignment: -0.2, // Was 0.8x, now -0.2 (less important for swing)
  roic: -0.5          // Was 0.5x, now -0.5 (fundamental less important for short term)
};

const TREND_ADJUSTMENTS: Record<string, number> = {
  smaAlignment: 0.5,  // Was 1.5x, now +0.5
  adx: 0.5,           // Was 1.5x, now +0.5
  macd: 0.3,          // Was 1.3x, now +0.3
  roic: 0.2,          // Was 1.2x, now +0.2
  momentum: -0.3,     // Was 0.7x, now -0.3 (noise for trend)
  vwap: -0.2          // Was 0.8x, now -0.2
};

// Weight clamp range to prevent extreme values
const WEIGHT_MIN = 0.1;
const WEIGHT_MAX = 3.0;

// Action labels for display
const actionLabels: Record<ActionType, string> = {
  buyShares: 'Buy Shares',
  sellShares: 'Sell Shares',
  openCSP: 'Open CSP',
  openCC: 'Open CC',
  buyCall: 'Buy Call',
  buyPut: 'Buy Put'
};

// Metric labels for display
const metricLabels: Record<MetricType, string> = {
  rsi: 'RSI',
  macd: 'MACD',
  bollingerBands: 'Bollinger Bands',
  bollingerSqueeze: 'Vol. Squeeze',
  vwap: 'VWAP',
  momentum: 'Momentum',
  volume: 'Volume',
  pricePosition: 'Price Position',
  smaAlignment: 'SMA Alignment',
  rvol: 'Rel. Volume',
  adx: 'ADX (Trend)',
  crossPattern: 'Cross Pattern',
  cmf: 'Money Flow',
  divergence: 'Divergence',
  alpha: 'Alpha (vs SPY)',
  roic: 'ROIC',
  callPutRatio: 'Call/Put Ratio',
  ivPercentile: 'IV Percentile',
  sectorBeta: 'Sector Beta',
  earningsProximity: 'Earnings',
  optionsSentiment: 'Options Flow',
  relMomentum: 'Rel. Momentum',
  rsRatio: 'RS Ratio',
  avwap: 'AVWAP',
  obv: 'OBV',
  chandelier: 'Trend Stop'
};

// ============================================================================
// Signal Interpretation Functions
// ============================================================================

interface SignalResult {
  signal: number;        // -1 to +1
  rawValue: string;      // Human-readable
  reasoning: string;     // Explanation
}

/**
 * Interpret RSI using backend-consistent generic signal.
 *
 * SIMPLIFIED: Removed action-specific logic to match backend WFO optimization.
 * The backend uses a generic RSI signal for all actions, so the frontend must too.
 * Signal flipping for bearish actions is preserved (this is correct behavior).
 */
function interpretRSI(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const rsi = indicators.rsi;
  if (!rsi || rsi.value === undefined || rsi.value === null) return null;

  const value = rsi.value;
  const adx = indicators.adx;
  const isTrending = adx?.isTrending ?? false;
  const trendDirection = adx?.direction ?? 'neutral';

  let signal = 0;
  let reasoning = '';

  // Generic RSI signal matching backend's calculate_indicator_signals()
  // Backend: base_rsi scaled from +1 (RSI=0) to -1 (RSI=100) with breakpoints at 30 and 70
  const rsi_oversold = 30;
  const rsi_overbought = 70;
  const rsi_mid = 50;

  if (value <= rsi_oversold) {
    // RSI 0-30: signal 1.0 to 0.5
    signal = 1.0 - (value / rsi_oversold) * 0.5;
    reasoning = `Oversold (${value.toFixed(0)})`;
  } else if (value <= rsi_mid) {
    // RSI 30-50: signal 0.5 to 0
    signal = 0.5 - ((value - rsi_oversold) / (rsi_mid - rsi_oversold)) * 0.5;
    reasoning = `Neutral-bullish (${value.toFixed(0)})`;
  } else if (value <= rsi_overbought) {
    // RSI 50-70: signal 0 to -0.5
    signal = -((value - rsi_mid) / (rsi_overbought - rsi_mid)) * 0.5;
    reasoning = `Neutral-bearish (${value.toFixed(0)})`;
  } else {
    // RSI 70-100: signal -0.5 to -1.0
    signal = -0.5 - ((value - rsi_overbought) / (100 - rsi_overbought)) * 0.5;
    reasoning = `Overbought (${value.toFixed(0)})`;
  }

  // Regime override: trending bullish market (matches backend line 824-825)
  if (value >= 65 && value < 85 && isTrending && trendDirection === 'bullish') {
    signal = 0.8;
    reasoning = `Trending bullish (${value.toFixed(0)}, ADX ${adx?.adx?.toFixed(0) ?? '?'})`;
  }

  // Flip signal for bearish actions (sellShares, buyPut)
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `RSI: ${value.toFixed(1)}`, reasoning };
}

function interpretMACD(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const macd = indicators.macd;
  if (!macd) return null;
  
  let signal = 0;
  let reasoning = '';
  const histogramStrength = Math.min(Math.abs(macd.histogram) / 2, 1);
  
  if (macd.isBullish) {
    signal = 0.3 + histogramStrength * 0.7;
    reasoning = macd.histogram > 0 ? `Bullish crossover, histogram +${macd.histogram.toFixed(2)}` : `Above signal, converging`;
  } else {
    signal = -(0.3 + histogramStrength * 0.7);
    reasoning = macd.histogram < 0 ? `Bearish crossover, histogram ${macd.histogram.toFixed(2)}` : `Below signal, converging`;
  }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `MACD: ${macd.macdLine.toFixed(2)}`, reasoning };
}

function interpretBollingerBands(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const bb = indicators.bollingerBands;
  if (!bb) return null;
  
  let signal = 0;
  let reasoning = '';
  const percentB = bb.percentB;
  
  if (percentB <= 0) { signal = 1; reasoning = 'Below lower band (oversold)'; }
  else if (percentB <= 0.2) { signal = 0.8; reasoning = 'Near lower band'; }
  else if (percentB <= 0.4) { signal = 0.4; reasoning = 'Lower half of bands'; }
  else if (percentB <= 0.6) { signal = 0; reasoning = 'Middle of bands'; }
  else if (percentB <= 0.8) { signal = -0.4; reasoning = 'Upper half of bands'; }
  else if (percentB < 1) { signal = -0.8; reasoning = 'Near upper band'; }
  else { signal = -1; reasoning = 'Above upper band (overbought)'; }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal, rawValue: `%B: ${(percentB * 100).toFixed(0)}%`, reasoning };
}

function interpretVWAP(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const vwap = indicators.vwap;
  if (!vwap) return null;
  
  let signal = 0;
  let reasoning = '';
  const deviation = vwap.priceVsVwap;
  const vwapType = vwap.isIntraday ? '' : ' (rolling)';
  
  if (deviation < -3) { signal = 0.8; reasoning = `${Math.abs(deviation).toFixed(1)}% below VWAP${vwapType}`; }
  else if (deviation < -1) { signal = 0.4; reasoning = `Slightly below VWAP${vwapType}`; }
  else if (deviation <= 1) { signal = 0; reasoning = `At VWAP${vwapType}`; }
  else if (deviation <= 3) { signal = -0.4; reasoning = `Slightly above VWAP${vwapType}`; }
  else { signal = -0.8; reasoning = `${deviation.toFixed(1)}% above VWAP${vwapType}`; }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal, rawValue: `VWAP: ${vwap.value.toFixed(2)}`, reasoning };
}

function interpretMomentum(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const momentum = indicators.momentum;
  if (!momentum) return null;
  
  let signal = 0;
  let reasoning = '';
  const avgMomentum = (momentum.shortTerm + momentum.mediumTerm) / 2;
  
  if (momentum.trend === 'bullish') {
    signal = Math.min(avgMomentum / 10, 1) * 0.8;
    reasoning = `Bullish trend (+${avgMomentum.toFixed(1)}%)`;
  } else if (momentum.trend === 'bearish') {
    signal = Math.max(avgMomentum / 10, -1) * 0.8;
    reasoning = `Bearish trend (${avgMomentum.toFixed(1)}%)`;
  } else {
    signal = avgMomentum / 20;
    reasoning = `Mixed momentum (${avgMomentum.toFixed(1)}%)`;
  }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `Mom: ${avgMomentum.toFixed(1)}%`, reasoning };
}

function interpretVolume(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const volume = indicators.volume;
  const momentum = indicators.momentum;
  if (!volume) return null;
  
  let signal = 0;
  let reasoning = '';
  const isPriceUp = momentum ? momentum.shortTerm > 0 : null;
  let direction = isPriceUp !== null ? (isPriceUp ? 1 : -1) : 0;
  
  if (volume.isHighVolume) {
    if (direction !== 0) {
      signal = direction * 1.0;
      reasoning = `High volume confirms ${isPriceUp ? 'bullish' : 'bearish'} move (${volume.volumeRatio.toFixed(1)}x avg)`;
    } else {
      signal = 0.3;
      reasoning = `High volume (${volume.volumeRatio.toFixed(1)}x avg)`;
    }
  } else if (volume.isLowVolume) {
    if (direction !== 0) {
      signal = direction * 0.2;
      reasoning = `Low volume weakens ${isPriceUp ? 'bullish' : 'bearish'} move`;
    } else {
      signal = -0.1;
      reasoning = `Low volume - weak conviction`;
    }
  } else {
    signal = direction * 0.5;
    reasoning = direction !== 0 ? `Normal volume ${isPriceUp ? 'bullish' : 'bearish'}` : `Normal volume`;
  }
  
  if (volume.trend === 'increasing') { signal += direction * 0.15; reasoning += ', trend increasing'; }
  else if (volume.trend === 'decreasing') { signal -= direction * 0.1; reasoning += ', trend decreasing'; }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `Vol: ${volume.volumeRatio.toFixed(1)}x`, reasoning };
}

function interpretPricePosition(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const pos = indicators.pricePosition;
  if (!pos) return null;
  
  let signal = 0;
  let reasoning = '';
  const range = pos.rangePosition;
  
  if (range <= 0.2) { signal = 0.8; reasoning = `Near 52-week low (${(range * 100).toFixed(0)}% of range)`; }
  else if (range <= 0.4) { signal = 0.4; reasoning = `Lower range (${(range * 100).toFixed(0)}%)`; }
  else if (range <= 0.6) { signal = 0; reasoning = `Mid-range (${(range * 100).toFixed(0)}%)`; }
  else if (range <= 0.8) { signal = -0.4; reasoning = `Upper range (${(range * 100).toFixed(0)}%)`; }
  else { signal = -0.8; reasoning = `Near 52-week high (${(range * 100).toFixed(0)}%)`; }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal, rawValue: `Range: ${(range * 100).toFixed(0)}%`, reasoning };
}

function interpretSMAAlignment(indicators: TechnicalIndicators, currentPrice: number, action: ActionType): SignalResult | null {
  const { sma20, sma50, sma200 } = indicators;
  if (!sma20 || !sma50) return null;
  
  let signal = 0;
  let reasoning = '';
  
  const aboveSma20 = currentPrice > sma20;
  const aboveSma50 = currentPrice > sma50;
  const aboveSma200 = sma200 ? currentPrice > sma200 : null;
  const sma20Above50 = sma20 > sma50;
  const sma50Above200 = sma200 ? sma50 > sma200 : null;
  
  let bullishPoints = 0;
  const factors: string[] = [];
  if (aboveSma20) { bullishPoints++; factors.push('> SMA20'); }
  if (aboveSma50) { bullishPoints++; factors.push('> SMA50'); }
  if (aboveSma200) { bullishPoints++; factors.push('> SMA200'); }
  if (sma20Above50) { bullishPoints++; factors.push('SMA20 > SMA50'); }
  if (sma50Above200) { bullishPoints++; factors.push('SMA50 > SMA200'); }
  
  const maxPoints = sma200 ? 5 : 3;
  signal = (bullishPoints / maxPoints) * 2 - 1;
  
  if (bullishPoints >= maxPoints - 1) reasoning = `Bullish alignment: ${factors.slice(0, 2).join(', ')}`;
  else if (bullishPoints <= 1) reasoning = `Bearish alignment: below key SMAs`;
  else reasoning = `Mixed SMA signals`;
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `SMA20: ${sma20.toFixed(2)}`, reasoning };
}

function interpretRVOL(indicators: TechnicalIndicators, _action: ActionType): SignalResult | null {
  const rvol = indicators.rvol;
  if (!rvol) return null;
  
  let signal = 0;
  let reasoning = '';
  
  if (rvol.rvol > 3.0) { signal = 0.5; reasoning = `Very high volume (${rvol.rvol.toFixed(1)}x) - unusual activity`; }
  else if (rvol.rvol > 1.5) { signal = 0.3; reasoning = `Above avg volume (${rvol.rvol.toFixed(1)}x)`; }
  else if (rvol.rvol >= 0.8) { signal = 0; reasoning = `Normal volume (${rvol.rvol.toFixed(1)}x)`; }
  else if (rvol.rvol >= 0.5) { signal = -0.2; reasoning = `Low volume (${rvol.rvol.toFixed(1)}x)`; }
  else { signal = -0.4; reasoning = `Very low volume (${rvol.rvol.toFixed(1)}x) - lack of interest`; }
  
  return { signal, rawValue: `RVOL: ${rvol.rvol.toFixed(2)}x`, reasoning };
}

function interpretCrossPattern(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const cross = indicators.crossPattern;
  if (!cross) return null;
  
  let signal = 0;
  let reasoning = cross.description;
  
  switch (cross.pattern) {
    case 'golden_cross': signal = cross.strength === 'strong' ? 0.9 : 0.6; break;
    case 'golden_star': signal = 1.0; break;
    case 'death_cross': signal = cross.strength === 'strong' ? -0.9 : -0.6; break;
    case 'none': 
      if (cross.trendAlignment === 'bullish') signal = 0.4;
      else if (cross.trendAlignment === 'bearish') signal = -0.4;
      break;
  }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: cross.pattern !== 'none' ? cross.pattern.replace('_', ' ') : cross.trendAlignment, reasoning };
}

/**
 * Interpret ADX using backend-consistent generic signal.
 *
 * SIMPLIFIED: Removed action-specific logic to match backend WFO optimization.
 * Backend uses: trend strength + direction adjustment (bearish = negative signal).
 */
function interpretADX(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const adx = indicators.adx;
  if (!adx || adx.adx === undefined || adx.adx === null) return null;

  const adxValue = adx.adx;
  const direction = adx.direction;

  // Generic ADX signal matching backend's calculate_indicator_signals()
  // Backend thresholds: no_trend=20, trending=25, strong_trend=40
  let signal = 0;
  let reasoning = '';

  if (adxValue < 20) {
    signal = 0.0;
    reasoning = `No trend (ADX ${adxValue.toFixed(0)})`;
  } else if (adxValue < 25) {
    signal = 0.3;
    reasoning = `Weak trend (ADX ${adxValue.toFixed(0)})`;
  } else if (adxValue < 40) {
    signal = 0.6;
    reasoning = `Trending (ADX ${adxValue.toFixed(0)})`;
  } else {
    signal = 0.9;
    reasoning = `Strong trend (ADX ${adxValue.toFixed(0)})`;
  }

  // Direction adjustment (matches backend line 857)
  if (direction === 'bearish') {
    signal = -signal;
    reasoning += ` ${direction}`;
  } else if (direction === 'bullish') {
    reasoning += ` ${direction}`;
  }

  // Flip signal for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `ADX: ${adxValue.toFixed(0)}`, reasoning };
}

/**
 * Interpret Bollinger Squeeze using backend-consistent generic signal.
 *
 * SIMPLIFIED: Removed action-specific logic to match backend WFO optimization.
 * Backend uses: squeeze=0.8 (cheap options), expansion=-0.5 (expensive options), else 0.
 */
function interpretBollingerSqueeze(indicators: TechnicalIndicators, _action: ActionType): SignalResult | null {
  const squeeze = indicators.bollingerSqueeze;
  if (!squeeze || squeeze.bandwidthPercentile === undefined) return null;

  const { bandwidthPercentile, isSqueeze, isExpansion, squeezeIntensity } = squeeze;

  // Generic squeeze signal matching backend's calculate_indicator_signals()
  let signal = 0;
  let reasoning = '';

  if (isSqueeze) {
    signal = 0.8;
    reasoning = `${squeezeIntensity} squeeze (${bandwidthPercentile}%ile) - low volatility`;
  } else if (isExpansion) {
    signal = -0.5;
    reasoning = `Expansion (${bandwidthPercentile}%ile) - high volatility`;
  } else {
    signal = 0;
    reasoning = `Normal volatility (${bandwidthPercentile}%ile)`;
  }

  // Note: Backend squeeze signal is NOT direction-aware, so no flipping needed
  // The signal represents volatility state, not directional bias

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `BWP: ${bandwidthPercentile}%`, reasoning };
}

function interpretDivergence(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const divergence = indicators.divergence;
  if (!divergence) return null;
  
  let signal = 0;
  let reasoning = divergence.description;
  
  const { hasBullishDivergence, hasBearishDivergence, divergenceStrength } = divergence;
  
  if (hasBearishDivergence) {
    const strength = divergenceStrength === 'strong' ? -0.9 : -0.6;
    signal = strength;
  } else if (hasBullishDivergence) {
    const strength = divergenceStrength === 'strong' ? 0.8 : 0.5;
    signal = strength;
  } else {
    signal = 0;
    reasoning = 'No divergence';
  }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: divergence.volumeDivergence !== 'none' ? `${divergence.volumeDivergence} div.` : 'No div.', reasoning };
}

function interpretAlpha(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.alpha === undefined || additionalData?.alpha === null) return null;
  const alpha = additionalData.alpha;
  let signal = 0;
  let reasoning = '';
  
  if (alpha > 5) { signal = 1.0; reasoning = `Strong leader (+${alpha.toFixed(1)}% vs SPY)`; }
  else if (alpha > 2) { signal = 0.6; reasoning = `Outperforming (+${alpha.toFixed(1)}% vs SPY)`; }
  else if (alpha >= -2) { signal = alpha / 5; reasoning = `In line with market (${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}% vs SPY)`; }
  else if (alpha >= -5) { signal = -0.5; reasoning = `Underperforming (${alpha.toFixed(1)}% vs SPY)`; }
  else { signal = -0.9; reasoning = `Laggard (${alpha.toFixed(1)}% vs SPY)`; }
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `α: ${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}%`, reasoning };
}

function interpretEarningsProximity(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.daysToEarnings === undefined || additionalData?.daysToEarnings === null) return null;
  const days = additionalData.daysToEarnings;
  let signal = 0;
  let reasoning = '';
  
  if (days <= 0) { signal = -0.3; reasoning = `Earnings ${days === 0 ? 'today' : 'passed'} - IV crush risk`; }
  else if (days <= 3) { signal = -0.8; reasoning = `Earnings in ${days} day${days === 1 ? '' : 's'} - technicals unreliable`; }
  else if (days <= 7) { signal = -0.4; reasoning = `Earnings in ${days} days - elevated uncertainty`; }
  else if (days <= 14) { signal = -0.1; reasoning = `Earnings in ${days} days`; }
  else { signal = 0.1; reasoning = `Earnings ${days}+ days away - technicals reliable`; }
  
  const optionActions: ActionType[] = ['openCSP', 'openCC', 'buyCall', 'buyPut'];
  if (optionActions.includes(action)) {
    if (days <= 3) { signal = -1.0; reasoning += ' (IV crush risk)'; }
    else if (days <= 7) signal *= 1.5;
  }
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: days <= 0 ? 'Earnings passed' : `${days}d to ER`, reasoning };
}

function interpretSectorBeta(additionalData: AdditionalAnalysisData | undefined, _action: ActionType): SignalResult | null {
  if (!additionalData?.betaToSector && !additionalData?.sectorCorrelation) return null;
  const beta = additionalData.betaToSector;
  const correlation = additionalData.sectorCorrelation;
  let signal = 0;
  let reasoning = '';
  
  if (beta !== null && beta !== undefined) {
    if (beta > 1.5) { signal = 0.2; reasoning = `High sector beta (${beta.toFixed(2)}) - amplified moves`; }
    else if (beta >= 1.0) { signal = 0.1; reasoning = `Normal sector beta (${beta.toFixed(2)})`; }
    else if (beta >= 0.8) { signal = 0; reasoning = `Tracks sector (β=${beta.toFixed(2)})`; }
    else if (beta >= 0.5) { signal = -0.1; reasoning = `Lower sector beta (${beta.toFixed(2)}) - defensive`; }
    else { signal = 0; reasoning = `Low sector correlation (β=${beta.toFixed(2)})`; }
  } else if (correlation !== null && correlation !== undefined) {
    signal = correlation > 0.7 ? 0.1 : correlation < 0.3 ? -0.1 : 0;
    reasoning = `Sector corr: ${(correlation * 100).toFixed(0)}%`;
  }
  
  return { signal, rawValue: beta ? `β: ${beta.toFixed(2)}` : `Corr: ${((correlation ?? 0) * 100).toFixed(0)}%`, reasoning };
}

function interpretCMF(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const cmf = indicators.cmf;
  if (!cmf) return null;
  
  let signal = 0;
  let reasoning = '';
  const { value, interpretation, volumeStrength, closeLocation } = cmf;
  
  if (interpretation === 'strong_accumulation') { signal = 1.0; reasoning = `Strong accumulation (${(value * 100).toFixed(0)}%)`; }
  else if (interpretation === 'accumulation') { signal = 0.6; reasoning = `Accumulation (${(value * 100).toFixed(0)}%)`; }
  else if (interpretation === 'neutral') { signal = value * 2; reasoning = `Neutral money flow (${(value * 100).toFixed(0)}%)`; }
  else if (interpretation === 'distribution') { signal = -0.6; reasoning = `Distribution (${(value * 100).toFixed(0)}%)`; }
  else { signal = -1.0; reasoning = `Strong distribution (${(value * 100).toFixed(0)}%)`; }
  
  if (volumeStrength === 'high') { signal *= 1.2; reasoning += ', high volume'; }
  else if (volumeStrength === 'low') { signal *= 0.6; reasoning += ', low volume'; }
  
  if (closeLocation > 0.8 && value > 0) reasoning += ', closed near high';
  else if (closeLocation < 0.2 && value < 0) reasoning += ', closed near low';
  
  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;
  
  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `CMF: ${(value * 100).toFixed(0)}%`, reasoning };
}

function detectBuyingContext(indicators: TechnicalIndicators, currentPrice: number): 'trend' | 'value' | 'neutral' {
  const sma50 = indicators.sma50;
  const sma200 = indicators.sma200;
  const bb = indicators.bollingerBands;
  const rsi = indicators.rsi;
  
  if (!sma50) return 'neutral';
  
  const priceBelowSma50 = currentPrice < sma50;
  const priceNearLowerBand = bb ? bb.percentB < 0.3 : false;
  const isOversold = rsi ? rsi.value < 40 : false;
  const priceAboveSma50 = currentPrice > sma50;
  const priceAboveSma200 = sma200 ? currentPrice > sma200 : priceAboveSma50;
  
  if (priceBelowSma50 && (priceNearLowerBand || isOversold)) return 'value';
  if (priceAboveSma50 && priceAboveSma200) return 'trend';
  return 'neutral';
}

/**
 * Apply context-aware weight adjustments (additive).
 * Returns the adjusted weight clamped to [WEIGHT_MIN, WEIGHT_MAX].
 */
function getContextAdjustedWeight(
  metric: MetricType,
  action: ActionType,
  context: 'trend' | 'value' | 'neutral',
  baseWeight: number
): number {
  if (action !== 'buyShares') return baseWeight;

  let adjustment = 0;

  if (context === 'value') {
    // Value context: favor mean-reversion indicators
    switch (metric) {
      case 'smaAlignment': adjustment = -0.8; break;  // Was 0.2x → -0.8
      case 'crossPattern': adjustment = -0.7; break;  // Was 0.3x → -0.7
      case 'rsi': adjustment = 1.0; break;            // Was 2.0x → +1.0
      case 'bollingerBands': adjustment = 0.8; break; // Was 1.8x → +0.8
      case 'pricePosition': adjustment = 0.5; break;  // Was 1.5x → +0.5
      case 'cmf': adjustment = 1.0; break;            // Was 2.0x → +1.0
      case 'divergence': adjustment = 0.8; break;     // Was 1.8x → +0.8
    }
  } else if (context === 'trend') {
    // Trend context: favor trend-following indicators
    switch (metric) {
      case 'smaAlignment': adjustment = 0.2; break;   // Was 1.2x → +0.2
      case 'macd': adjustment = 0.2; break;           // Was 1.2x → +0.2
      case 'adx': adjustment = 0.3; break;            // Was 1.3x → +0.3
      case 'pricePosition': adjustment = -0.3; break; // Was 0.7x → -0.3
    }
  }

  const result = baseWeight + adjustment;
  return Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, result));
}

// ============================================================================
// WFO Integration Helpers
// ============================================================================

const BACKEND_TO_FRONTEND_METRICS: Record<string, MetricType> = {
  'rsi': 'rsi',
  'macd': 'macd',
  'bollinger': 'bollingerBands',
  'squeeze': 'bollingerSqueeze',
  'adx': 'adx',
  'cmf': 'cmf',
  'momentum': 'momentum',
  'volume': 'volume',
  'rvol': 'rvol',
  'sma': 'smaAlignment',
  'position': 'pricePosition',
  'vwap': 'vwap',  // Volume-weighted average price
  'rel_momentum': 'relMomentum',
  'rs_ratio': 'rsRatio'
};

/**
 * Apply WFO calibration to the weight matrix using additive offsets.
 * Backend stores multipliers (e.g., 1.5 = 50% boost), converted to offsets: (multiplier - 1.0).
 * All weights are clamped to [WEIGHT_MIN, WEIGHT_MAX] to prevent extreme values.
 */
function applyWfoMultipliers(
  defaults: WeightMatrix,
  multipliers: Record<string, number>
): WeightMatrix {
  const result: any = JSON.parse(JSON.stringify(defaults));

  for (const [backendKey, multiplier] of Object.entries(multipliers)) {
    const metricKey = BACKEND_TO_FRONTEND_METRICS[backendKey];
    if (!metricKey || !result[metricKey]) continue;

    // Convert multiplier to additive offset: 1.5x → +0.5, 0.5x → -0.5
    const offset = multiplier - 1.0;

    for (const action of Object.keys(result[metricKey])) {
      const adjusted = result[metricKey][action] + offset;
      result[metricKey][action] = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, adjusted));
    }
  }

  return result;
}

/**
 * Apply horizon-specific adjustments to the default weights using additive offsets.
 * Used when no WFO calibration is available to give "default" behavior for different horizons.
 * All weights are clamped to [WEIGHT_MIN, WEIGHT_MAX] to prevent extreme values.
 */
function applyHorizonDefaults(
  defaults: WeightMatrix,
  horizon: number
): WeightMatrix {
  const result: any = JSON.parse(JSON.stringify(defaults));
  const adjustments = horizon === 15 ? TREND_ADJUSTMENTS : SWING_ADJUSTMENTS;

  for (const [metric, adjustment] of Object.entries(adjustments)) {
    // Skip if metric not found
    if (!result[metric]) continue;

    for (const action of Object.keys(result[metric])) {
      const adjusted = result[metric][action] + adjustment;
      result[metric][action] = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, adjusted));
    }
  }

  return result;
}

function interpretROIC(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.roic === undefined || additionalData?.roic === null) return null;
  const roic = additionalData.roic;
  let signal = 0;
  let reasoning = '';

  if (roic > 20) { signal = 1.0; reasoning = `Exceptional ROIC (${roic.toFixed(1)}%)`; }
  else if (roic > 15) { signal = 0.7; reasoning = `Strong ROIC (${roic.toFixed(1)}%)`; }
  else if (roic > 10) { signal = 0.3; reasoning = `Decent ROIC (${roic.toFixed(1)}%)`; }
  else if (roic > 0) { signal = 0; reasoning = `Positive ROIC (${roic.toFixed(1)}%)`; }
  else { signal = -0.5; reasoning = `Negative ROIC (${roic.toFixed(1)}%)`; }

  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `ROIC: ${roic.toFixed(1)}%`, reasoning };
}

function interpretCallPutRatio(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.callPutRatio === undefined || additionalData?.callPutRatio === null) return null;
  const ratio = additionalData.callPutRatio;
  let signal = 0;
  let reasoning = '';

  // High ratio = Bullish sentiment (more calls than puts)
  // Low ratio = Bearish sentiment (more puts than calls)
  if (ratio > 2.0) { signal = 0.8; reasoning = `Strong bullish sentiment (C/P ${ratio.toFixed(2)})`; }
  else if (ratio > 1.2) { signal = 0.4; reasoning = `Bullish sentiment (C/P ${ratio.toFixed(2)})`; }
  else if (ratio >= 0.8) { signal = 0; reasoning = `Neutral sentiment (C/P ${ratio.toFixed(2)})`; }
  else if (ratio >= 0.5) { signal = -0.4; reasoning = `Bearish sentiment (C/P ${ratio.toFixed(2)})`; }
  else { signal = -0.8; reasoning = `Strong bearish sentiment (C/P ${ratio.toFixed(2)})`; }

  const bearishActions: ActionType[] = ['sellShares', 'buyPut'];
  if (bearishActions.includes(action)) signal = -signal;

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `C/P: ${ratio.toFixed(2)}`, reasoning };
}

function interpretIVPercentile(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.ivPercentile === undefined || additionalData?.ivPercentile === null) return null;
  const ivp = additionalData.ivPercentile;
  let signal = 0;
  let reasoning = '';

  // High IVP: Sell premium (openCSP, openCC)
  // Low IVP: Buy premium (buyCall, buyPut)
  
  if (action === 'openCSP' || action === 'openCC') {
    if (ivp > 80) { signal = 1.0; reasoning = `High IV (${ivp}%) - Rich premiums`; }
    else if (ivp > 50) { signal = 0.6; reasoning = `Elevated IV (${ivp}%)`; }
    else if (ivp > 20) { signal = 0; reasoning = `Moderate IV (${ivp}%)`; }
    else { signal = -0.5; reasoning = `Low IV (${ivp}%) - Cheap premiums`; }
  } else if (action === 'buyCall' || action === 'buyPut') {
    if (ivp < 20) { signal = 0.8; reasoning = `Low IV (${ivp}%) - Cheap options`; }
    else if (ivp < 50) { signal = 0.4; reasoning = `Moderate IV (${ivp}%)`; }
    else if (ivp < 80) { signal = -0.2; reasoning = `Elevated IV (${ivp}%)`; }
    else { signal = -0.8; reasoning = `High IV (${ivp}%) - Expensive options`; }
  } else {
    // For shares, high IV usually means fear/uncertainty
    if (ivp > 80) { signal = -0.3; reasoning = `Extreme volatility (${ivp}%)`; }
    else { signal = 0; reasoning = `IV Percentile: ${ivp}%`; }
  }

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `IVP: ${ivp}%`, reasoning };
}

function interpretOptionsSentiment(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (!additionalData?.optionsSentiment) return null;
  const sentiment = additionalData.optionsSentiment.toLowerCase();
  let signal = 0;
  let reasoning = '';

  // Options flow sentiment based on call/put ratios
  // Bullish sentiment = more calls being traded/held
  // Bearish sentiment = more puts being traded/held

  if (sentiment === 'bullish') {
    signal = 0.6;
    reasoning = 'Bullish options flow';
  } else if (sentiment === 'bearish') {
    signal = -0.6;
    reasoning = 'Bearish options flow';
  } else {
    signal = 0;
    reasoning = 'Neutral options flow';
  }

  // Adjust based on action type
  if (action === 'openCSP') {
    // For CSP, bullish flow confirms we want stock to stay flat/up
    // signal stays as-is
  } else if (action === 'openCC') {
    // For CC, bearish flow means less chance of being called away
    signal = -signal; // Bearish is good for CC
    reasoning = sentiment === 'bullish'
      ? 'Bullish flow - assignment risk'
      : sentiment === 'bearish'
        ? 'Bearish flow - safe CC'
        : reasoning;
  } else if (action === 'buyCall') {
    // Bullish flow confirms call direction
    // signal stays as-is
  } else if (action === 'buyPut') {
    // Bearish flow confirms put direction
    signal = -signal;
    reasoning = sentiment === 'bullish'
      ? 'Bullish flow - against put'
      : sentiment === 'bearish'
        ? 'Bearish flow - confirms put'
        : reasoning;
  } else if (action === 'sellShares') {
    // Bearish actions - flip signal
    signal = -signal;
  }

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `Flow: ${sentiment}`, reasoning };
}

/**
 * Interpret Relative Momentum (stock vs sector performance).
 * Positive = outperforming sector, Negative = underperforming sector.
 */
function interpretRelMomentum(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.relMomentum === undefined || additionalData?.relMomentum === null) return null;
  const relMom = additionalData.relMomentum;
  let signal = 0;
  let reasoning = '';

  if (relMom > 10) { signal = 1.0; reasoning = `Strong outperformance (+${relMom.toFixed(1)}% vs sector)`; }
  else if (relMom > 5) { signal = 0.6; reasoning = `Outperforming sector (+${relMom.toFixed(1)}%)`; }
  else if (relMom > 0) { signal = 0.3; reasoning = `Slightly outperforming (+${relMom.toFixed(1)}%)`; }
  else if (relMom > -5) { signal = -0.3; reasoning = `Slightly underperforming (${relMom.toFixed(1)}%)`; }
  else if (relMom > -10) { signal = -0.6; reasoning = `Underperforming sector (${relMom.toFixed(1)}%)`; }
  else { signal = -1.0; reasoning = `Strong underperformance (${relMom.toFixed(1)}% vs sector)`; }

  // For sell actions, flip the signal
  if (action === 'sellShares' || action === 'buyPut') {
    signal = -signal;
  }

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `Rel Mom: ${relMom.toFixed(1)}%`, reasoning };
}

/**
 * Interpret RS Ratio (RRG-style relative strength).
 * > 100 = outperforming, < 100 = underperforming.
 */
function interpretRsRatio(additionalData: AdditionalAnalysisData | undefined, action: ActionType): SignalResult | null {
  if (additionalData?.rsRatio === undefined || additionalData?.rsRatio === null) return null;
  const rsRatio = additionalData.rsRatio;
  const rsMomentum = additionalData.rsMomentum ?? 0;
  let signal = 0;
  let reasoning = '';

  const rsAbove100 = rsRatio >= 100;
  const momPositive = rsMomentum >= 0;

  // RRG Quadrant logic
  if (rsAbove100 && momPositive) {
    // Leading quadrant
    signal = 0.8;
    reasoning = `Leading (RS ${rsRatio.toFixed(0)}, Mom+)`;
  } else if (rsAbove100 && !momPositive) {
    // Weakening quadrant
    signal = 0.2;
    reasoning = `Weakening (RS ${rsRatio.toFixed(0)}, Mom-)`;
  } else if (!rsAbove100 && !momPositive) {
    // Lagging quadrant
    signal = -0.8;
    reasoning = `Lagging (RS ${rsRatio.toFixed(0)}, Mom-)`;
  } else {
    // Improving quadrant
    signal = 0.4;
    reasoning = `Improving (RS ${rsRatio.toFixed(0)}, Mom+)`;
  }

  // For sell actions, flip the signal
  if (action === 'sellShares' || action === 'buyPut') {
    signal = -signal;
  }

  return { signal: Math.max(-1, Math.min(1, signal)), rawValue: `RS: ${rsRatio.toFixed(0)}`, reasoning };
}

/**
 * Interpret Anchored VWAP signal.
 * AVWAP shows if recent holders are profitable (support) or underwater (resistance).
 */
function interpretAVWAP(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const avwap = indicators.avwap;
  if (!avwap) return null;

  let signal = 0;
  let reasoning = '';

  // Price above AVWAP = holders profitable = support level (bullish)
  // Price below AVWAP = holders underwater = resistance level (bearish)
  if (avwap.status === 'support') {
    signal = Math.min(1, avwap.priceVsAvwap / 10); // Scale: 10% above = +1
    reasoning = `${avwap.priceVsAvwap.toFixed(1)}% above AVWAP - holders profitable`;
  } else {
    signal = Math.max(-1, avwap.priceVsAvwap / 10); // Negative priceVsAvwap
    reasoning = `${Math.abs(avwap.priceVsAvwap).toFixed(1)}% below AVWAP - holders underwater`;
  }

  // For sell/bearish actions, flip the signal
  if (action === 'sellShares' || action === 'buyPut') {
    signal = -signal;
  }

  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `$${avwap.value.toFixed(2)}`,
    reasoning
  };
}

/**
 * Interpret On-Balance Volume signal.
 * OBV reveals institutional accumulation or distribution.
 */
function interpretOBV(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const obv = indicators.obv;
  if (!obv) return null;

  let signal = 0;
  let reasoning = '';

  // Divergence is the strongest signal
  if (obv.hasDivergence) {
    if (obv.divergenceType === 'bullish') {
      signal = 0.9;
      reasoning = `Bullish divergence: institutions accumulating despite price weakness`;
    } else if (obv.divergenceType === 'bearish') {
      signal = -0.9;
      reasoning = `Bearish divergence: institutions distributing despite price strength`;
    }
  } else {
    // Normal trend
    if (obv.trend === 'accumulation') {
      signal = Math.min(0.7, obv.trendStrength / 100);
      reasoning = `Accumulation: +${obv.obvChange.toFixed(0)}% OBV`;
    } else if (obv.trend === 'distribution') {
      signal = -Math.min(0.7, obv.trendStrength / 100);
      reasoning = `Distribution: ${obv.obvChange.toFixed(0)}% OBV`;
    } else {
      signal = 0;
      reasoning = `Neutral volume flow`;
    }
  }

  // For sell/bearish actions, flip the signal
  if (action === 'sellShares' || action === 'buyPut') {
    signal = -signal;
  }

  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: obv.trend,
    reasoning
  };
}

/**
 * Interpret Chandelier Exit signal.
 * Trailing stop that shows if trend is intact or broken.
 */
function interpretChandelier(indicators: TechnicalIndicators, action: ActionType): SignalResult | null {
  const chandelier = indicators.chandelier;
  if (!chandelier) return null;

  let signal = 0;
  let reasoning = '';

  if (chandelier.status === 'intact') {
    // Trend intact - more buffer to stop = stronger
    const bufferStrength = Math.min(1, chandelier.distanceToStop / 10); // 10% buffer = max
    signal = 0.3 + bufferStrength * 0.5; // Range: 0.3 to 0.8
    reasoning = `Trend intact - stop at $${chandelier.stopPrice.toFixed(2)} (${chandelier.distanceToStop.toFixed(1)}% buffer)`;
  } else {
    // Trend broken - strong bearish signal
    signal = -0.8;
    reasoning = `Trend BROKEN - price below stop at $${chandelier.stopPrice.toFixed(2)}`;
  }

  // Chandelier is particularly important for sell decisions
  if (action === 'sellShares') {
    // Broken trend is a strong sell signal (keep negative)
    // Intact trend is a hold signal (flip to negative = don't sell)
    signal = -signal;
  } else if (action === 'buyPut') {
    // Broken trend supports buying puts (keep as is, broken = positive for puts)
    signal = -signal;
  } else if (action === 'openCC') {
    // Intact trend makes covered calls safer
    // No change needed
  }

  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: chandelier.status,
    reasoning
  };
}

/**
 * Calculate score for a single action based on all indicators.
 */
function calculateActionScore(
  action: ActionType,
  indicators: TechnicalIndicators,
  currentPrice: number,
  weights: WeightMatrix,
  additionalData?: AdditionalAnalysisData
): ActionScore {
  const signals: MetricSignal[] = [];
  let totalScore = 0;
  let totalWeight = 0;
  
  // Detect buying context for dynamic weighting
  const buyingContext = detectBuyingContext(indicators, currentPrice);

  for (const metric of ALL_METRICS) {
    let result: SignalResult | null = null;
    
    // Call appropriate interpreter
    switch (metric) {
      case 'rsi': result = interpretRSI(indicators, action); break;
      case 'macd': result = interpretMACD(indicators, action); break;
      case 'bollingerBands': result = interpretBollingerBands(indicators, action); break;
      case 'bollingerSqueeze': result = interpretBollingerSqueeze(indicators, action); break;
      case 'vwap': result = interpretVWAP(indicators, action); break;
      case 'momentum': result = interpretMomentum(indicators, action); break;
      case 'volume': result = interpretVolume(indicators, action); break;
      case 'pricePosition': result = interpretPricePosition(indicators, action); break;
      case 'smaAlignment': result = interpretSMAAlignment(indicators, currentPrice, action); break;
      case 'rvol': result = interpretRVOL(indicators, action); break;
      case 'adx': result = interpretADX(indicators, action); break;
      case 'crossPattern': result = interpretCrossPattern(indicators, action); break;
      case 'cmf': result = interpretCMF(indicators, action); break;
      case 'divergence': result = interpretDivergence(indicators, action); break;
      case 'alpha': result = interpretAlpha(additionalData, action); break;
      case 'earningsProximity': result = interpretEarningsProximity(additionalData, action); break;
      case 'sectorBeta': result = interpretSectorBeta(additionalData, action); break;
      case 'roic': result = interpretROIC(additionalData, action); break;
      case 'callPutRatio': result = interpretCallPutRatio(additionalData, action); break;
      case 'ivPercentile': result = interpretIVPercentile(additionalData, action); break;
      case 'optionsSentiment': result = interpretOptionsSentiment(additionalData, action); break;
      case 'relMomentum': result = interpretRelMomentum(additionalData, action); break;
      case 'rsRatio': result = interpretRsRatio(additionalData, action); break;
      case 'avwap': result = interpretAVWAP(indicators, action); break;
      case 'obv': result = interpretOBV(indicators, action); break;
      case 'chandelier': result = interpretChandelier(indicators, action); break;
    }

    if (result) {
      let weight = weights[metric]?.[action] ?? 0;
      
      // Apply context-aware weighting for buyShares
      if (action === 'buyShares') {
        weight = getContextAdjustedWeight(metric, action, buyingContext, weight);
      }
      
      const contribution = result.signal * weight;
      
      signals.push({
        metric,
        metricLabel: metricLabels[metric],
        rawValue: result.rawValue,
        signal: result.signal,
        weight,
        contribution,
        reasoning: result.reasoning
      });
      
      totalScore += contribution;
      totalWeight += weight;
    }
  }
  
  // Normalize score to 0-100
  // Normalized = ((totalScore / totalWeight) + 1) * 50
  // If totalWeight is 0, defaults to 50
  const normalizedScore = totalWeight > 0 
    ? Math.round(((totalScore / totalWeight) + 1) * 50) 
    : 50;
    
  return {
    action,
    label: actionLabels[action],
    totalScore: Math.max(0, Math.min(100, normalizedScore)),
    rawScore: totalScore,
    signals,
    confidence: signals.length > 5 ? 'high' : signals.length > 2 ? 'medium' : 'low'
  };
}

// ============================================================================
// Ensemble Scoring
// ============================================================================

/**
 * Get ensemble weights based on calibration quality (SQN).
 * Higher SQN = trust WFO more; lower SQN = rely more on rules.
 */
function getEnsembleWeights(sqn: number | null): { wfo: number; regime: number; consensus: number } {
  if (!sqn || sqn < 0.5) {
    // Poor calibration - don't trust WFO much
    return { wfo: 0.2, regime: 0.5, consensus: 0.3 };
  }
  if (sqn < 1.5) {
    // Fair calibration - balanced approach
    return { wfo: 0.35, regime: 0.35, consensus: 0.3 };
  }
  if (sqn < 2.5) {
    // Good calibration - trust WFO
    return { wfo: 0.5, regime: 0.25, consensus: 0.25 };
  }
  // Excellent calibration - trust WFO heavily
  return { wfo: 0.6, regime: 0.2, consensus: 0.2 };
}

/**
 * Apply ensemble scoring to a base score.
 * Combines WFO-calibrated score with regime rules and consensus signals.
 *
 * @param baseScore - The WFO-weighted score (0-100)
 * @param action - The action being scored
 * @param regime - Current market regime
 * @param indicators - Technical indicators for consensus
 * @param sqn - Calibration quality score
 * @param wfoWeights - Optional WFO calibration weights (passed to consensus for weight-aware filtering)
 * @returns Adjusted ensemble score (0-100)
 */
function applyEnsembleScoring(
  baseScore: number,
  action: ActionType,
  regime: MarketRegime | null | undefined,
  indicators: TechnicalIndicators,
  sqn: number | null,
  wfoWeights?: Record<string, number>
): number {
  // If no regime data, return base score
  if (!regime) {
    return baseScore;
  }

  // Get ensemble weights
  const weights = getEnsembleWeights(sqn);

  // Normalize baseScore to -1 to +1 range
  const normalizedBase = (baseScore - 50) / 50;

  // Get regime score (-1 to +1)
  const regimeScore = getRegimeScore(regime, action);

  // Get consensus score (-1 to +1) - now weight-aware
  const consensusScore = calculateConsensusScore(indicators, action, wfoWeights);

  // Weighted combination
  const ensembleScore =
    weights.wfo * normalizedBase +
    weights.regime * regimeScore +
    weights.consensus * consensusScore;

  // Convert back to 0-100
  const finalScore = (ensembleScore + 1) * 50;

  return Math.max(0, Math.min(100, Math.round(finalScore)));
}

// ============================================================================
// Main Analysis Function
// ============================================================================

/**
 * Strategy-specific weights map type.
 * Maps strategy class names to their calibrated WFO multipliers.
 */
export type StrategyWeightsMap = Record<string, Record<string, number>>;

/**
 * Calculate scores for all actions and produce complete analysis.
 *
 * @param strategyWeightsMap - Optional map of strategy class → WFO multipliers.
 *   When provided, each action uses weights calibrated for its strategy class.
 */
export function analyzeStock(
  ticker: string,
  history: HistoryPoint[],
  currentPrice: number,
  high52w: number | null,
  low52w: number | null,
  wfoMultipliers?: Record<string, number>,
  additionalData?: AdditionalAnalysisData,
  calibrationMetadata?: { lastCalibrated: string; sqn: number | null; period: number },
  horizon: number = 3,
  strategyWeightsMap?: StrategyWeightsMap
): StockAnalysis {
  const indicators = calculateAllIndicators(history, currentPrice, high52w, low52w);

  // 1. Determine weights per strategy class
  // Build a map of strategy → weight matrix
  const strategyToWeights: Record<StrategyClass, WeightMatrix> = {
    directional: BASE_WEIGHTS,
    premium_sell: BASE_WEIGHTS,
    premium_buy: BASE_WEIGHTS,
    all: BASE_WEIGHTS,
  };

  // Apply WFO multipliers per strategy class if available
  if (strategyWeightsMap && Object.keys(strategyWeightsMap).length > 0) {
    // We have strategy-specific weights
    for (const strategyClass of ['directional', 'premium_sell', 'premium_buy'] as StrategyClass[]) {
      const multipliers = strategyWeightsMap[strategyClass];
      if (multipliers) {
        strategyToWeights[strategyClass] = applyWfoMultipliers(BASE_WEIGHTS, multipliers);
      } else {
        // Fallback to horizon defaults if no calibration for this strategy
        strategyToWeights[strategyClass] = applyHorizonDefaults(BASE_WEIGHTS, horizon);
      }
    }
  } else if (wfoMultipliers) {
    // Legacy: single wfoMultipliers applied to all strategies
    const calibratedWeights = applyWfoMultipliers(BASE_WEIGHTS, wfoMultipliers);
    strategyToWeights.directional = calibratedWeights;
    strategyToWeights.premium_sell = calibratedWeights;
    strategyToWeights.premium_buy = calibratedWeights;
    strategyToWeights.all = calibratedWeights;
  } else {
    // No WFO calibration - apply horizon defaults
    const defaultWeights = applyHorizonDefaults(BASE_WEIGHTS, horizon);
    strategyToWeights.directional = defaultWeights;
    strategyToWeights.premium_sell = defaultWeights;
    strategyToWeights.premium_buy = defaultWeights;
    strategyToWeights.all = defaultWeights;
  }

  // Calculate base scores for each action using the appropriate strategy weights
  const baseScores: ActionScore[] = ALL_ACTIONS.map(action => {
    const strategyClass = getStrategyForAction(action);
    const weights = strategyToWeights[strategyClass];
    return calculateActionScore(action, indicators, currentPrice, weights, additionalData);
  });

  // Apply ensemble scoring if regime data is available
  const regime = additionalData?.marketRegime;
  const sqn = calibrationMetadata?.sqn ?? null;

  const scores: ActionScore[] = baseScores.map(score => {
    // Check if this action is HARD BLOCKED by the current regime
    // This matches the backend's WFO behavior where signals are zeroed out
    const isBlocked = isActionBlockedByRegime(regime, score.action);

    if (isBlocked) {
      // Hard block: set score to 0 to match backend's zero-signal behavior
      return {
        ...score,
        totalScore: 0,
        confidence: 'low' as const,
        // Add a signal explaining the block
        signals: [
          {
            metric: 'pricePosition' as MetricType,
            metricLabel: 'Regime Filter',
            rawValue: `${regime} blocks ${score.action}`,
            signal: -1,
            weight: 1,
            contribution: -100,
            reasoning: `Action blocked by market regime (${regime}). Backend WFO simulation zeros out these signals.`
          },
          ...score.signals.slice(0, 5) // Keep some original signals for context
        ]
      };
    }

    if (regime) {
      const ensembleScore = applyEnsembleScoring(
        score.totalScore,
        score.action,
        regime,
        indicators,
        sqn,
        wfoMultipliers // Pass WFO weights for weight-aware consensus
      );
      return {
        ...score,
        totalScore: ensembleScore,
        // Keep rawScore as the pre-ensemble WFO score for reference
      };
    }
    return score;
  });

  // Find the best non-blocked action
  const nonBlockedScores = scores.filter(s => !isActionBlockedByRegime(regime, s.action));
  const bestAction = nonBlockedScores.length > 0
    ? nonBlockedScores.reduce((prev, current) =>
        (current.totalScore > prev.totalScore) ? current : prev
      )
    : scores.reduce((prev, current) =>
        (current.totalScore > prev.totalScore) ? current : prev
      );

  // Check if the highest scoring action overall was blocked
  const highestOverallAction = scores.reduce((prev, current) => {
    // Compare raw scores to see what WOULD have been recommended
    const prevOriginal = baseScores.find(s => s.action === prev.action)?.totalScore ?? 0;
    const currentOriginal = baseScores.find(s => s.action === current.action)?.totalScore ?? 0;
    return currentOriginal > prevOriginal ? current : prev;
  });
  const wouldHaveBeenBlocked = isActionBlockedByRegime(regime, highestOverallAction.action);
  
  const totalMetrics = ALL_METRICS.length;
  const availableMetrics = scores[0].signals.length;
  const missingMetrics = ALL_METRICS.filter(m => 
    !scores[0].signals.some(s => s.metric === m)
  );
  
  // Determine status: regime_blocked takes precedence over other checks
  // (low_confidence is set by the hook, not here)
  const status: 'ok' | 'regime_blocked' = wouldHaveBeenBlocked ? 'regime_blocked' : 'ok';

  return {
    ticker,
    analyzedAt: new Date(),
    scores,
    bestAction,
    hasOptions: additionalData?.hasOptions ?? false, // Default to false - only show options if explicitly available
    calibration: calibrationMetadata,
    marketRegime: regime ?? null, // Include regime for UI display
    status,
    dataQuality: {
      availableMetrics,
      totalMetrics,
      missingMetrics,
      historyDays: history.length,
      hasSMA200: !!indicators.sma200
    }
  };
}