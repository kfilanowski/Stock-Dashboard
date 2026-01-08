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

// ============================================================================
// Weight Matrix Configuration
// ============================================================================

/**
 * Weight matrix: [metric][action] = weight
 * 
 * Optimized weights based on trading strategy requirements:
 * - buyShares: Long-term hold - focus on trend + fundamentals
 * - sellShares: Exit position - focus on reversal signals
 * - openCSP: Sell cash-secured puts - mean reversion + high IV
 * - openCC: Sell covered calls - premium collection on held positions
 * - buyCall: Long options - momentum + explosion (time decay hurts)
 * - buyPut: Long puts - bearish momentum + protection
 * 
 * Positive weight = metric signal aligns with action
 * Weight magnitude = importance (0 = ignore, 2 = very important)
 */
type WeightMatrix = Record<MetricType, Record<ActionType, number>>;

/**
 * OPTIMIZED WEIGHT MATRIX v2.0
 * 
 * Key Design Principles:
 * 1. MULTICOLLINEARITY REDUCTION: Momentum weight reduced since MACD/RSI capture velocity
 * 2. STRATEGY SEPARATION: Long Gamma (buyCall/buyPut) vs Short Vega (openCSP/openCC)
 *    have distinct factor exposures
 * 3. REGIME AWARENESS: ADX and BollingerSqueeze now included for regime detection
 * 4. VALUE vs TREND: Context switch logic handles SMA alignment dynamically
 * 5. DIVERGENCE DETECTION: Fakeout detection via price-volume divergence
 * 6. SMART MONEY TRACKING: CMF (Chaikin Money Flow) for accumulation/distribution
 * 7. ALPHA FACTOR: Relative strength vs market (leaders vs laggards)
 * 
 * Greeks alignment:
 * - buyCall/buyPut: Long Gamma, Long Vega → favor ADX trending + squeeze (cheap vol)
 * - openCSP/openCC: Short Gamma, Short Vega → favor ADX sideways + expansion (rich vol)
 */
const DEFAULT_WEIGHTS: WeightMatrix = {
  // ============================================================================
  // OSCILLATORS (Mean Reversion vs Momentum)
  // ============================================================================
  
  // RSI: Regime-aware now via ADX filter in interpretation
  rsi: {
    buyShares: 0.8,   // Moderate - avoid extremes
    sellShares: 1.0,  // Important - sell overbought
    openCSP: 1.8,      // HIGH - sell puts in oversold (high IV + bounce expected)
    openCC: 1.5,       // Important - sell calls in overbought
    buyCall: 0.3,     // LOW - RSI interpretation now ADX-aware
    buyPut: 0.3       // LOW - RSI interpretation now ADX-aware
  },
  
  // MACD: Primary trend-following signal (consolidated with momentum)
  macd: {
    buyShares: 1.2,   // Important for entry timing
    sellShares: 1.2,
    openCSP: 0.6,      // Lower - not chasing trend for premium selling
    openCC: 0.6,
    buyCall: 1.8,     // HIGH - crossovers = immediate directional move
    buyPut: 1.8
  },
  
  // ============================================================================
  // VOLATILITY REGIME (NEW - Core for Options Strategy Separation)
  // ============================================================================
  
  // Bollinger Bands: Mean reversion position
  bollingerBands: {
    buyShares: 0.8,
    sellShares: 0.8,
    openCSP: 1.5,      // Important - sell puts at lower band
    openCC: 1.5,       // Important - sell calls at upper band
    buyCall: 0.4,     // Low - breakouts pierce bands
    buyPut: 0.4
  },
  
  // Bollinger Squeeze (NEW): Volatility contraction/expansion
  // CRITICAL for options strategy selection
  bollingerSqueeze: {
    buyShares: 0.3,   // Minor consideration
    sellShares: 0.3,
    openCSP: 1.2,      // Moderate - prefer high vol for premium
    openCC: 1.2,
    buyCall: 2.0,     // CRITICAL - squeeze = cheap options before explosion
    buyPut: 2.0
  },
  
  // ============================================================================
  // TREND STRENGTH (NEW - Regime Filter)
  // ============================================================================
  
  // ADX (NEW): Trend strength indicator - regime filter
  adx: {
    buyShares: 0.8,   // Moderate - prefer trending
    sellShares: 0.8,
    openCSP: 1.5,      // Important - prefer sideways (low ADX)
    openCC: 1.5,
    buyCall: 2.2,     // CRITICAL - need trend for directional bets
    buyPut: 2.2
  },
  
  // ============================================================================
  // PRICE/VOLUME CONFIRMATION
  // ============================================================================
  
  // VWAP: Fair value reference
  vwap: {
    buyShares: 0.8,
    sellShares: 0.8,
    openCSP: 1.0,
    openCC: 1.0,
    buyCall: 0.6,
    buyPut: 0.6
  },
  
  // Momentum: REDUCED weight due to multicollinearity with MACD
  // Kept for short-term velocity confirmation only
  momentum: {
    buyShares: 0.6,   // Reduced - MACD captures this
    sellShares: 0.6,
    openCSP: 0.3,      // Low - mean reversion strategy
    openCC: 0.3,
    buyCall: 1.0,     // Moderate - already captured by ADX
    buyPut: 1.0
  },
  
  // Volume: Price-direction correlated (improved interpretation)
  volume: {
    buyShares: 1.0,
    sellShares: 1.0,
    openCSP: 0.6,      // Lower - volume direction matters less for premium
    openCC: 0.6,
    buyCall: 1.8,     // HIGH - need volume conviction
    buyPut: 1.8
  },
  
  // RVOL: Relative volume for unusual activity
  rvol: {
    buyShares: 0.8,
    sellShares: 0.8,
    openCSP: 0.5,
    openCC: 0.5,
    buyCall: 1.5,     // Important - confirms participation
    buyPut: 1.5
  },
  
  // ============================================================================
  // STRUCTURAL SIGNALS
  // ============================================================================
  
  // Price Position: 52-week range context
  pricePosition: {
    buyShares: 1.0,   // Buy low in range
    sellShares: 1.0,
    openCSP: 1.5,      // Important - sell puts near lows
    openCC: 1.5,       // Important - sell calls near highs
    buyCall: 0.4,     // Low - breakouts happen at highs
    buyPut: 0.4
  },
  
  // SMA Alignment: Long-term trend structure
  smaAlignment: {
    buyShares: 1.8,   // HIGH - trend alignment critical
    sellShares: 1.5,
    openCSP: 0.8,      // Moderate
    openCC: 0.8,
    buyCall: 1.0,
    buyPut: 1.0
  },
  
  // Cross Patterns: Golden cross/death cross
  crossPattern: {
    buyShares: 1.5,   // Important for long-term
    sellShares: 1.5,
    openCSP: 0.6,
    openCC: 0.6,
    buyCall: 0.8,
    buyPut: 0.8
  },
  
  // ============================================================================
  // FUNDAMENTAL & SENTIMENT (External Data)
  // ============================================================================
  
  // ROIC: Company quality
  roic: {
    buyShares: 1.5,   // Important for holding
    sellShares: 0.5,
    openCSP: 1.2,      // Want quality for assignment risk
    openCC: 0.6,
    buyCall: 0.2,     // Short-term doesn't care
    buyPut: 0.2
  },
  
  // Call/Put Ratio: Options sentiment
  callPutRatio: {
    buyShares: 0.6,
    sellShares: 0.6,
    openCSP: 1.0,
    openCC: 1.0,
    buyCall: 1.2,
    buyPut: 1.2
  },
  
  // IV Percentile (NEW): Implied volatility regime
  // High IV = expensive options = sell premium
  // Low IV = cheap options = buy options
  ivPercentile: {
    buyShares: 0.2,   // Minor consideration
    sellShares: 0.2,
    openCSP: 2.0,      // CRITICAL - sell puts in high IV
    openCC: 2.0,       // CRITICAL - sell calls in high IV
    buyCall: 1.8,     // HIGH - buy calls in low IV
    buyPut: 1.8
  },
  
  // Sector Beta: Sector correlation
  sectorBeta: {
    buyShares: 1.0,
    sellShares: 0.8,
    openCSP: 0.6,
    openCC: 0.6,
    buyCall: 0.5,
    buyPut: 0.5
  },
  
  // ============================================================================
  // NEW: Advanced Predictive Metrics
  // ============================================================================
  
  // CMF (Chaikin Money Flow): Accumulation/Distribution detection
  // Tracks "smart money" behavior before price moves
  cmf: {
    buyShares: 1.8,   // HIGH - accumulation is bullish signal
    sellShares: 1.5,  // Important - distribution is sell signal
    openCSP: 1.2,     // Moderate - accumulation good for puts
    openCC: 1.0,
    buyCall: 2.0,     // CRITICAL - accumulation precedes breakouts
    buyPut: 2.0       // Distribution precedes breakdowns
  },
  
  // Divergence: Price-Volume divergence detection
  // Detects "fakeouts" where price and volume disagree
  divergence: {
    buyShares: 1.5,   // Important - avoid buying fakeouts
    sellShares: 1.5,
    openCSP: 1.0,     // Moderate consideration
    openCC: 1.0,
    buyCall: 2.2,     // CRITICAL - avoid buying calls on bearish divergence
    buyPut: 2.2
  },
  
  // Alpha: Relative strength vs market (SPY)
  // Leaders outperform, laggards underperform in corrections
  alpha: {
    buyShares: 1.5,   // Important - buy leaders not laggards
    sellShares: 1.2,
    openCSP: 0.8,     // Moderate
    openCC: 0.8,
    buyCall: 1.8,     // HIGH - need alpha for momentum plays
    buyPut: 1.8
  },
  
  // Earnings Proximity: Days until next earnings
  // Technical setups are unreliable near earnings
  earningsProximity: {
    buyShares: 0.8,   // Moderate penalty
    sellShares: 0.5,
    openCSP: 1.5,     // Important - avoid selling premium into earnings
    openCC: 1.5,
    buyCall: 2.0,     // CRITICAL - technicals fail at earnings
    buyPut: 2.0
  }
};

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
  earningsProximity: 'Earnings'
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
 * Interpret RSI for different actions.
 * 
 * Standard mean reversion interpretation:
 * RSI < 30: Oversold (bullish for buying)
 * RSI > 70: Overbought (bullish for selling)
 * 
 * REGIME-AWARE INTERPRETATION (Cardwell/Brown Theory):
 * The Momentum Override is now conditioned on ADX to distinguish:
 * - Trending Market (ADX > 25): RSI 65-85 is a "Bull Regime" - strength, not exhaustion
 * - Range-Bound Market (ADX < 20): RSI 70+ is a ceiling - mean reversion applies
 * 
 * NEW: HOOK LOGIC (Falling Knife Protection)
 * Never buy the STATE of being oversold; buy the EXIT from being oversold.
 * - RSI < 30 AND rising (hooking up) = Strong buy signal
 * - RSI < 30 AND falling = Falling knife, avoid!
 * 
 * This prevents false momentum signals in sideways markets and falling knives.
 */
function interpretRSI(
  indicators: TechnicalIndicators, 
  action: ActionType
): SignalResult | null {
  const rsi = indicators.rsi;
  if (!rsi) return null;
  
  const value = rsi.value;
  const prevValue = rsi.prevValue;
  const adx = indicators.adx;
  const isTrending = adx?.isTrending ?? false; // ADX > 25
  const isStrongTrend = adx?.isStrongTrend ?? false; // ADX > 40
  const trendDirection = adx?.direction ?? 'neutral';
  
  let signal = 0;
  let reasoning = '';
  
  // ============================================================================
  // HOOK LOGIC: Falling Knife Protection (NEW)
  // ============================================================================
  
  // For bullish actions, apply hook logic in oversold territory
  const bullishActions: ActionType[] = ['buyShares', 'openCSP', 'buyCall'];
  
  if (bullishActions.includes(action) && value <= 35 && prevValue !== null) {
    if (value > prevValue) {
      // RSI is turning UP from oversold - this is the signal to buy
      const hookStrength = Math.min((prevValue <= 30 ? 0.3 : 0.15), 1);
      signal = 0.7 + hookStrength;
      reasoning = `Oversold + Hooking Up (${prevValue.toFixed(0)}→${value.toFixed(0)})`;
      
      return {
        signal: Math.max(-1, Math.min(1, signal)),
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning
      };
    } else if (value < prevValue) {
      // RSI is still falling - FALLING KNIFE, drastically reduce signal
      signal = value <= 20 ? 0.1 : 0.2; // Very weak - don't catch knives
      reasoning = `Oversold but FALLING (${prevValue.toFixed(0)}→${value.toFixed(0)}) - Knife risk`;
      
      return {
        signal,
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning
      };
    }
  }
  
  // For bearish actions, apply inverse hook logic in overbought territory
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  
  if (bearishActions.includes(action) && value >= 65 && prevValue !== null) {
    if (value < prevValue) {
      // RSI is turning DOWN from overbought - this is the signal to sell/short
      const hookStrength = Math.min((prevValue >= 70 ? 0.3 : 0.15), 1);
      signal = 0.7 + hookStrength;
      reasoning = `Overbought + Hooking Down (${prevValue.toFixed(0)}→${value.toFixed(0)})`;
      
      return {
        signal: Math.max(-1, Math.min(1, signal)),
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning
      };
    } else if (value > prevValue) {
      // RSI still rising - don't short yet
      signal = value >= 80 ? 0.2 : 0.1;
      reasoning = `Overbought but RISING (${prevValue.toFixed(0)}→${value.toFixed(0)}) - Wait for turn`;
      
      return {
        signal,
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning
      };
    }
  }
  
  // ============================================================================
  // REGIME-AWARE MOMENTUM OVERRIDE (Cardwell/Brown Range Shift Theory)
  // ============================================================================
  
  // For Buy Calls: High RSI is favorable ONLY in trending markets
  if (action === 'buyCall') {
    if (value >= 65 && value < 85) {
      if (isTrending && trendDirection !== 'bearish') {
        // Trending market: RSI 65-85 is bullish (Bull Regime)
        const strengthBonus = isStrongTrend ? 0.15 : 0;
        return {
          signal: 0.8 + strengthBonus,
          rawValue: `RSI: ${value.toFixed(1)}`,
          reasoning: `Bull Regime (${value.toFixed(0)}) + ADX ${adx?.adx.toFixed(0)} trending`
        };
      } else {
        // Range-bound: RSI 65-85 is near resistance - caution
        return {
          signal: -0.2,
          rawValue: `RSI: ${value.toFixed(1)}`,
          reasoning: `RSI ${value.toFixed(0)} near ceiling (ADX ${adx?.adx?.toFixed(0) ?? '?'} = sideways)`
        };
      }
    } else if (value >= 85) {
      // Extreme overbought - caution regardless of trend
      return {
        signal: 0.1,
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning: `Extreme RSI (${value.toFixed(0)}) - Exhaustion risk`
      };
    }
  }
  
  // For Buy Puts: Low RSI is favorable ONLY in trending bear markets
  if (action === 'buyPut') {
    if (value <= 35 && value > 15) {
      if (isTrending && trendDirection !== 'bullish') {
        // Bear trend: RSI 15-35 is continuation zone
        const strengthBonus = isStrongTrend ? 0.15 : 0;
        return {
          signal: 0.8 + strengthBonus,
          rawValue: `RSI: ${value.toFixed(1)}`,
          reasoning: `Bear Regime (${value.toFixed(0)}) + ADX ${adx?.adx.toFixed(0)} trending`
        };
      } else {
        // Range-bound: RSI 15-35 is bounce zone - puts risky
        return {
          signal: -0.3,
          rawValue: `RSI: ${value.toFixed(1)}`,
          reasoning: `RSI ${value.toFixed(0)} near floor (ADX ${adx?.adx?.toFixed(0) ?? '?'} = bounce likely)`
        };
      }
    } else if (value <= 15) {
      return {
        signal: 0.1,
        rawValue: `RSI: ${value.toFixed(1)}`,
        reasoning: `Extreme oversold (${value.toFixed(0)}) - Bounce imminent`
      };
    }
  }
  
  // ============================================================================
  // STANDARD MEAN REVERSION INTERPRETATION (for non-momentum trades)
  // ============================================================================
  
  // Adjust interpretation based on market regime
  if (value <= 30) {
    signal = 1 - (value / 30) * 0.5; // 0→1, 30→0.5
    reasoning = `Oversold (${value.toFixed(0)})`;
    // In bear trend, oversold less bullish (could go lower)
    if (isTrending && trendDirection === 'bearish') {
      signal *= 0.6;
      reasoning += ' [bear trend caution]';
    }
  } else if (value <= 50) {
    signal = 0.5 - ((value - 30) / 20) * 0.5; // 30→0.5, 50→0
    reasoning = `Neutral-bullish (${value.toFixed(0)})`;
  } else if (value <= 70) {
    signal = -((value - 50) / 20) * 0.5; // 50→0, 70→-0.5
    reasoning = `Neutral-bearish (${value.toFixed(0)})`;
  } else {
    signal = -0.5 - ((value - 70) / 30) * 0.5; // 70→-0.5, 100→-1
    reasoning = `Overbought (${value.toFixed(0)})`;
    // In bull trend, overbought less bearish (can stay elevated)
    if (isTrending && trendDirection === 'bullish') {
      signal *= 0.6;
      reasoning += ' [bull trend support]';
    }
  }
  
  // Adjust signal direction based on action
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `RSI: ${value.toFixed(1)}`,
    reasoning
  };
}

/**
 * Interpret MACD for different actions.
 */
function interpretMACD(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const macd = indicators.macd;
  if (!macd) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // Histogram magnitude indicates strength
  const histogramStrength = Math.min(Math.abs(macd.histogram) / 2, 1);
  
  if (macd.isBullish) {
    signal = 0.3 + histogramStrength * 0.7;
    reasoning = macd.histogram > 0 
      ? `Bullish crossover, histogram +${macd.histogram.toFixed(2)}`
      : `Above signal, converging`;
  } else {
    signal = -(0.3 + histogramStrength * 0.7);
    reasoning = macd.histogram < 0
      ? `Bearish crossover, histogram ${macd.histogram.toFixed(2)}`
      : `Below signal, converging`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `MACD: ${macd.macdLine.toFixed(2)}`,
    reasoning
  };
}

/**
 * Interpret Bollinger Bands position.
 */
function interpretBollingerBands(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const bb = indicators.bollingerBands;
  if (!bb) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // percentB: 0 = at lower band (oversold), 1 = at upper band (overbought)
  const percentB = bb.percentB;
  
  if (percentB <= 0) {
    signal = 1;
    reasoning = 'Below lower band (oversold)';
  } else if (percentB <= 0.2) {
    signal = 0.8;
    reasoning = 'Near lower band';
  } else if (percentB <= 0.4) {
    signal = 0.4;
    reasoning = 'Lower half of bands';
  } else if (percentB <= 0.6) {
    signal = 0;
    reasoning = 'Middle of bands';
  } else if (percentB <= 0.8) {
    signal = -0.4;
    reasoning = 'Upper half of bands';
  } else if (percentB < 1) {
    signal = -0.8;
    reasoning = 'Near upper band';
  } else {
    signal = -1;
    reasoning = 'Above upper band (overbought)';
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal,
    rawValue: `%B: ${(percentB * 100).toFixed(0)}%`,
    reasoning
  };
}

/**
 * Interpret VWAP position.
 * 
 * Note: For daily data, this is a Rolling VWAP (anchor-style) rather than
 * true intraday VWAP. Still useful for support/resistance but interpret accordingly.
 */
function interpretVWAP(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const vwap = indicators.vwap;
  if (!vwap) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // Price vs VWAP percentage
  const deviation = vwap.priceVsVwap;
  
  // Indicate if this is intraday or rolling
  const vwapType = vwap.isIntraday ? '' : ' (rolling)';
  
  if (deviation < -3) {
    signal = 0.8;
    reasoning = `${Math.abs(deviation).toFixed(1)}% below VWAP${vwapType}`;
  } else if (deviation < -1) {
    signal = 0.4;
    reasoning = `Slightly below VWAP${vwapType}`;
  } else if (deviation <= 1) {
    signal = 0;
    reasoning = `At VWAP${vwapType}`;
  } else if (deviation <= 3) {
    signal = -0.4;
    reasoning = `Slightly above VWAP${vwapType}`;
  } else {
    signal = -0.8;
    reasoning = `${deviation.toFixed(1)}% above VWAP${vwapType}`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal,
    rawValue: `VWAP: ${vwap.value.toFixed(2)}`,
    reasoning
  };
}

/**
 * Interpret momentum.
 */
function interpretMomentum(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const momentum = indicators.momentum;
  if (!momentum) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // Combine short and medium term momentum
  const avgMomentum = (momentum.shortTerm + momentum.mediumTerm) / 2;
  
  if (momentum.trend === 'bullish') {
    signal = Math.min(avgMomentum / 10, 1) * 0.8;
    reasoning = `Bullish trend (+${avgMomentum.toFixed(1)}%)`;
  } else if (momentum.trend === 'bearish') {
    signal = Math.max(avgMomentum / 10, -1) * 0.8;
    reasoning = `Bearish trend (${avgMomentum.toFixed(1)}%)`;
  } else {
    signal = avgMomentum / 20; // Weaker signal for neutral trend
    reasoning = `Mixed momentum (${avgMomentum.toFixed(1)}%)`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `Mom: ${avgMomentum.toFixed(1)}%`,
    reasoning
  };
}

/**
 * Interpret volume patterns.
 * 
 * CORRECTED LOGIC: Volume acts as a MULTIPLIER (magnitude), not additive constant.
 * High volume on a down day = strong bearish signal
 * High volume on an up day = strong bullish signal
 * Low volume = weak conviction in either direction
 */
function interpretVolume(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const volume = indicators.volume;
  const momentum = indicators.momentum;
  if (!volume) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // Determine price direction from momentum
  // If no momentum data, fall back to neutral interpretation
  const isPriceUp = momentum ? momentum.shortTerm > 0 : null;
  
  // Base directional signal (-1 for down, +1 for up, 0 for unknown)
  let direction = 0;
  if (isPriceUp !== null) {
    direction = isPriceUp ? 1 : -1;
  }
  
  // Volume acts as confidence multiplier
  if (volume.isHighVolume) {
    if (direction !== 0) {
      // High volume strongly confirms the price direction
      signal = direction * 1.0;
      reasoning = `High volume confirms ${isPriceUp ? 'bullish' : 'bearish'} move (${volume.volumeRatio.toFixed(1)}x avg)`;
    } else {
      // High volume with unknown direction - just indicates activity
      signal = 0.3;
      reasoning = `High volume (${volume.volumeRatio.toFixed(1)}x avg)`;
    }
  } else if (volume.isLowVolume) {
    if (direction !== 0) {
      // Low volume suggests weak conviction
      signal = direction * 0.2;
      reasoning = `Low volume weakens ${isPriceUp ? 'bullish' : 'bearish'} move`;
    } else {
      signal = -0.1;
      reasoning = `Low volume - weak conviction`;
    }
  } else {
    // Normal volume provides moderate confirmation
    signal = direction * 0.5;
    reasoning = direction !== 0 
      ? `Normal volume ${isPriceUp ? 'bullish' : 'bearish'}` 
      : `Normal volume`;
  }
  
  // Volume trend can indicate accumulation/distribution
  if (volume.trend === 'increasing') {
    signal += direction * 0.15; // Increasing trend amplifies direction
    reasoning += ', trend increasing';
  } else if (volume.trend === 'decreasing') {
    signal -= direction * 0.1; // Decreasing trend weakens direction
    reasoning += ', trend decreasing';
  }
  
  // Invert signal for bearish actions
  // If signal is -1 (bearish w/ high vol), sellShares wants +1
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `Vol: ${volume.volumeRatio.toFixed(1)}x`,
    reasoning
  };
}

/**
 * Interpret price position vs 52-week range.
 */
function interpretPricePosition(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const pos = indicators.pricePosition;
  if (!pos) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // rangePosition: 0 = at 52w low, 1 = at 52w high
  const range = pos.rangePosition;
  
  if (range <= 0.2) {
    signal = 0.8;
    reasoning = `Near 52-week low (${(range * 100).toFixed(0)}% of range)`;
  } else if (range <= 0.4) {
    signal = 0.4;
    reasoning = `Lower range (${(range * 100).toFixed(0)}%)`;
  } else if (range <= 0.6) {
    signal = 0;
    reasoning = `Mid-range (${(range * 100).toFixed(0)}%)`;
  } else if (range <= 0.8) {
    signal = -0.4;
    reasoning = `Upper range (${(range * 100).toFixed(0)}%)`;
  } else {
    signal = -0.8;
    reasoning = `Near 52-week high (${(range * 100).toFixed(0)}%)`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal,
    rawValue: `Range: ${(range * 100).toFixed(0)}%`,
    reasoning
  };
}

/**
 * Interpret SMA alignment (price vs SMAs, SMA crossovers).
 */
function interpretSMAAlignment(
  indicators: TechnicalIndicators,
  currentPrice: number,
  action: ActionType
): SignalResult | null {
  const { sma20, sma50, sma200 } = indicators;
  
  // Need at least sma20 and sma50
  if (!sma20 || !sma50) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // Check price position relative to SMAs
  const aboveSma20 = currentPrice > sma20;
  const aboveSma50 = currentPrice > sma50;
  const aboveSma200 = sma200 ? currentPrice > sma200 : null;
  
  // Check SMA alignment (golden cross vs death cross)
  const sma20Above50 = sma20 > sma50;
  const sma50Above200 = sma200 ? sma50 > sma200 : null;
  
  // Score based on alignment
  let bullishPoints = 0;
  const factors: string[] = [];
  
  if (aboveSma20) { bullishPoints++; factors.push('> SMA20'); }
  if (aboveSma50) { bullishPoints++; factors.push('> SMA50'); }
  if (aboveSma200) { bullishPoints++; factors.push('> SMA200'); }
  if (sma20Above50) { bullishPoints++; factors.push('SMA20 > SMA50'); }
  if (sma50Above200) { bullishPoints++; factors.push('SMA50 > SMA200'); }
  
  const maxPoints = sma200 ? 5 : 3;
  signal = (bullishPoints / maxPoints) * 2 - 1; // Scale to -1 to +1
  
  if (bullishPoints >= maxPoints - 1) {
    reasoning = `Bullish alignment: ${factors.slice(0, 2).join(', ')}`;
  } else if (bullishPoints <= 1) {
    reasoning = `Bearish alignment: below key SMAs`;
  } else {
    reasoning = `Mixed SMA signals`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `SMA20: ${sma20.toFixed(2)}`,
    reasoning
  };
}

/**
 * Interpret RVOL (Relative Volume).
 * High RVOL confirms moves, very high RVOL suggests unusual activity.
 */
function interpretRVOL(
  indicators: TechnicalIndicators,
  _action: ActionType
): SignalResult | null {
  const rvol = indicators.rvol;
  if (!rvol) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // RVOL is action-agnostic - it indicates conviction, not direction
  // High RVOL is generally positive for any action (confirms the move)
  if (rvol.rvol > 3.0) {
    signal = 0.5;
    reasoning = `Very high volume (${rvol.rvol.toFixed(1)}x) - unusual activity`;
  } else if (rvol.rvol > 1.5) {
    signal = 0.3;
    reasoning = `Above avg volume (${rvol.rvol.toFixed(1)}x)`;
  } else if (rvol.rvol >= 0.8) {
    signal = 0;
    reasoning = `Normal volume (${rvol.rvol.toFixed(1)}x)`;
  } else if (rvol.rvol >= 0.5) {
    signal = -0.2;
    reasoning = `Low volume (${rvol.rvol.toFixed(1)}x)`;
  } else {
    signal = -0.4;
    reasoning = `Very low volume (${rvol.rvol.toFixed(1)}x) - lack of interest`;
  }
  
  return {
    signal,
    rawValue: `RVOL: ${rvol.rvol.toFixed(2)}x`,
    reasoning
  };
}

/**
 * Interpret Golden Cross / Death Cross patterns.
 */
function interpretCrossPattern(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const cross = indicators.crossPattern;
  if (!cross) return null;
  
  let signal = 0;
  let reasoning = cross.description;
  
  switch (cross.pattern) {
    case 'golden_cross':
      signal = cross.strength === 'strong' ? 0.9 : 0.6;
      break;
    case 'golden_star':
      signal = 1.0; // Strongest bullish signal
      break;
    case 'death_cross':
      signal = cross.strength === 'strong' ? -0.9 : -0.6;
      break;
    case 'none':
      if (cross.trendAlignment === 'bullish') {
        signal = 0.4;
      } else if (cross.trendAlignment === 'bearish') {
        signal = -0.4;
      }
      break;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: cross.pattern !== 'none' ? cross.pattern.replace('_', ' ') : cross.trendAlignment,
    reasoning
  };
}

/**
 * Interpret ADX (Average Directional Index) for trend strength.
 * 
 * ADX is used differently per action:
 * - Momentum trades (buyCall, buyPut): High ADX is good (trend continuation)
 * - Mean reversion (openCSP, openCC): Low ADX is good (range-bound)
 * - Long-term (buyShares): Moderate importance
 */
function interpretADX(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const adx = indicators.adx;
  if (!adx) return null;
  
  let signal = 0;
  let reasoning = '';
  
  const { adx: adxValue, direction, isTrending, isStrongTrend } = adx;
  
  // For momentum trades (long options), trending markets are favorable
  if (action === 'buyCall' || action === 'buyPut') {
    if (isStrongTrend) {
      // Strong trend - excellent for directional options
      const directionMatch = (action === 'buyCall' && direction === 'bullish') ||
                            (action === 'buyPut' && direction === 'bearish');
      signal = directionMatch ? 1.0 : 0.3;
      reasoning = `Strong ${direction} trend (ADX ${adxValue.toFixed(0)})`;
    } else if (isTrending) {
      const directionMatch = (action === 'buyCall' && direction === 'bullish') ||
                            (action === 'buyPut' && direction === 'bearish');
      signal = directionMatch ? 0.7 : 0.1;
      reasoning = `${direction} trend (ADX ${adxValue.toFixed(0)})`;
    } else {
      // No trend - bad for directional options (time decay hurts)
      signal = -0.6;
      reasoning = `Sideways market (ADX ${adxValue.toFixed(0)}) - avoid long options`;
    }
  }
  // For mean reversion trades (short options), range-bound is favorable
  else if (action === 'openCSP' || action === 'openCC') {
    if (!isTrending) {
      // Range-bound - ideal for selling options
      signal = 0.7;
      reasoning = `Range-bound (ADX ${adxValue.toFixed(0)}) - ideal for premium selling`;
    } else if (isStrongTrend) {
      // Strong trend - risky for short options
      signal = -0.5;
      reasoning = `Strong trend (ADX ${adxValue.toFixed(0)}) - assignment risk`;
    } else {
      signal = 0.1;
      reasoning = `Weak trend (ADX ${adxValue.toFixed(0)})`;
    }
  }
  // For buy/sell shares - moderate weighting
  else {
    if (direction === 'bullish' && isTrending) {
      signal = action === 'buyShares' ? 0.5 : -0.3;
      reasoning = `Bullish trend (ADX ${adxValue.toFixed(0)})`;
    } else if (direction === 'bearish' && isTrending) {
      signal = action === 'sellShares' ? 0.5 : -0.3;
      reasoning = `Bearish trend (ADX ${adxValue.toFixed(0)})`;
    } else {
      signal = 0;
      reasoning = `No clear trend (ADX ${adxValue.toFixed(0)})`;
    }
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `ADX: ${adxValue.toFixed(0)}`,
    reasoning
  };
}

/**
 * Interpret Bollinger Squeeze for volatility regime.
 * 
 * Volatility regime affects options strategies differently:
 * - Squeeze (low vol): Buy options cheap before expansion
 * - Expansion (high vol): Sell options at inflated premiums
 */
function interpretBollingerSqueeze(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const squeeze = indicators.bollingerSqueeze;
  if (!squeeze) return null;
  
  let signal = 0;
  let reasoning = '';
  
  const { bandwidthPercentile, isSqueeze, isExpansion, squeezeIntensity } = squeeze;
  
  // For long options (buyCall, buyPut): Squeeze is bullish (cheap premiums + expansion coming)
  if (action === 'buyCall' || action === 'buyPut') {
    if (isSqueeze) {
      const intensityBonus = squeezeIntensity === 'extreme' ? 0.3 : 
                            squeezeIntensity === 'moderate' ? 0.15 : 0;
      signal = 0.7 + intensityBonus;
      reasoning = `${squeezeIntensity} squeeze (${bandwidthPercentile}%ile) - cheap premiums`;
    } else if (isExpansion) {
      signal = -0.4;
      reasoning = `High volatility (${bandwidthPercentile}%ile) - expensive premiums`;
    } else {
      signal = 0;
      reasoning = `Normal volatility (${bandwidthPercentile}%ile)`;
    }
  }
  // For short options (openCSP, openCC): Expansion is bullish (sell expensive premiums)
  else if (action === 'openCSP' || action === 'openCC') {
    if (isExpansion) {
      signal = 0.6;
      reasoning = `High volatility (${bandwidthPercentile}%ile) - rich premiums`;
    } else if (isSqueeze) {
      signal = -0.3;
      reasoning = `${squeezeIntensity} squeeze (${bandwidthPercentile}%ile) - low premiums`;
    } else {
      signal = 0.1;
      reasoning = `Normal volatility (${bandwidthPercentile}%ile)`;
    }
  }
  // For shares - less important, but high vol = cheaper entries possible
  else {
    if (isSqueeze) {
      signal = 0.2;
      reasoning = `Low volatility - stable entry`;
    } else if (isExpansion) {
      signal = action === 'buyShares' ? 0.3 : 0.1; // Volatility can mean opportunity
      reasoning = `High volatility - watch for entry`;
    } else {
      signal = 0;
      reasoning = `Normal volatility`;
    }
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `BWP: ${bandwidthPercentile}%`,
    reasoning
  };
}

// ============================================================================
// Additional Data Interface (Fundamentals, Options, Sector)
// ============================================================================

export interface AdditionalAnalysisData {
  // Fundamentals
  roic?: number | null;           // Return on Invested Capital (%)
  roe?: number | null;            // Return on Equity (%)
  sector?: string | null;
  industry?: string | null;
  beta?: number | null;
  
  // Options
  callPutRatioOI?: number | null;     // Call/Put ratio by open interest
  callPutRatioVolume?: number | null; // Call/Put ratio by volume
  optionsSentiment?: 'bullish' | 'bearish' | 'neutral' | null;
  avgImpliedVolatility?: number | null;
  ivPercentile?: number | null;
  hasOptions?: boolean;
  
  // Sector correlation
  sectorCorrelation?: number | null;  // -1 to 1
  betaToSector?: number | null;
  
  // NEW: Alpha / Relative Strength
  alpha?: number | null;              // Stock % change - SPY % change (relative strength)
  spyChange?: number | null;          // SPY % change for context
  
  // NEW: Earnings Proximity
  daysToEarnings?: number | null;     // Days until next earnings announcement
  nextEarningsDate?: string | null;   // ISO date of next earnings
}

/**
 * Interpret ROIC (Return on Invested Capital).
 * High ROIC indicates efficient capital allocation - good for long-term holds.
 */
function interpretROIC(
  additionalData: AdditionalAnalysisData | undefined,
  action: ActionType
): SignalResult | null {
  if (!additionalData?.roic) return null;
  
  const roic = additionalData.roic;
  let signal = 0;
  let reasoning = '';
  
  // ROIC interpretation:
  // > 20%: Excellent (strong moat, efficient)
  // 15-20%: Good
  // 10-15%: Average
  // 5-10%: Below average
  // < 5%: Poor
  if (roic > 20) {
    signal = 0.8;
    reasoning = `Excellent ROIC (${roic.toFixed(1)}%) - efficient capital use`;
  } else if (roic > 15) {
    signal = 0.5;
    reasoning = `Good ROIC (${roic.toFixed(1)}%)`;
  } else if (roic > 10) {
    signal = 0.2;
    reasoning = `Average ROIC (${roic.toFixed(1)}%)`;
  } else if (roic > 5) {
    signal = -0.2;
    reasoning = `Below avg ROIC (${roic.toFixed(1)}%)`;
  } else if (roic > 0) {
    signal = -0.5;
    reasoning = `Poor ROIC (${roic.toFixed(1)}%)`;
  } else {
    signal = -0.8;
    reasoning = `Negative ROIC (${roic.toFixed(1)}%) - capital destruction`;
  }
  
  // ROIC is most relevant for buying shares (long-term investment)
  // Less relevant for short-term options
  if (action === 'buyCall' || action === 'buyPut') {
    signal *= 0.5; // Reduce impact for options
  }
  
  // Invert for sell actions (good company = less reason to sell)
  const bearishActions: ActionType[] = ['sellShares'];
  if (bearishActions.includes(action)) {
    signal = -signal * 0.5; // Reduced impact
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `ROIC: ${roic.toFixed(1)}%`,
    reasoning
  };
}

/**
 * Interpret Call/Put ratio from options market.
 * High C/P ratio (>1.5) = bullish sentiment
 * Low C/P ratio (<0.7) = bearish sentiment
 */
function interpretCallPutRatio(
  additionalData: AdditionalAnalysisData | undefined,
  action: ActionType
): SignalResult | null {
  if (!additionalData?.hasOptions) return null;
  
  const ratio = additionalData.callPutRatioOI ?? additionalData.callPutRatioVolume;
  if (!ratio) return null;
  
  let signal = 0;
  let reasoning = '';
  
  // C/P ratio interpretation:
  // > 4.0: Extremely bullish (high predictor of upward move)
  // > 2.0: Very bullish
  // > 1.5: Bullish
  // 0.8-1.5: Neutral
  // 0.5-0.8: Bearish
  // < 0.5: Very bearish
  if (ratio > 4.0) {
    signal = 1.0;
    reasoning = `Extremely bullish C/P ratio (${ratio.toFixed(2)})`;
  } else if (ratio > 2.0) {
    signal = 0.8;
    reasoning = `Very bullish C/P ratio (${ratio.toFixed(2)})`;
  } else if (ratio > 1.5) {
    signal = 0.5;
    reasoning = `Bullish C/P ratio (${ratio.toFixed(2)})`;
  } else if (ratio >= 0.8) {
    signal = 0;
    reasoning = `Neutral C/P ratio (${ratio.toFixed(2)})`;
  } else if (ratio >= 0.5) {
    signal = -0.5;
    reasoning = `Bearish C/P ratio (${ratio.toFixed(2)})`;
  } else {
    signal = -0.8;
    reasoning = `Very bearish C/P ratio (${ratio.toFixed(2)})`;
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `C/P: ${ratio.toFixed(2)}`,
    reasoning
  };
}

/**
 * Interpret IV Percentile for options strategy selection.
 * 
 * IV Percentile indicates where current IV sits relative to its historical range:
 * - High IV (>70%): Options expensive → Favor selling premium (CSP, CC)
 * - Low IV (<30%): Options cheap → Favor buying options (Calls, Puts)
 * 
 * This is CRITICAL for options strategy profitability.
 */
function interpretIVPercentile(
  additionalData: AdditionalAnalysisData | undefined,
  action: ActionType
): SignalResult | null {
  if (!additionalData?.ivPercentile) return null;
  
  const ivPct = additionalData.ivPercentile;
  let signal = 0;
  let reasoning = '';
  
  // For SELLING options (CSP, CC): High IV is favorable
  if (action === 'openCSP' || action === 'openCC') {
    if (ivPct >= 80) {
      signal = 1.0;
      reasoning = `Very high IV (${ivPct}%ile) - rich premiums`;
    } else if (ivPct >= 60) {
      signal = 0.6;
      reasoning = `Elevated IV (${ivPct}%ile) - good premiums`;
    } else if (ivPct >= 40) {
      signal = 0;
      reasoning = `Normal IV (${ivPct}%ile)`;
    } else if (ivPct >= 20) {
      signal = -0.4;
      reasoning = `Low IV (${ivPct}%ile) - thin premiums`;
    } else {
      signal = -0.8;
      reasoning = `Very low IV (${ivPct}%ile) - poor premium environment`;
    }
  }
  // For BUYING options (Calls, Puts): Low IV is favorable
  else if (action === 'buyCall' || action === 'buyPut') {
    if (ivPct <= 20) {
      signal = 1.0;
      reasoning = `Very low IV (${ivPct}%ile) - cheap options`;
    } else if (ivPct <= 40) {
      signal = 0.6;
      reasoning = `Low IV (${ivPct}%ile) - reasonable premiums`;
    } else if (ivPct <= 60) {
      signal = 0;
      reasoning = `Normal IV (${ivPct}%ile)`;
    } else if (ivPct <= 80) {
      signal = -0.4;
      reasoning = `Elevated IV (${ivPct}%ile) - expensive`;
    } else {
      signal = -0.8;
      reasoning = `Very high IV (${ivPct}%ile) - overpaying for vol`;
    }
  }
  // For shares - IV is minor factor
  else {
    signal = 0;
    reasoning = `IV ${ivPct}%ile (not applicable)`;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `IV: ${ivPct}%ile`,
    reasoning
  };
}

/**
 * Interpret sector beta (correlation and beta to sector ETF).
 * High correlation with sector during stability is normal.
 * Divergence from sector can indicate specific catalysts.
 */
function interpretSectorBeta(
  additionalData: AdditionalAnalysisData | undefined,
  _action: ActionType
): SignalResult | null {
  if (!additionalData?.betaToSector && !additionalData?.sectorCorrelation) return null;
  
  const beta = additionalData.betaToSector;
  const correlation = additionalData.sectorCorrelation;
  
  let signal = 0;
  let reasoning = '';
  
  if (beta !== null && beta !== undefined) {
    // Beta interpretation:
    // > 1.5: High beta (more volatile than sector)
    // 1.0-1.5: Normal-high
    // 0.8-1.0: Tracks sector closely
    // 0.5-0.8: Less volatile
    // < 0.5: Low correlation with sector
    if (beta > 1.5) {
      signal = 0.2; // Higher risk/reward
      reasoning = `High sector beta (${beta.toFixed(2)}) - amplified moves`;
    } else if (beta >= 1.0) {
      signal = 0.1;
      reasoning = `Normal sector beta (${beta.toFixed(2)})`;
    } else if (beta >= 0.8) {
      signal = 0;
      reasoning = `Tracks sector (β=${beta.toFixed(2)})`;
    } else if (beta >= 0.5) {
      signal = -0.1;
      reasoning = `Lower sector beta (${beta.toFixed(2)}) - defensive`;
    } else {
      signal = 0;
      reasoning = `Low sector correlation (β=${beta.toFixed(2)})`;
    }
  } else if (correlation !== null && correlation !== undefined) {
    signal = correlation > 0.7 ? 0.1 : correlation < 0.3 ? -0.1 : 0;
    reasoning = `Sector corr: ${(correlation * 100).toFixed(0)}%`;
  }
  
  return {
    signal,
    rawValue: beta ? `β: ${beta.toFixed(2)}` : `Corr: ${((correlation ?? 0) * 100).toFixed(0)}%`,
    reasoning
  };
}

/**
 * Interpret Chaikin Money Flow (CMF) for accumulation/distribution.
 * 
 * CMF is a LEADING indicator that detects institutional buying/selling
 * BEFORE price moves manifest. This is the "smart money" signal.
 * 
 * - CMF > 0.25: Strong accumulation (institutions aggressively buying)
 * - CMF > 0.1: Accumulation (buying pressure)
 * - CMF -0.1 to 0.1: Neutral
 * - CMF < -0.1: Distribution (selling pressure)
 * - CMF < -0.25: Strong distribution (institutions aggressively selling)
 */
function interpretCMF(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const cmf = indicators.cmf;
  if (!cmf) return null;
  
  let signal = 0;
  let reasoning = '';
  
  const { value, interpretation, volumeStrength, closeLocation } = cmf;
  
  // Base signal from CMF value
  if (interpretation === 'strong_accumulation') {
    signal = 1.0;
    reasoning = `Strong accumulation (${(value * 100).toFixed(0)}%)`;
  } else if (interpretation === 'accumulation') {
    signal = 0.6;
    reasoning = `Accumulation (${(value * 100).toFixed(0)}%)`;
  } else if (interpretation === 'neutral') {
    signal = value * 2; // Slight lean based on value
    reasoning = `Neutral money flow (${(value * 100).toFixed(0)}%)`;
  } else if (interpretation === 'distribution') {
    signal = -0.6;
    reasoning = `Distribution (${(value * 100).toFixed(0)}%)`;
  } else {
    signal = -1.0;
    reasoning = `Strong distribution (${(value * 100).toFixed(0)}%)`;
  }
  
  // Volume strength modifier
  if (volumeStrength === 'high') {
    signal *= 1.2; // High volume confirms the signal
    reasoning += ', high volume';
  } else if (volumeStrength === 'low') {
    signal *= 0.6; // Low volume weakens the signal
    reasoning += ', low volume';
  }
  
  // Close location provides additional context
  if (closeLocation > 0.8 && value > 0) {
    reasoning += ', closed near high';
  } else if (closeLocation < 0.2 && value < 0) {
    reasoning += ', closed near low';
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `CMF: ${(value * 100).toFixed(0)}%`,
    reasoning
  };
}

/**
 * Interpret Price-Volume Divergence.
 * 
 * Divergence is one of the MOST powerful predictive signals.
 * It detects "fakeouts" where price action and volume disagree.
 * 
 * BEARISH DIVERGENCE (Fakeout Warning):
 * - Price makes new high, but volume is LOWER than previous high
 * - Smart money is NOT participating in the new high
 * - High probability of reversal
 * 
 * BULLISH DIVERGENCE (Reversal Signal):
 * - Price makes new low, but volume is LOWER than previous low
 * - Selling exhaustion - bears are running out of ammunition
 */
function interpretDivergence(
  indicators: TechnicalIndicators,
  action: ActionType
): SignalResult | null {
  const divergence = indicators.divergence;
  if (!divergence) return null;
  
  let signal = 0;
  let reasoning = divergence.description;
  
  const { hasBullishDivergence, hasBearishDivergence, divergenceStrength } = divergence;
  
  // Divergence signals are CONTRA-INDICATORS
  // Bearish divergence = bearish for bullish actions
  // Bullish divergence = bullish for bullish actions
  
  if (hasBearishDivergence) {
    // Price high on low volume = FAKEOUT WARNING
    const strength = divergenceStrength === 'strong' ? -0.9 : -0.6;
    signal = strength;
  } else if (hasBullishDivergence) {
    // Price low on low volume = selling exhaustion
    const strength = divergenceStrength === 'strong' ? 0.8 : 0.5;
    signal = strength;
  } else {
    // No divergence - neutral
    signal = 0;
    reasoning = 'No divergence';
  }
  
  // Invert for bearish actions
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: divergence.volumeDivergence !== 'none' 
      ? `${divergence.volumeDivergence} div.` 
      : 'No div.',
    reasoning
  };
}

/**
 * Interpret Alpha (Relative Strength vs Market).
 * 
 * A stock rising 1% on a day the SPY rises 3% is actually showing WEAKNESS.
 * It is a "laggard" that will likely drop first in a correction.
 * 
 * Alpha = Stock % Change - SPY % Change
 * - Positive alpha: Stock is a LEADER (outperforming market)
 * - Negative alpha: Stock is a LAGGARD (underperforming market)
 * 
 * You want to own leaders, not laggards.
 */
function interpretAlpha(
  additionalData: AdditionalAnalysisData | undefined,
  action: ActionType
): SignalResult | null {
  if (additionalData?.alpha === undefined || additionalData?.alpha === null) return null;
  
  const alpha = additionalData.alpha;
  let signal = 0;
  let reasoning = '';
  
  // Alpha interpretation:
  // > 5%: Strong outperformance (leader)
  // 2-5%: Moderate outperformance
  // -2% to 2%: In line with market
  // -5% to -2%: Moderate underperformance
  // < -5%: Strong underperformance (laggard)
  
  if (alpha > 5) {
    signal = 1.0;
    reasoning = `Strong leader (+${alpha.toFixed(1)}% vs SPY)`;
  } else if (alpha > 2) {
    signal = 0.6;
    reasoning = `Outperforming (+${alpha.toFixed(1)}% vs SPY)`;
  } else if (alpha >= -2) {
    signal = alpha / 5; // Slight lean based on alpha
    reasoning = `In line with market (${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}% vs SPY)`;
  } else if (alpha >= -5) {
    signal = -0.5;
    reasoning = `Underperforming (${alpha.toFixed(1)}% vs SPY)`;
  } else {
    signal = -0.9;
    reasoning = `Laggard (${alpha.toFixed(1)}% vs SPY)`;
  }
  
  // Invert for bearish actions (laggards are good short candidates)
  const bearishActions: ActionType[] = ['sellShares', 'openCC', 'buyPut'];
  if (bearishActions.includes(action)) {
    signal = -signal;
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: `α: ${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}%`,
    reasoning
  };
}

/**
 * Interpret Earnings Proximity.
 * 
 * Technical setups are UNRELIABLE near earnings announcements.
 * A "perfect" technical setup can be completely invalidated by earnings.
 * 
 * This is an EARNINGS BLACKOUT signal:
 * - < 3 days to earnings: CRITICAL penalty (technicals unreliable)
 * - 3-7 days: Moderate penalty
 * - 7-14 days: Slight caution
 * - > 14 days: No penalty
 * 
 * Especially important for options (IV crush risk).
 */
function interpretEarningsProximity(
  additionalData: AdditionalAnalysisData | undefined,
  action: ActionType
): SignalResult | null {
  if (additionalData?.daysToEarnings === undefined || additionalData?.daysToEarnings === null) {
    return null;
  }
  
  const days = additionalData.daysToEarnings;
  let signal = 0;
  let reasoning = '';
  
  // Earnings blackout zones
  if (days <= 0) {
    // Earnings today or already passed - could be post-earnings move
    signal = -0.3;
    reasoning = `Earnings ${days === 0 ? 'today' : 'passed'} - IV crush risk`;
  } else if (days <= 3) {
    // DANGER ZONE: Technicals are unreliable
    signal = -0.8;
    reasoning = `Earnings in ${days} day${days === 1 ? '' : 's'} - technicals unreliable`;
  } else if (days <= 7) {
    // Caution zone
    signal = -0.4;
    reasoning = `Earnings in ${days} days - elevated uncertainty`;
  } else if (days <= 14) {
    // Slight caution
    signal = -0.1;
    reasoning = `Earnings in ${days} days`;
  } else {
    // Clear of earnings
    signal = 0.1;
    reasoning = `Earnings ${days}+ days away - technicals reliable`;
  }
  
  // Options are MORE sensitive to earnings (IV crush)
  const optionActions: ActionType[] = ['openCSP', 'openCC', 'buyCall', 'buyPut'];
  if (optionActions.includes(action)) {
    if (days <= 3) {
      signal = -1.0; // Maximum penalty for options near earnings
      reasoning += ' (IV crush risk)';
    } else if (days <= 7) {
      signal *= 1.5; // Amplify penalty
    }
  }
  
  return {
    signal: Math.max(-1, Math.min(1, signal)),
    rawValue: days <= 0 ? 'Earnings passed' : `${days}d to ER`,
    reasoning
  };
}

// ============================================================================
// Main Scoring Functions
// ============================================================================

const ALL_ACTIONS: ActionType[] = [
  'buyShares', 'sellShares', 'openCSP', 'openCC', 'buyCall', 'buyPut'
];

const ALL_METRICS: MetricType[] = [
  'rsi', 'macd', 'bollingerBands', 'bollingerSqueeze', 'vwap', 'momentum', 
  'volume', 'pricePosition', 'smaAlignment', 'rvol', 'adx', 'crossPattern',
  'cmf', 'divergence', 'alpha',  // NEW: Leading indicators
  'roic', 'callPutRatio', 'ivPercentile', 'sectorBeta', 'earningsProximity'
];

/**
 * Get signal for a specific metric and action.
 */
function getSignalForMetric(
  metric: MetricType,
  indicators: TechnicalIndicators,
  currentPrice: number,
  action: ActionType,
  additionalData?: AdditionalAnalysisData
): SignalResult | null {
  switch (metric) {
    case 'rsi':
      return interpretRSI(indicators, action);
    case 'macd':
      return interpretMACD(indicators, action);
    case 'bollingerBands':
      return interpretBollingerBands(indicators, action);
    case 'bollingerSqueeze':
      return interpretBollingerSqueeze(indicators, action);
    case 'vwap':
      return interpretVWAP(indicators, action);
    case 'momentum':
      return interpretMomentum(indicators, action);
    case 'volume':
      return interpretVolume(indicators, action);
    case 'pricePosition':
      return interpretPricePosition(indicators, action);
    case 'smaAlignment':
      return interpretSMAAlignment(indicators, currentPrice, action);
    case 'rvol':
      return interpretRVOL(indicators, action);
    case 'adx':
      return interpretADX(indicators, action);
    case 'crossPattern':
      return interpretCrossPattern(indicators, action);
    // NEW: Leading indicators
    case 'cmf':
      return interpretCMF(indicators, action);
    case 'divergence':
      return interpretDivergence(indicators, action);
    case 'alpha':
      return interpretAlpha(additionalData, action);
    // External data
    case 'roic':
      return interpretROIC(additionalData, action);
    case 'callPutRatio':
      return interpretCallPutRatio(additionalData, action);
    case 'ivPercentile':
      return interpretIVPercentile(additionalData, action);
    case 'sectorBeta':
      return interpretSectorBeta(additionalData, action);
    case 'earningsProximity':
      return interpretEarningsProximity(additionalData, action);
    default:
      return null;
  }
}

/**
 * Detect the trading regime context for buyShares action.
 * 
 * This resolves the "Identity Crisis" between Value and Trend strategies.
 * - TREND mode: Price above SMA50, focus on momentum continuation
 * - VALUE mode: Price below SMA50, focus on mean reversion
 * 
 * In VALUE mode, we REDUCE the weight of SMA alignment (broken trend is expected)
 * and INCREASE the weight of RSI/Bollinger (oversold conditions).
 */
function detectBuyingContext(
  indicators: TechnicalIndicators,
  currentPrice: number
): 'trend' | 'value' | 'neutral' {
  const sma50 = indicators.sma50;
  const sma200 = indicators.sma200;
  const bb = indicators.bollingerBands;
  const rsi = indicators.rsi;
  
  if (!sma50) return 'neutral';
  
  // Check for VALUE context (dip buying opportunity)
  // Price is below SMA50, potentially oversold
  const priceBelowSma50 = currentPrice < sma50;
  const priceNearLowerBand = bb ? bb.percentB < 0.3 : false;
  const isOversold = rsi ? rsi.value < 40 : false;
  
  // Check for TREND context (momentum following)
  const priceAboveSma50 = currentPrice > sma50;
  const priceAboveSma200 = sma200 ? currentPrice > sma200 : priceAboveSma50;
  
  // If price is below SMA50 AND (near lower band OR oversold) = VALUE mode
  if (priceBelowSma50 && (priceNearLowerBand || isOversold)) {
    return 'value';
  }
  
  // If price is above both SMAs = TREND mode
  if (priceAboveSma50 && priceAboveSma200) {
    return 'trend';
  }
  
  return 'neutral';
}

/**
 * Get context-adjusted weights for buyShares action.
 * 
 * In VALUE mode (dip buying), we don't want the broken trend to kill the score.
 * Instead, we focus on oversold conditions and mean reversion signals.
 */
function getContextAdjustedWeight(
  metric: MetricType,
  action: ActionType,
  context: 'trend' | 'value' | 'neutral',
  baseWeight: number
): number {
  if (action !== 'buyShares') {
    return baseWeight;
  }
  
  if (context === 'value') {
    // VALUE mode: Dip buying - ignore broken trend, focus on oversold
    switch (metric) {
      case 'smaAlignment':
        return baseWeight * 0.2; // Drastically reduce - broken trend is expected
      case 'crossPattern':
        return baseWeight * 0.3; // Death cross is expected in a dip
      case 'rsi':
        return baseWeight * 2.0; // Heavily weight oversold conditions
      case 'bollingerBands':
        return baseWeight * 1.8; // Near lower band is good
      case 'pricePosition':
        return baseWeight * 1.5; // Low in range is the point
      case 'cmf':
        return baseWeight * 2.0; // Accumulation during dip is key signal
      case 'divergence':
        return baseWeight * 1.8; // Bullish divergence is key for dip buying
      default:
        return baseWeight;
    }
  }
  
  if (context === 'trend') {
    // TREND mode: Momentum following - standard weights apply
    // Slight boost to trend-following indicators
    switch (metric) {
      case 'smaAlignment':
        return baseWeight * 1.2;
      case 'macd':
        return baseWeight * 1.2;
      case 'adx':
        return baseWeight * 1.3;
      case 'pricePosition':
        return baseWeight * 0.7; // High in range is OK in uptrend
      default:
        return baseWeight;
    }
  }
  
  return baseWeight;
}

/**
 * Calculate score for a single action.
 */
function calculateActionScore(
  action: ActionType,
  indicators: TechnicalIndicators,
  currentPrice: number,
  weights: WeightMatrix = DEFAULT_WEIGHTS,
  additionalData?: AdditionalAnalysisData
): ActionScore {
  const signals: MetricSignal[] = [];
  let rawScore = 0;
  let availableMetrics = 0;
  
  // Detect context for buyShares to resolve Value vs Trend identity crisis
  const buyingContext = action === 'buyShares' 
    ? detectBuyingContext(indicators, currentPrice)
    : 'neutral';
  
  for (const metric of ALL_METRICS) {
    const signalResult = getSignalForMetric(metric, indicators, currentPrice, action, additionalData);
    
    if (signalResult) {
      // Get context-adjusted weight
      const baseWeight = weights[metric][action];
      const weight = getContextAdjustedWeight(metric, action, buyingContext, baseWeight);
      const contribution = signalResult.signal * weight;
      
      signals.push({
        metric,
        metricLabel: metricLabels[metric],
        rawValue: signalResult.rawValue,
        signal: signalResult.signal,
        weight,
        contribution,
        reasoning: signalResult.reasoning + (
          buyingContext !== 'neutral' && action === 'buyShares' && weight !== baseWeight
            ? ` [${buyingContext} mode]`
            : ''
        )
      });
      
      rawScore += contribution;
      availableMetrics++;
    }
  }
  
  // Normalize score to 0-100 scale
  // Raw score range is approximately -8 to +8 (8 metrics, each -1 to +1)
  // We map this to 0-100
  const normalizedScore = Math.round(((rawScore / (availableMetrics || 1)) + 1) * 50);
  const totalScore = Math.max(0, Math.min(100, normalizedScore));
  
  // Determine confidence based on signal agreement and strength
  // This is now per-action, not just based on data availability
  let confidence: 'high' | 'medium' | 'low';
  
  if (availableMetrics < 4) {
    // Not enough data for any confidence
    confidence = 'low';
  } else {
    // Calculate signal agreement: how many signals point in the same direction as the total?
    const overallDirection = rawScore > 0 ? 1 : rawScore < 0 ? -1 : 0;
    let agreeingSignals = 0;
    let strongSignals = 0; // Signals with |contribution| > 0.3
    
    for (const sig of signals) {
      const sigDirection = sig.contribution > 0 ? 1 : sig.contribution < 0 ? -1 : 0;
      if (sigDirection === overallDirection && overallDirection !== 0) {
        agreeingSignals++;
      }
      if (Math.abs(sig.contribution) > 0.3) {
        strongSignals++;
      }
    }
    
    const agreementRatio = availableMetrics > 0 ? agreeingSignals / availableMetrics : 0;
    const strongRatio = availableMetrics > 0 ? strongSignals / availableMetrics : 0;
    
    // High confidence: >70% signals agree AND >40% are strong signals AND score is decisive (>60 or <40)
    // Medium confidence: >50% signals agree OR score is somewhat decisive
    // Low confidence: signals are mixed or weak
    const isDecisiveScore = totalScore >= 65 || totalScore <= 35;
    const isSomewhatDecisive = totalScore >= 55 || totalScore <= 45;
    
    if (agreementRatio >= 0.7 && strongRatio >= 0.4 && isDecisiveScore) {
      confidence = 'high';
    } else if ((agreementRatio >= 0.5 && isSomewhatDecisive) || (strongRatio >= 0.5)) {
      confidence = 'medium';
    } else {
      confidence = 'low';
    }
  }
  
  return {
    action,
    label: actionLabels[action],
    totalScore,
    rawScore,
    signals,
    confidence
  };
}

/**
 * Calculate scores for all actions and produce complete analysis.
 */
export function analyzeStock(
  ticker: string,
  history: HistoryPoint[],
  currentPrice: number,
  high52w: number | null,
  low52w: number | null,
  weights: WeightMatrix = DEFAULT_WEIGHTS,
  additionalData?: AdditionalAnalysisData
): StockAnalysis {
  // Calculate all technical indicators
  const indicators = calculateAllIndicators(history, currentPrice, high52w, low52w);
  
  // Determine if options are available for this stock
  // Default to TRUE (most stocks have options) - only exclude if explicitly set to false
  const hasOptions = additionalData?.hasOptions !== false;
  
  // Calculate scores for all actions
  const scores = ALL_ACTIONS.map(action => 
    calculateActionScore(action, indicators, currentPrice, weights, additionalData)
  );
  
  // Options-related actions
  const optionsActions: ActionType[] = ['openCSP', 'openCC', 'buyCall', 'buyPut'];
  
  // Filter to eligible actions (exclude options if not available)
  const eligibleScores = hasOptions 
    ? scores 
    : scores.filter(s => !optionsActions.includes(s.action));
  
  // Find best action among eligible actions
  const bestAction = eligibleScores.reduce((best, current) => 
    current.totalScore > best.totalScore ? current : best
  );
  
  // Determine which metrics are missing
  const missingMetrics: MetricType[] = [];
  if (!indicators.rsi) missingMetrics.push('rsi');
  if (!indicators.macd) missingMetrics.push('macd');
  if (!indicators.bollingerBands) missingMetrics.push('bollingerBands');
  if (!indicators.bollingerSqueeze) missingMetrics.push('bollingerSqueeze');
  if (!indicators.vwap) missingMetrics.push('vwap');
  if (!indicators.momentum) missingMetrics.push('momentum');
  if (!indicators.volume) missingMetrics.push('volume');
  if (!indicators.pricePosition) missingMetrics.push('pricePosition');
  if (!indicators.sma20 || !indicators.sma50) missingMetrics.push('smaAlignment');
  if (!indicators.rvol) missingMetrics.push('rvol');
  if (!indicators.adx) missingMetrics.push('adx');
  if (!indicators.crossPattern) missingMetrics.push('crossPattern');
  // NEW: Leading indicators
  if (!indicators.cmf) missingMetrics.push('cmf');
  if (!indicators.divergence) missingMetrics.push('divergence');
  if (additionalData?.alpha === undefined || additionalData?.alpha === null) {
    missingMetrics.push('alpha');
  }
  // External data
  if (!additionalData?.roic) missingMetrics.push('roic');
  if (!additionalData?.hasOptions) missingMetrics.push('callPutRatio');
  if (additionalData?.ivPercentile === undefined || additionalData?.ivPercentile === null) {
    missingMetrics.push('ivPercentile');
  }
  if (!additionalData?.betaToSector && !additionalData?.sectorCorrelation) missingMetrics.push('sectorBeta');
  if (additionalData?.daysToEarnings === undefined || additionalData?.daysToEarnings === null) {
    missingMetrics.push('earningsProximity');
  }
  
  return {
    ticker,
    analyzedAt: new Date(),
    scores,
    bestAction,
    hasOptions,
    dataQuality: {
      availableMetrics: ALL_METRICS.length - missingMetrics.length,
      totalMetrics: ALL_METRICS.length,
      missingMetrics,
      historyDays: history.length,
      hasSMA200: indicators.sma200 !== null
    }
  };
}

/**
 * Get a quick summary of the best action without full analysis.
 * Useful for badge display on holding cards.
 */
export function getQuickAnalysis(
  history: HistoryPoint[],
  currentPrice: number,
  high52w: number | null,
  low52w: number | null
): { action: ActionType; label: string; score: number; confidence: 'high' | 'medium' | 'low' } | null {
  if (!history.length || !currentPrice) return null;
  
  try {
    const analysis = analyzeStock('', history, currentPrice, high52w, low52w);
    return {
      action: analysis.bestAction.action,
      label: analysis.bestAction.label,
      score: analysis.bestAction.totalScore,
      confidence: analysis.bestAction.confidence
    };
  } catch {
    return null;
  }
}

// Export for external configuration
export { DEFAULT_WEIGHTS, actionLabels, metricLabels };
export type { WeightMatrix };

