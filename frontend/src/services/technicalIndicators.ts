/**
 * Technical Indicators Calculator
 * 
 * Pure calculation functions for various technical analysis indicators.
 * All functions take price/volume data and return indicator values.
 */

import type { HistoryPoint } from '../types';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Extract closing prices from history data.
 */
function getClosePrices(history: HistoryPoint[]): number[] {
  return history.map(h => h.close).filter(p => p > 0);
}

/**
 * Calculate Simple Moving Average for a given period.
 */
export function calculateSMA(prices: number[], period: number): number | null {
  if (prices.length < period) return null;
  const slice = prices.slice(-period);
  return slice.reduce((sum, p) => sum + p, 0) / period;
}

/**
 * Calculate Exponential Moving Average.
 * Uses smoothing factor: 2 / (period + 1)
 */
export function calculateEMA(prices: number[], period: number): number | null {
  if (prices.length < period) return null;
  
  const multiplier = 2 / (period + 1);
  
  // Start with SMA for first EMA value
  let ema = prices.slice(0, period).reduce((sum, p) => sum + p, 0) / period;
  
  // Calculate EMA for remaining prices
  for (let i = period; i < prices.length; i++) {
    ema = (prices[i] - ema) * multiplier + ema;
  }
  
  return ema;
}

// ============================================================================
// RSI (Relative Strength Index)
// ============================================================================

export interface RSIResult {
  value: number;        // 0-100 scale
  isOversold: boolean;  // < 30
  isOverbought: boolean; // > 70
}

/**
 * Calculate RSI (Relative Strength Index).
 * Standard period is 14.
 * 
 * RSI = 100 - (100 / (1 + RS))
 * RS = Average Gain / Average Loss
 */
export function calculateRSI(history: HistoryPoint[], period: number = 14): RSIResult | null {
  const prices = getClosePrices(history);
  
  if (prices.length < period + 1) return null;
  
  // Calculate price changes
  const changes: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }
  
  // Separate gains and losses
  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? Math.abs(c) : 0);
  
  // Calculate initial average gain/loss (simple average for first period)
  let avgGain = gains.slice(0, period).reduce((sum, g) => sum + g, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((sum, l) => sum + l, 0) / period;
  
  // Apply smoothing for subsequent periods
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
  }
  
  // Calculate RSI
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  return {
    value: Math.round(rsi * 100) / 100,
    isOversold: rsi < 30,
    isOverbought: rsi > 70
  };
}

// ============================================================================
// MACD (Moving Average Convergence Divergence)
// ============================================================================

export interface MACDResult {
  macdLine: number;     // EMA(12) - EMA(26)
  signalLine: number;   // EMA(9) of MACD line
  histogram: number;    // MACD - Signal
  isBullish: boolean;   // MACD > Signal
  isBearish: boolean;   // MACD < Signal
}

/**
 * Calculate MACD indicator.
 * Standard settings: 12, 26, 9
 */
export function calculateMACD(
  history: HistoryPoint[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDResult | null {
  const prices = getClosePrices(history);
  
  // Need enough data for slow EMA + signal period
  if (prices.length < slowPeriod + signalPeriod) return null;
  
  // Calculate MACD line values for the entire series
  const macdValues: number[] = [];
  
  for (let i = slowPeriod; i <= prices.length; i++) {
    const slice = prices.slice(0, i);
    const fastEMA = calculateEMA(slice, fastPeriod);
    const slowEMA = calculateEMA(slice, slowPeriod);
    
    if (fastEMA !== null && slowEMA !== null) {
      macdValues.push(fastEMA - slowEMA);
    }
  }
  
  if (macdValues.length < signalPeriod) return null;
  
  // Calculate signal line (EMA of MACD values)
  const multiplier = 2 / (signalPeriod + 1);
  let signalLine = macdValues.slice(0, signalPeriod).reduce((sum, v) => sum + v, 0) / signalPeriod;
  
  for (let i = signalPeriod; i < macdValues.length; i++) {
    signalLine = (macdValues[i] - signalLine) * multiplier + signalLine;
  }
  
  const macdLine = macdValues[macdValues.length - 1];
  const histogram = macdLine - signalLine;
  
  return {
    macdLine: Math.round(macdLine * 1000) / 1000,
    signalLine: Math.round(signalLine * 1000) / 1000,
    histogram: Math.round(histogram * 1000) / 1000,
    isBullish: macdLine > signalLine,
    isBearish: macdLine < signalLine
  };
}

// ============================================================================
// Bollinger Bands
// ============================================================================

export interface BollingerBandsResult {
  upper: number;        // SMA + 2*stddev
  middle: number;       // SMA(20)
  lower: number;        // SMA - 2*stddev
  bandwidth: number;    // (upper - lower) / middle
  percentB: number;     // (price - lower) / (upper - lower), 0-1 scale
  isAboveUpper: boolean;
  isBelowLower: boolean;
}

/**
 * Calculate Bollinger Bands.
 * Standard settings: 20-period SMA, 2 standard deviations
 * 
 * Note: Uses SAMPLE standard deviation (n-1) to match TradingView/ThinkOrSwim
 */
export function calculateBollingerBands(
  history: HistoryPoint[],
  period: number = 20,
  stdDevMultiplier: number = 2
): BollingerBandsResult | null {
  const prices = getClosePrices(history);
  
  if (prices.length < period) return null;
  
  const recentPrices = prices.slice(-period);
  const currentPrice = prices[prices.length - 1];
  
  // Calculate SMA
  const middle = recentPrices.reduce((sum, p) => sum + p, 0) / period;
  
  // Calculate SAMPLE standard deviation (n-1) to match trading platforms
  const squaredDiffs = recentPrices.map(p => Math.pow(p - middle, 2));
  const variance = squaredDiffs.reduce((sum, d) => sum + d, 0) / (period - 1);
  const stdDev = Math.sqrt(variance);
  
  const upper = middle + stdDevMultiplier * stdDev;
  const lower = middle - stdDevMultiplier * stdDev;
  const bandwidth = (upper - lower) / middle;
  const percentB = (currentPrice - lower) / (upper - lower);
  
  return {
    upper: Math.round(upper * 100) / 100,
    middle: Math.round(middle * 100) / 100,
    lower: Math.round(lower * 100) / 100,
    bandwidth: Math.round(bandwidth * 1000) / 1000,
    percentB: Math.round(percentB * 1000) / 1000,
    isAboveUpper: currentPrice > upper,
    isBelowLower: currentPrice < lower
  };
}

// ============================================================================
// VWAP (Volume Weighted Average Price)
// ============================================================================

export interface VWAPResult {
  value: number;
  priceVsVwap: number;  // Percentage above/below VWAP
  isAbove: boolean;
  isBelow: boolean;
  isIntraday: boolean;  // True if calculated from intraday data
}

/**
 * Detect if history data is intraday (has time component in date string).
 */
function isIntradayHistory(history: HistoryPoint[]): boolean {
  if (!history.length) return false;
  // Intraday data has format "YYYY-MM-DD HH:MM"
  return history[0].date.includes(' ');
}

/**
 * Calculate VWAP (Volume Weighted Average Price).
 * 
 * For INTRADAY data: Calculates true session VWAP (resets at market open)
 * For DAILY data: Calculates a 20-day Rolling VWAP (not standard VWAP, but useful)
 * 
 * Standard VWAP is an intraday indicator that resets at market open.
 * If passed daily data, we use a rolling window to provide meaningful support/resistance.
 */
export function calculateVWAP(history: HistoryPoint[], rollingPeriod: number = 20): VWAPResult | null {
  if (history.length === 0) return null;
  
  const isIntraday = isIntradayHistory(history);
  
  let dataToUse: HistoryPoint[];
  
  if (isIntraday) {
    // For intraday: Get only today's data (same date prefix)
    const today = history[history.length - 1].date.split(' ')[0];
    dataToUse = history.filter(h => h.date.startsWith(today));
  } else {
    // For daily: Use rolling window (not true VWAP, but provides value)
    dataToUse = history.slice(-rollingPeriod);
  }
  
  // Filter out entries with zero volume
  const validData = dataToUse.filter(h => h.volume > 0);
  if (validData.length === 0) return null;
  
  // Calculate typical price * volume for each period
  let cumulativeTPV = 0;
  let cumulativeVolume = 0;
  
  for (const h of validData) {
    const typicalPrice = (h.high + h.low + h.close) / 3;
    cumulativeTPV += typicalPrice * h.volume;
    cumulativeVolume += h.volume;
  }
  
  if (cumulativeVolume === 0) return null;
  
  const vwap = cumulativeTPV / cumulativeVolume;
  const currentPrice = history[history.length - 1].close;
  const priceVsVwap = ((currentPrice - vwap) / vwap) * 100;
  
  return {
    value: Math.round(vwap * 100) / 100,
    priceVsVwap: Math.round(priceVsVwap * 100) / 100,
    isIntraday,
    isAbove: currentPrice > vwap,
    isBelow: currentPrice < vwap
  };
}

// ============================================================================
// Price Momentum
// ============================================================================

export interface MomentumResult {
  shortTerm: number;    // 5-day momentum (%)
  mediumTerm: number;   // 20-day momentum (%)
  longTerm: number;     // 50-day momentum (%)
  trend: 'bullish' | 'bearish' | 'neutral';
}

/**
 * Calculate price momentum over various periods.
 * Momentum = ((Current Price - Price N days ago) / Price N days ago) * 100
 */
export function calculateMomentum(history: HistoryPoint[]): MomentumResult | null {
  const prices = getClosePrices(history);
  
  if (prices.length < 5) return null;
  
  const currentPrice = prices[prices.length - 1];
  
  const calculatePeriodMomentum = (period: number): number => {
    if (prices.length < period) return 0;
    const pastPrice = prices[prices.length - period];
    return ((currentPrice - pastPrice) / pastPrice) * 100;
  };
  
  const shortTerm = calculatePeriodMomentum(5);
  const mediumTerm = prices.length >= 20 ? calculatePeriodMomentum(20) : shortTerm;
  const longTerm = prices.length >= 50 ? calculatePeriodMomentum(50) : mediumTerm;
  
  // Determine trend based on momentum alignment
  let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
  if (shortTerm > 0 && mediumTerm > 0 && longTerm > 0) {
    trend = 'bullish';
  } else if (shortTerm < 0 && mediumTerm < 0 && longTerm < 0) {
    trend = 'bearish';
  }
  
  return {
    shortTerm: Math.round(shortTerm * 100) / 100,
    mediumTerm: Math.round(mediumTerm * 100) / 100,
    longTerm: Math.round(longTerm * 100) / 100,
    trend
  };
}

// ============================================================================
// Volume Analysis
// ============================================================================

export interface VolumeAnalysisResult {
  avgVolume: number;         // 20-day average volume
  currentVolume: number;     // Most recent volume
  volumeRatio: number;       // Current / Average
  isHighVolume: boolean;     // > 1.5x average
  isLowVolume: boolean;      // < 0.5x average
  trend: 'increasing' | 'decreasing' | 'stable';
}

/**
 * Analyze volume patterns.
 */
export function analyzeVolume(history: HistoryPoint[]): VolumeAnalysisResult | null {
  if (history.length < 5) return null;
  
  const volumes = history.map(h => h.volume).filter(v => v > 0);
  if (volumes.length < 5) return null;
  
  const currentVolume = volumes[volumes.length - 1];
  const period = Math.min(20, volumes.length);
  const recentVolumes = volumes.slice(-period);
  const avgVolume = recentVolumes.reduce((sum, v) => sum + v, 0) / period;
  
  const volumeRatio = avgVolume > 0 ? currentVolume / avgVolume : 1;
  
  // Determine volume trend (compare first half to second half of recent period)
  const halfPeriod = Math.floor(period / 2);
  const firstHalf = recentVolumes.slice(0, halfPeriod);
  const secondHalf = recentVolumes.slice(halfPeriod);
  
  const firstHalfAvg = firstHalf.reduce((sum, v) => sum + v, 0) / firstHalf.length;
  const secondHalfAvg = secondHalf.reduce((sum, v) => sum + v, 0) / secondHalf.length;
  
  let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
  if (secondHalfAvg > firstHalfAvg * 1.2) {
    trend = 'increasing';
  } else if (secondHalfAvg < firstHalfAvg * 0.8) {
    trend = 'decreasing';
  }
  
  return {
    avgVolume: Math.round(avgVolume),
    currentVolume,
    volumeRatio: Math.round(volumeRatio * 100) / 100,
    isHighVolume: volumeRatio > 1.5,
    isLowVolume: volumeRatio < 0.5,
    trend
  };
}

// ============================================================================
// Price Position Analysis
// ============================================================================

export interface PricePositionResult {
  priceVs52wHigh: number;    // % below 52-week high
  priceVs52wLow: number;     // % above 52-week low
  rangePosition: number;     // 0-1, where 0 = at low, 1 = at high
  isNear52wHigh: boolean;    // Within 5% of high
  isNear52wLow: boolean;     // Within 5% of low
}

/**
 * Analyze price position relative to 52-week range.
 */
export function analyzePricePosition(
  currentPrice: number,
  high52w: number | null,
  low52w: number | null
): PricePositionResult | null {
  if (!high52w || !low52w || high52w <= low52w) return null;
  
  const priceVs52wHigh = ((currentPrice - high52w) / high52w) * 100;
  const priceVs52wLow = ((currentPrice - low52w) / low52w) * 100;
  const rangePosition = (currentPrice - low52w) / (high52w - low52w);
  
  return {
    priceVs52wHigh: Math.round(priceVs52wHigh * 100) / 100,
    priceVs52wLow: Math.round(priceVs52wLow * 100) / 100,
    rangePosition: Math.round(rangePosition * 1000) / 1000,
    isNear52wHigh: priceVs52wHigh >= -5,
    isNear52wLow: priceVs52wLow <= 5
  };
}

// ============================================================================
// Relative Volume (RVOL)
// ============================================================================

export interface RVOLResult {
  rvol: number;              // Current volume / Average volume ratio
  avgVolume: number;         // 20-period average volume
  currentVolume: number;     // Most recent volume
  interpretation: 'very_high' | 'high' | 'normal' | 'low' | 'very_low';
  isSignificant: boolean;    // RVOL > 2.0 or < 0.5
}

/**
 * Calculate Relative Volume (RVOL).
 * RVOL = Current Volume / Average Volume
 * 
 * Interpretation:
 * - > 3.0: Very high (unusual activity, potential catalyst)
 * - > 1.5: High (above average interest)
 * - 0.8 - 1.5: Normal
 * - 0.5 - 0.8: Low
 * - < 0.5: Very low (lack of interest)
 */
export function calculateRVOL(history: HistoryPoint[], period: number = 20): RVOLResult | null {
  if (history.length < period) return null;
  
  const volumes = history.map(h => h.volume).filter(v => v > 0);
  if (volumes.length < period) return null;
  
  const currentVolume = volumes[volumes.length - 1];
  const recentVolumes = volumes.slice(-period);
  const avgVolume = recentVolumes.reduce((sum, v) => sum + v, 0) / period;
  
  if (avgVolume === 0) return null;
  
  const rvol = currentVolume / avgVolume;
  
  let interpretation: RVOLResult['interpretation'];
  if (rvol > 3.0) interpretation = 'very_high';
  else if (rvol > 1.5) interpretation = 'high';
  else if (rvol >= 0.8) interpretation = 'normal';
  else if (rvol >= 0.5) interpretation = 'low';
  else interpretation = 'very_low';
  
  return {
    rvol: Math.round(rvol * 100) / 100,
    avgVolume: Math.round(avgVolume),
    currentVolume,
    interpretation,
    isSignificant: rvol > 2.0 || rvol < 0.5
  };
}

// ============================================================================
// ADX (Average Directional Index) - Trend Strength Indicator
// ============================================================================

export interface ADXResult {
  adx: number;                    // 0-100 scale
  plusDI: number;                 // +DI (Directional Indicator)
  minusDI: number;                // -DI (Directional Indicator)
  trend: 'strong_trend' | 'trending' | 'weak_trend' | 'no_trend';
  direction: 'bullish' | 'bearish' | 'neutral';
  isTrending: boolean;            // ADX > 25
  isStrongTrend: boolean;         // ADX > 40
}

/**
 * Calculate ADX (Average Directional Index).
 * 
 * ADX measures trend strength (not direction):
 * - ADX < 20: No trend / sideways market (mean reversion favored)
 * - ADX 20-25: Weak trend emerging
 * - ADX 25-40: Trending market (trend following favored)
 * - ADX > 40: Strong trend
 * - ADX > 50: Very strong trend (potential exhaustion)
 * 
 * Uses Wilder's smoothing (same as RSI) for consistency.
 * Standard period is 14.
 */
export function calculateADX(history: HistoryPoint[], period: number = 14): ADXResult | null {
  if (history.length < period * 2 + 1) return null;
  
  // Calculate True Range (TR), +DM, -DM for each period
  const trueRanges: number[] = [];
  const plusDMs: number[] = [];
  const minusDMs: number[] = [];
  
  for (let i = 1; i < history.length; i++) {
    const current = history[i];
    const previous = history[i - 1];
    
    // True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    const tr = Math.max(
      current.high - current.low,
      Math.abs(current.high - previous.close),
      Math.abs(current.low - previous.close)
    );
    trueRanges.push(tr);
    
    // +DM = High - PrevHigh (if positive and > |Low - PrevLow|)
    // -DM = PrevLow - Low (if positive and > High - PrevHigh)
    const upMove = current.high - previous.high;
    const downMove = previous.low - current.low;
    
    let plusDM = 0;
    let minusDM = 0;
    
    if (upMove > downMove && upMove > 0) {
      plusDM = upMove;
    }
    if (downMove > upMove && downMove > 0) {
      minusDM = downMove;
    }
    
    plusDMs.push(plusDM);
    minusDMs.push(minusDM);
  }
  
  if (trueRanges.length < period) return null;
  
  // Calculate initial smoothed values using SMA (Wilder's method)
  let smoothedTR = trueRanges.slice(0, period).reduce((sum, v) => sum + v, 0);
  let smoothedPlusDM = plusDMs.slice(0, period).reduce((sum, v) => sum + v, 0);
  let smoothedMinusDM = minusDMs.slice(0, period).reduce((sum, v) => sum + v, 0);
  
  // Apply Wilder's smoothing for subsequent periods
  const dxValues: number[] = [];
  
  for (let i = period; i < trueRanges.length; i++) {
    // Wilder's smoothing: smoothed = prev - (prev/period) + current
    smoothedTR = smoothedTR - (smoothedTR / period) + trueRanges[i];
    smoothedPlusDM = smoothedPlusDM - (smoothedPlusDM / period) + plusDMs[i];
    smoothedMinusDM = smoothedMinusDM - (smoothedMinusDM / period) + minusDMs[i];
    
    // Calculate +DI and -DI
    const plusDI = smoothedTR > 0 ? (smoothedPlusDM / smoothedTR) * 100 : 0;
    const minusDI = smoothedTR > 0 ? (smoothedMinusDM / smoothedTR) * 100 : 0;
    
    // Calculate DX
    const diSum = plusDI + minusDI;
    const dx = diSum > 0 ? (Math.abs(plusDI - minusDI) / diSum) * 100 : 0;
    dxValues.push(dx);
  }
  
  if (dxValues.length < period) return null;
  
  // Calculate ADX as smoothed average of DX
  let adx = dxValues.slice(0, period).reduce((sum, v) => sum + v, 0) / period;
  
  for (let i = period; i < dxValues.length; i++) {
    adx = ((adx * (period - 1)) + dxValues[i]) / period;
  }
  
  // Get current +DI and -DI for direction
  const currentPlusDI = smoothedTR > 0 ? (smoothedPlusDM / smoothedTR) * 100 : 0;
  const currentMinusDI = smoothedTR > 0 ? (smoothedMinusDM / smoothedTR) * 100 : 0;
  
  // Determine trend strength
  let trend: ADXResult['trend'];
  if (adx >= 40) {
    trend = 'strong_trend';
  } else if (adx >= 25) {
    trend = 'trending';
  } else if (adx >= 20) {
    trend = 'weak_trend';
  } else {
    trend = 'no_trend';
  }
  
  // Determine direction from +DI vs -DI
  let direction: ADXResult['direction'];
  if (currentPlusDI > currentMinusDI + 5) {
    direction = 'bullish';
  } else if (currentMinusDI > currentPlusDI + 5) {
    direction = 'bearish';
  } else {
    direction = 'neutral';
  }
  
  return {
    adx: Math.round(adx * 100) / 100,
    plusDI: Math.round(currentPlusDI * 100) / 100,
    minusDI: Math.round(currentMinusDI * 100) / 100,
    trend,
    direction,
    isTrending: adx >= 25,
    isStrongTrend: adx >= 40
  };
}

// ============================================================================
// Bollinger Band Squeeze (Volatility Contraction)
// ============================================================================

export interface BollingerSqueezeResult {
  bandwidth: number;              // Current bandwidth
  avgBandwidth: number;           // Average bandwidth over period
  bandwidthPercentile: number;    // 0-100, where low = squeeze
  isSqueeze: boolean;             // Bandwidth < 20th percentile
  isExpansion: boolean;           // Bandwidth > 80th percentile
  squeezeIntensity: 'extreme' | 'moderate' | 'mild' | 'none';
}

/**
 * Detect Bollinger Band Squeeze (volatility contraction).
 * 
 * A "squeeze" occurs when Bollinger Bands contract, indicating low volatility
 * which often precedes a significant price move (breakout).
 * 
 * - Squeeze (low bandwidth): Favors buying options (cheap premiums, expected expansion)
 * - Expansion (high bandwidth): Favors selling options (expensive premiums)
 */
export function calculateBollingerSqueeze(
  history: HistoryPoint[],
  period: number = 20,
  lookback: number = 100
): BollingerSqueezeResult | null {
  if (history.length < Math.max(period, lookback)) return null;
  
  // Calculate bandwidth for each period in the lookback window
  const bandwidths: number[] = [];
  
  for (let i = period; i <= Math.min(lookback, history.length); i++) {
    const slice = history.slice(history.length - i, history.length - i + period);
    
    // Calculate bandwidth for this specific window
    const prices = slice.map(h => h.close).filter(p => p > 0);
    if (prices.length < period) continue;
    
    const middle = prices.reduce((sum, p) => sum + p, 0) / period;
    const squaredDiffs = prices.map(p => Math.pow(p - middle, 2));
    const variance = squaredDiffs.reduce((sum, d) => sum + d, 0) / (period - 1);
    const stdDev = Math.sqrt(variance);
    
    const upper = middle + 2 * stdDev;
    const lower = middle - 2 * stdDev;
    const bandwidth = (upper - lower) / middle;
    
    bandwidths.push(bandwidth);
  }
  
  if (bandwidths.length < 20) {
    // Not enough data for percentile calculation, use current bandwidth only
    const currentBB = calculateBollingerBands(history);
    if (!currentBB) return null;
    
    return {
      bandwidth: currentBB.bandwidth,
      avgBandwidth: currentBB.bandwidth,
      bandwidthPercentile: 50,
      isSqueeze: false,
      isExpansion: false,
      squeezeIntensity: 'none'
    };
  }
  
  const currentBandwidth = bandwidths[bandwidths.length - 1];
  const avgBandwidth = bandwidths.reduce((sum, b) => sum + b, 0) / bandwidths.length;
  
  // Calculate percentile
  const sortedBandwidths = [...bandwidths].sort((a, b) => a - b);
  const currentRank = sortedBandwidths.filter(b => b <= currentBandwidth).length;
  const bandwidthPercentile = (currentRank / sortedBandwidths.length) * 100;
  
  // Determine squeeze intensity
  let squeezeIntensity: BollingerSqueezeResult['squeezeIntensity'];
  if (bandwidthPercentile <= 5) {
    squeezeIntensity = 'extreme';
  } else if (bandwidthPercentile <= 15) {
    squeezeIntensity = 'moderate';
  } else if (bandwidthPercentile <= 25) {
    squeezeIntensity = 'mild';
  } else {
    squeezeIntensity = 'none';
  }
  
  return {
    bandwidth: Math.round(currentBandwidth * 1000) / 1000,
    avgBandwidth: Math.round(avgBandwidth * 1000) / 1000,
    bandwidthPercentile: Math.round(bandwidthPercentile),
    isSqueeze: bandwidthPercentile <= 20,
    isExpansion: bandwidthPercentile >= 80,
    squeezeIntensity
  };
}

// ============================================================================
// Golden Cross / Death Cross Pattern Detection
// ============================================================================

export interface CrossPatternResult {
  pattern: 'golden_cross' | 'death_cross' | 'golden_star' | 'none';
  description: string;
  daysAgo: number | null;      // How many days ago the cross occurred
  strength: 'strong' | 'moderate' | 'weak';
  // Current SMA positions
  sma50AboveSma200: boolean;
  priceAboveSma50: boolean;
  priceAboveSma200: boolean;
  // Trend context
  trendAlignment: 'bullish' | 'bearish' | 'mixed';
}

/**
 * Detect Golden Cross, Death Cross, and Golden Star patterns.
 * 
 * Golden Cross: SMA(50) crosses above SMA(200) - bullish
 * Death Cross: SMA(50) crosses below SMA(200) - bearish
 * Golden Star: Price breaks above both SMAs with volume confirmation
 */
export function detectCrossPatterns(history: HistoryPoint[]): CrossPatternResult | null {
  if (history.length < 200) {
    // Need enough data for SMA200
    return null;
  }
  
  const prices = getClosePrices(history);
  const currentPrice = prices[prices.length - 1];
  
  // Calculate SMAs for the last 20 days to detect recent crosses
  const smaHistory: { sma50: number; sma200: number }[] = [];
  
  for (let i = 200; i <= prices.length; i++) {
    const slice = prices.slice(0, i);
    const sma50 = calculateSMA(slice, 50);
    const sma200 = calculateSMA(slice, 200);
    if (sma50 && sma200) {
      smaHistory.push({ sma50, sma200 });
    }
  }
  
  if (smaHistory.length < 2) return null;
  
  const current = smaHistory[smaHistory.length - 1];
  const sma50AboveSma200 = current.sma50 > current.sma200;
  const priceAboveSma50 = currentPrice > current.sma50;
  const priceAboveSma200 = currentPrice > current.sma200;
  
  // Detect when the cross occurred
  let crossType: 'golden_cross' | 'death_cross' | 'none' = 'none';
  let daysAgo: number | null = null;
  
  // Look back up to 20 days for a cross
  for (let i = smaHistory.length - 2; i >= Math.max(0, smaHistory.length - 20); i--) {
    const prev = smaHistory[i];
    const curr = smaHistory[i + 1];
    
    // Golden Cross: SMA50 crosses above SMA200
    if (prev.sma50 <= prev.sma200 && curr.sma50 > curr.sma200) {
      crossType = 'golden_cross';
      daysAgo = smaHistory.length - 1 - i;
      break;
    }
    
    // Death Cross: SMA50 crosses below SMA200
    if (prev.sma50 >= prev.sma200 && curr.sma50 < curr.sma200) {
      crossType = 'death_cross';
      daysAgo = smaHistory.length - 1 - i;
      break;
    }
  }
  
  // Determine trend alignment
  let trendAlignment: CrossPatternResult['trendAlignment'];
  if (sma50AboveSma200 && priceAboveSma50 && priceAboveSma200) {
    trendAlignment = 'bullish';
  } else if (!sma50AboveSma200 && !priceAboveSma50 && !priceAboveSma200) {
    trendAlignment = 'bearish';
  } else {
    trendAlignment = 'mixed';
  }
  
  // Check for Golden Star pattern
  // Golden Star: Recent upward momentum with volume + price above key SMAs
  let pattern: CrossPatternResult['pattern'] = crossType;
  let description = '';
  let strength: CrossPatternResult['strength'] = 'moderate';
  
  if (crossType === 'golden_cross') {
    description = `Golden Cross detected ${daysAgo} day${daysAgo === 1 ? '' : 's'} ago`;
    strength = daysAgo !== null && daysAgo <= 5 ? 'strong' : 'moderate';
  } else if (crossType === 'death_cross') {
    description = `Death Cross detected ${daysAgo} day${daysAgo === 1 ? '' : 's'} ago`;
    strength = daysAgo !== null && daysAgo <= 5 ? 'strong' : 'moderate';
  } else if (trendAlignment === 'bullish') {
    // Check for Golden Star (all signals aligned bullishly)
    const rvol = calculateRVOL(history);
    if (rvol && rvol.rvol > 1.5) {
      pattern = 'golden_star';
      description = 'Golden Star: Bullish alignment with high volume';
      strength = 'strong';
    } else {
      description = 'Bullish trend: Price above both SMAs';
      strength = 'moderate';
    }
  } else if (trendAlignment === 'bearish') {
    description = 'Bearish trend: Price below both SMAs';
    strength = 'moderate';
  } else {
    description = 'Mixed signals: No clear trend';
    strength = 'weak';
  }
  
  return {
    pattern,
    description,
    daysAgo,
    strength,
    sma50AboveSma200,
    priceAboveSma50,
    priceAboveSma200,
    trendAlignment
  };
}

// ============================================================================
// Aggregate Technical Indicators
// ============================================================================

export interface TechnicalIndicators {
  rsi: RSIResult | null;
  macd: MACDResult | null;
  bollingerBands: BollingerBandsResult | null;
  bollingerSqueeze: BollingerSqueezeResult | null;
  vwap: VWAPResult | null;
  momentum: MomentumResult | null;
  volume: VolumeAnalysisResult | null;
  pricePosition: PricePositionResult | null;
  rvol: RVOLResult | null;
  adx: ADXResult | null;
  crossPattern: CrossPatternResult | null;
  sma20: number | null;
  sma50: number | null;
  sma200: number | null;
  ema12: number | null;
  ema26: number | null;
}

/**
 * Calculate all technical indicators for a stock.
 */
export function calculateAllIndicators(
  history: HistoryPoint[],
  currentPrice: number,
  high52w: number | null,
  low52w: number | null
): TechnicalIndicators {
  const prices = getClosePrices(history);
  
  return {
    rsi: calculateRSI(history),
    macd: calculateMACD(history),
    bollingerBands: calculateBollingerBands(history),
    bollingerSqueeze: calculateBollingerSqueeze(history),
    vwap: calculateVWAP(history),
    momentum: calculateMomentum(history),
    volume: analyzeVolume(history),
    pricePosition: analyzePricePosition(currentPrice, high52w, low52w),
    rvol: calculateRVOL(history),
    adx: calculateADX(history),
    crossPattern: detectCrossPatterns(history),
    sma20: calculateSMA(prices, 20),
    sma50: calculateSMA(prices, 50),
    sma200: calculateSMA(prices, 200),
    ema12: calculateEMA(prices, 12),
    ema26: calculateEMA(prices, 26)
  };
}

