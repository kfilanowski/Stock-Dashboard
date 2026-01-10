/**
 * Golden Master Export Service
 * 
 * Exports signal values from the TypeScript frontend for parity testing
 * with the Python backend. This ensures the WFO calibration engine
 * optimizes the SAME strategy as the live trading system.
 * 
 * Usage:
 * 1. Call exportGoldenMaster() with a ticker and date range
 * 2. Save the output JSON to backend/tests/fixtures/golden_master.json
 * 3. Run test_indicator_parity.py to verify Python matches TypeScript
 */

import type { HistoryPoint } from '../types';
import { calculateAllIndicators, type TechnicalIndicators } from './technicalIndicators';

// ============================================================================
// Types
// ============================================================================

export interface GoldenMasterDataPoint {
  date: string;
  // OHLCV
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  // Indicator signals (normalized -1 to +1)
  signals: {
    rsi_signal: number | null;
    rsi_value: number | null;
    macd_signal: number | null;
    macd_line: number | null;
    macd_histogram: number | null;
    bollinger_signal: number | null;
    bb_percent_b: number | null;
    bb_bandwidth: number | null;
    adx_signal: number | null;
    adx_value: number | null;
    cmf_signal: number | null;
    cmf_value: number | null;
    momentum_signal: number | null;
    momentum_short: number | null;
    volume_signal: number | null;
    volume_ratio: number | null;
    rvol_signal: number | null;
    rvol_value: number | null;
    sma_signal: number | null;
    sma_20: number | null;
    sma_50: number | null;
    sma_200: number | null;
    position_signal: number | null;
    range_position: number | null;
    squeeze_signal: number | null;
    squeeze_percentile: number | null;
  };
}

export interface GoldenMasterExport {
  ticker: string;
  exportedAt: string;
  dataRange: {
    start: string;
    end: string;
    totalDays: number;
  };
  // First 50 days are warmup (ignore in parity tests)
  warmupPeriod: number;
  dataPoints: GoldenMasterDataPoint[];
  // Summary for quick validation
  summary: {
    rsiRange: [number, number];
    macdRange: [number, number];
    adxRange: [number, number];
    cmfRange: [number, number];
  };
}

// ============================================================================
// Signal Calculation (mirrors Python logic)
// ============================================================================

/**
 * Convert RSI to signal (-1 to +1).
 * Uses mean reversion interpretation.
 */
function rsiToSignal(rsi: number | null, adxTrending: boolean): number | null {
  if (rsi === null) return null;
  
  // Regime override: Trending market (ADX > 25)
  if (adxTrending && rsi >= 65 && rsi < 85) {
    return 0.8;
  }
  
  // Base RSI signal (mean reversion)
  if (rsi <= 30) {
    return 1.0 - (rsi / 30) * 0.5;
  } else if (rsi <= 50) {
    return 0.5 - ((rsi - 30) / 20) * 0.5;
  } else if (rsi <= 70) {
    return -((rsi - 50) / 20) * 0.5;
  } else {
    return -0.5 - ((rsi - 70) / 30) * 0.5;
  }
}

/**
 * Convert MACD to signal (-1 to +1).
 */
function macdToSignal(histogram: number | null, isBullish: boolean): number | null {
  if (histogram === null) return null;
  
  const histogramStrength = Math.min(Math.abs(histogram) / 2, 1);
  
  if (isBullish) {
    return 0.3 + histogramStrength * 0.7;
  } else {
    return -(0.3 + histogramStrength * 0.7);
  }
}

/**
 * Convert Bollinger %B to signal (-1 to +1).
 */
function bollingerToSignal(percentB: number | null): number | null {
  if (percentB === null) return null;
  
  if (percentB <= 0) return 1.0;
  if (percentB <= 0.2) return 0.8;
  if (percentB <= 0.4) return 0.4;
  if (percentB <= 0.6) return 0.0;
  if (percentB <= 0.8) return -0.4;
  if (percentB <= 1.0) return -0.8;
  return -1.0;
}

/**
 * Convert ADX to signal (-1 to +1).
 */
function adxToSignal(adx: number | null, direction: string | null): number | null {
  if (adx === null) return null;
  
  let signal: number;
  if (adx < 20) {
    signal = 0.0;
  } else if (adx < 25) {
    signal = 0.3;
  } else if (adx < 40) {
    signal = 0.6;
  } else {
    signal = 0.9;
  }
  
  // Direction adjustment
  if (direction === 'bearish') {
    signal = -signal;
  }
  
  return Math.max(-1, Math.min(1, signal));
}

/**
 * Convert CMF to signal (-1 to +1).
 */
function cmfToSignal(cmf: number | null): number | null {
  if (cmf === null) return null;
  return Math.max(-1, Math.min(1, cmf * 2));
}

/**
 * Convert momentum to signal (-1 to +1).
 */
function momentumToSignal(shortTerm: number | null): number | null {
  if (shortTerm === null) return null;
  
  if (shortTerm > 10) return 1.0;
  if (shortTerm > 5) return 0.6;
  if (shortTerm > 0) return 0.2;
  if (shortTerm > -5) return -0.2;
  if (shortTerm > -10) return -0.6;
  return -1.0;
}

/**
 * Convert volume ratio to signal (-1 to +1).
 */
function volumeToSignal(volumeRatio: number | null): number | null {
  if (volumeRatio === null) return null;
  
  if (volumeRatio > 2.0) return 0.8;
  if (volumeRatio > 1.5) return 0.5;
  if (volumeRatio > 1.0) return 0.2;
  if (volumeRatio > 0.5) return -0.2;
  return -0.5;
}

/**
 * Convert RVOL to signal (-1 to +1).
 */
function rvolToSignal(rvol: number | null): number | null {
  if (rvol === null) return null;
  
  if (rvol > 3.0) return 0.9;
  if (rvol > 2.0) return 0.6;
  if (rvol > 1.5) return 0.3;
  if (rvol >= 0.8) return 0.0;
  if (rvol >= 0.5) return -0.3;
  return -0.6;
}

/**
 * Convert SMA alignment to signal (-1 to +1).
 */
function smaToSignal(
  currentPrice: number,
  _sma20: number | null,
  sma50: number | null,
  sma200: number | null
): number | null {
  if (sma50 === null) return null;
  
  const aboveSma50 = currentPrice > sma50;
  const aboveSma200 = sma200 ? currentPrice > sma200 : false;
  const bullishAligned = aboveSma50 && aboveSma200;
  const bearishAligned = !aboveSma50 && (sma200 ? !aboveSma200 : true);
  
  if (bullishAligned) return 0.8;
  if (bearishAligned) return -0.8;
  return 0.0;
}

/**
 * Convert range position to signal (-1 to +1).
 */
function positionToSignal(rangePosition: number | null): number | null {
  if (rangePosition === null) return null;
  // 0 → 1, 0.5 → 0, 1 → -1
  return Math.max(-1, Math.min(1, 1.0 - (rangePosition * 2)));
}

/**
 * Convert squeeze status to signal (-1 to +1).
 */
function squeezeToSignal(isSqueeze: boolean, isExpansion: boolean): number {
  if (isSqueeze) return 0.8;
  if (isExpansion) return -0.5;
  return 0.0;
}

// ============================================================================
// Main Export Function
// ============================================================================

/**
 * Export Golden Master data for a stock.
 * 
 * @param ticker Stock ticker symbol
 * @param history Price history data (should be at least 300 days for valid SMAs)
 * @param currentPrice Current stock price
 * @param high52w 52-week high (optional)
 * @param low52w 52-week low (optional)
 * @returns GoldenMasterExport object to save as JSON
 */
export function exportGoldenMaster(
  ticker: string,
  history: HistoryPoint[],
  _currentPrice: number,
  high52w: number | null,
  low52w: number | null
): GoldenMasterExport {
  if (history.length < 100) {
    throw new Error(`Insufficient history: ${history.length} days. Need at least 100.`);
  }
  
  const dataPoints: GoldenMasterDataPoint[] = [];
  
  // Track ranges for summary
  let rsiMin = Infinity, rsiMax = -Infinity;
  let macdMin = Infinity, macdMax = -Infinity;
  let adxMin = Infinity, adxMax = -Infinity;
  let cmfMin = Infinity, cmfMax = -Infinity;
  
  // Calculate indicators for each day using a sliding window
  // We need to recalculate for each point to get the proper time series
  for (let i = 50; i < history.length; i++) {
    const windowHistory = history.slice(0, i + 1);
    const point = history[i];
    
    // Calculate indicators using the history up to this point
    const indicators: TechnicalIndicators = calculateAllIndicators(
      windowHistory,
      point.close,
      high52w,
      low52w
    );
    
    // Extract raw values
    const rsiValue = indicators.rsi?.value ?? null;
    const macdLine = indicators.macd?.macdLine ?? null;
    const macdHistogram = indicators.macd?.histogram ?? null;
    const macdBullish = indicators.macd?.isBullish ?? false;
    const bbPercentB = indicators.bollingerBands?.percentB ?? null;
    const bbBandwidth = indicators.bollingerBands?.bandwidth ?? null;
    const adxValue = indicators.adx?.adx ?? null;
    const adxDirection = indicators.adx?.direction ?? null;
    const adxTrending = indicators.adx?.isTrending ?? false;
    const cmfValue = indicators.cmf?.value ?? null;
    const momentumShort = indicators.momentum?.shortTerm ?? null;
    const volumeRatio = indicators.volume?.volumeRatio ?? null;
    const rvolValue = indicators.rvol?.rvol ?? null;
    const sma20 = indicators.sma20;
    const sma50 = indicators.sma50;
    const sma200 = indicators.sma200;
    const rangePosition = indicators.pricePosition?.rangePosition ?? null;
    const isSqueeze = indicators.bollingerSqueeze?.isSqueeze ?? false;
    const isExpansion = indicators.bollingerSqueeze?.isExpansion ?? false;
    const squeezePercentile = indicators.bollingerSqueeze?.bandwidthPercentile ?? null;
    
    // Update ranges
    if (rsiValue !== null) {
      rsiMin = Math.min(rsiMin, rsiValue);
      rsiMax = Math.max(rsiMax, rsiValue);
    }
    if (macdLine !== null) {
      macdMin = Math.min(macdMin, macdLine);
      macdMax = Math.max(macdMax, macdLine);
    }
    if (adxValue !== null) {
      adxMin = Math.min(adxMin, adxValue);
      adxMax = Math.max(adxMax, adxValue);
    }
    if (cmfValue !== null) {
      cmfMin = Math.min(cmfMin, cmfValue);
      cmfMax = Math.max(cmfMax, cmfValue);
    }
    
    // Build data point
    const dataPoint: GoldenMasterDataPoint = {
      date: point.date,
      open: point.open,
      high: point.high,
      low: point.low,
      close: point.close,
      volume: point.volume,
      signals: {
        rsi_signal: rsiToSignal(rsiValue, adxTrending),
        rsi_value: rsiValue !== null ? Number(rsiValue.toFixed(4)) : null,
        macd_signal: macdToSignal(macdHistogram, macdBullish),
        macd_line: macdLine !== null ? Number(macdLine.toFixed(4)) : null,
        macd_histogram: macdHistogram !== null ? Number(macdHistogram.toFixed(4)) : null,
        bollinger_signal: bollingerToSignal(bbPercentB),
        bb_percent_b: bbPercentB !== null ? Number(bbPercentB.toFixed(4)) : null,
        bb_bandwidth: bbBandwidth !== null ? Number(bbBandwidth.toFixed(4)) : null,
        adx_signal: adxToSignal(adxValue, adxDirection),
        adx_value: adxValue !== null ? Number(adxValue.toFixed(4)) : null,
        cmf_signal: cmfToSignal(cmfValue),
        cmf_value: cmfValue !== null ? Number(cmfValue.toFixed(4)) : null,
        momentum_signal: momentumToSignal(momentumShort),
        momentum_short: momentumShort !== null ? Number(momentumShort.toFixed(4)) : null,
        volume_signal: volumeToSignal(volumeRatio),
        volume_ratio: volumeRatio !== null ? Number(volumeRatio.toFixed(4)) : null,
        rvol_signal: rvolToSignal(rvolValue),
        rvol_value: rvolValue !== null ? Number(rvolValue.toFixed(4)) : null,
        sma_signal: smaToSignal(point.close, sma20, sma50, sma200),
        sma_20: sma20 !== null ? Number(sma20.toFixed(4)) : null,
        sma_50: sma50 !== null ? Number(sma50.toFixed(4)) : null,
        sma_200: sma200 !== null ? Number(sma200.toFixed(4)) : null,
        position_signal: positionToSignal(rangePosition),
        range_position: rangePosition !== null ? Number(rangePosition.toFixed(4)) : null,
        squeeze_signal: squeezeToSignal(isSqueeze, isExpansion),
        squeeze_percentile: squeezePercentile !== null ? Number(squeezePercentile.toFixed(0)) : null,
      }
    };
    
    // Round signals to 4 decimal places
    for (const key of Object.keys(dataPoint.signals) as (keyof typeof dataPoint.signals)[]) {
      const val = dataPoint.signals[key];
      if (val !== null && typeof val === 'number') {
        dataPoint.signals[key] = Number(val.toFixed(4)) as never;
      }
    }
    
    dataPoints.push(dataPoint);
  }
  
  return {
    ticker,
    exportedAt: new Date().toISOString(),
    dataRange: {
      start: dataPoints[0]?.date ?? '',
      end: dataPoints[dataPoints.length - 1]?.date ?? '',
      totalDays: dataPoints.length
    },
    warmupPeriod: 50,
    dataPoints,
    summary: {
      rsiRange: [Number(rsiMin.toFixed(2)), Number(rsiMax.toFixed(2))],
      macdRange: [Number(macdMin.toFixed(4)), Number(macdMax.toFixed(4))],
      adxRange: [Number(adxMin.toFixed(2)), Number(adxMax.toFixed(2))],
      cmfRange: [Number(cmfMin.toFixed(4)), Number(cmfMax.toFixed(4))]
    }
  };
}

/**
 * Download Golden Master as JSON file.
 */
export function downloadGoldenMaster(data: GoldenMasterExport): void {
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `golden_master_${data.ticker}_${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

