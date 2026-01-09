/**
 * Generate Golden Master from TypeScript
 * 
 * This script generates the golden_master.json file using the ACTUAL
 * TypeScript indicator calculations from the frontend. This is critical
 * for parity testing - we must test Python vs TypeScript, not Python vs Python.
 * 
 * Usage (inside frontend container):
 *   npx tsx scripts/generateGoldenMaster.ts
 * 
 * Output: ../backend/tests/fixtures/golden_master.json
 */

import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// Import the actual indicator functions used in the frontend
import { calculateAllIndicators } from '../src/services/technicalIndicators.js';
import type { HistoryPoint } from '../src/types/index.js';

// ============================================================================
// Configuration
// ============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const TICKER = 'AAPL';
const WARMUP_PERIOD = 50;
const OUTPUT_PATH = path.join(__dirname, '../../backend/tests/fixtures/golden_master.json');

// ============================================================================
// Signal Calculation (must match Python exactly)
// ============================================================================

function rsiToSignal(rsi: number | null, adxTrending: boolean): number | null {
  if (rsi === null) return null;
  
  // Regime override: Trending market (ADX > 25)
  if (adxTrending && rsi >= 65 && rsi < 85) {
    return 0.8;
  }
  
  // Base RSI signal (mean reversion)
  let signal: number;
  if (rsi <= 30) {
    signal = 1.0 - (rsi / 30) * 0.5;
  } else if (rsi <= 50) {
    signal = 0.5 - ((rsi - 30) / 20) * 0.5;
  } else if (rsi <= 70) {
    signal = -((rsi - 50) / 20) * 0.5;
  } else {
    signal = -0.5 - ((rsi - 70) / 30) * 0.5;
  }
  
  return Math.max(-1, Math.min(1, signal));
}

function macdToSignal(histogram: number | null, isBullish: boolean): number | null {
  if (histogram === null) return null;
  
  const histogramStrength = Math.min(Math.abs(histogram) / 2, 1);
  
  if (isBullish) {
    return Math.max(-1, Math.min(1, 0.3 + histogramStrength * 0.7));
  } else {
    return Math.max(-1, Math.min(1, -(0.3 + histogramStrength * 0.7)));
  }
}

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
  
  if (direction === 'bearish') {
    signal = -signal;
  }
  
  return Math.max(-1, Math.min(1, signal));
}

function cmfToSignal(cmf: number | null): number | null {
  if (cmf === null) return null;
  return Math.max(-1, Math.min(1, cmf * 2));
}

function momentumToSignal(shortTerm: number | null): number | null {
  if (shortTerm === null) return null;
  
  if (shortTerm > 10) return 1.0;
  if (shortTerm > 5) return 0.6;
  if (shortTerm > 0) return 0.2;
  if (shortTerm > -5) return -0.2;
  if (shortTerm > -10) return -0.6;
  return -1.0;
}

function volumeToSignal(volumeRatio: number | null): number | null {
  if (volumeRatio === null) return null;
  
  if (volumeRatio > 2.0) return 0.8;
  if (volumeRatio > 1.5) return 0.5;
  if (volumeRatio > 1.0) return 0.2;
  if (volumeRatio > 0.5) return -0.2;
  return -0.5;
}

function rvolToSignal(rvol: number | null): number | null {
  if (rvol === null) return null;
  
  if (rvol > 3.0) return 0.9;
  if (rvol > 2.0) return 0.6;
  if (rvol > 1.5) return 0.3;
  if (rvol >= 0.8) return 0.0;
  if (rvol >= 0.5) return -0.3;
  return -0.6;
}

function smaToSignal(
  currentPrice: number,
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

function positionToSignal(rangePosition: number | null): number | null {
  if (rangePosition === null) return null;
  return Math.max(-1, Math.min(1, 1.0 - (rangePosition * 2)));
}

function squeezeToSignal(isSqueeze: boolean, isExpansion: boolean): number {
  if (isSqueeze) return 0.8;
  if (isExpansion) return -0.5;
  return 0.0;
}

function round(val: number | null, decimals: number = 4): number | null {
  if (val === null || val === undefined || isNaN(val)) return null;
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

// ============================================================================
// Fetch Stock Data (via backend API)
// ============================================================================

async function fetchStockData(ticker: string): Promise<HistoryPoint[]> {
  console.log(`Fetching ${ticker} data from backend API...`);
  
  // Fetch from backend API (note: endpoint is /stock/ not /stocks/)
  const response = await fetch(`http://backend:8000/api/v1/stock/${ticker}/history?period=2y`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch data: ${response.status} ${response.statusText}`);
  }
  
  const data = await response.json();
  
  if (!data.history || data.history.length < 100) {
    throw new Error(`Insufficient data: got ${data.history?.length || 0} days`);
  }
  
  console.log(`Got ${data.history.length} days of data`);
  return data.history;
}

// ============================================================================
// Generate Golden Master
// ============================================================================

interface GoldenMasterDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signals: Record<string, number | null>;
}

async function generateGoldenMaster(): Promise<void> {
  console.log('='.repeat(60));
  console.log('  Generating Golden Master from TypeScript');
  console.log('='.repeat(60));
  
  // Fetch data
  const history = await fetchStockData(TICKER);
  
  console.log(`Processing ${history.length} data points...`);
  
  const dataPoints: GoldenMasterDataPoint[] = [];
  
  // Calculate indicators for each day using a sliding window
  for (let i = WARMUP_PERIOD; i < history.length; i++) {
    const windowHistory = history.slice(0, i + 1);
    const point = history[i];
    
    // Calculate indicators using the ACTUAL TypeScript functions
    const indicators = calculateAllIndicators(
      windowHistory,
      point.close,
      null, // high52w
      null  // low52w
    );
    
    // Extract raw values and compute signals
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
    
    dataPoints.push({
      date: point.date,
      open: round(point.open, 2)!,
      high: round(point.high, 2)!,
      low: round(point.low, 2)!,
      close: round(point.close, 2)!,
      volume: point.volume,
      signals: {
        rsi_signal: round(rsiToSignal(rsiValue, adxTrending)),
        rsi_value: round(rsiValue),
        macd_signal: round(macdToSignal(macdHistogram, macdBullish)),
        macd_line: round(macdLine),
        macd_histogram: round(macdHistogram),
        bollinger_signal: round(bollingerToSignal(bbPercentB)),
        bb_percent_b: round(bbPercentB),
        bb_bandwidth: round(bbBandwidth),
        adx_signal: round(adxToSignal(adxValue, adxDirection)),
        adx_value: round(adxValue),
        cmf_signal: round(cmfToSignal(cmfValue)),
        cmf_value: round(cmfValue),
        momentum_signal: round(momentumToSignal(momentumShort)),
        momentum_short: round(momentumShort),
        volume_signal: round(volumeToSignal(volumeRatio)),
        volume_ratio: round(volumeRatio),
        rvol_signal: round(rvolToSignal(rvolValue)),
        rvol_value: round(rvolValue),
        sma_signal: round(smaToSignal(point.close, sma50, sma200)),
        sma_20: round(sma20),
        sma_50: round(sma50),
        sma_200: round(sma200),
        position_signal: round(positionToSignal(rangePosition)),
        range_position: round(rangePosition),
        squeeze_signal: squeezeToSignal(isSqueeze, isExpansion),
        squeeze_percentile: squeezePercentile !== null ? Math.round(squeezePercentile) : null,
      }
    });
  }
  
  // Calculate summary ranges
  const rsiVals = dataPoints.map(d => d.signals.rsi_value).filter((v): v is number => v !== null);
  const macdVals = dataPoints.map(d => d.signals.macd_line).filter((v): v is number => v !== null);
  const adxVals = dataPoints.map(d => d.signals.adx_value).filter((v): v is number => v !== null);
  const cmfVals = dataPoints.map(d => d.signals.cmf_value).filter((v): v is number => v !== null);
  
  // Also include ALL price data (including warmup) for Python calculation
  const allPriceData = history.map(h => ({
    date: h.date,
    open: round(h.open, 2)!,
    high: round(h.high, 2)!,
    low: round(h.low, 2)!,
    close: round(h.close, 2)!,
    volume: h.volume
  }));
  
  const goldenMaster = {
    ticker: TICKER,
    exportedAt: new Date().toISOString(),
    source: 'TypeScript (frontend/src/services/technicalIndicators.ts)',
    dataRange: {
      start: dataPoints[0]?.date ?? '',
      end: dataPoints[dataPoints.length - 1]?.date ?? '',
      totalDays: dataPoints.length
    },
    warmupPeriod: WARMUP_PERIOD,
    // Full price history for Python to calculate indicators on
    priceHistory: allPriceData,
    // Signal values for comparison (after warmup)
    dataPoints,
    summary: {
      rsiRange: [round(Math.min(...rsiVals), 2), round(Math.max(...rsiVals), 2)],
      macdRange: [round(Math.min(...macdVals), 4), round(Math.max(...macdVals), 4)],
      adxRange: [round(Math.min(...adxVals), 2), round(Math.max(...adxVals), 2)],
      cmfRange: [round(Math.min(...cmfVals), 4), round(Math.max(...cmfVals), 4)],
    }
  };
  
  // Write to file
  fs.mkdirSync(path.dirname(OUTPUT_PATH), { recursive: true });
  fs.writeFileSync(OUTPUT_PATH, JSON.stringify(goldenMaster, null, 2));
  
  console.log('\n' + '='.repeat(60));
  console.log('  ✅ Golden Master Generated (from TypeScript!)');
  console.log('='.repeat(60));
  console.log(`  Ticker: ${TICKER}`);
  console.log(`  Date Range: ${goldenMaster.dataRange.start} to ${goldenMaster.dataRange.end}`);
  console.log(`  Total Days: ${goldenMaster.dataRange.totalDays}`);
  console.log(`  RSI Range: ${goldenMaster.summary.rsiRange}`);
  console.log(`  ADX Range: ${goldenMaster.summary.adxRange}`);
  console.log(`  Output: ${OUTPUT_PATH}`);
  console.log('='.repeat(60));
}

// Run
generateGoldenMaster().catch(err => {
  console.error('Error generating golden master:', err);
  process.exit(1);
});

