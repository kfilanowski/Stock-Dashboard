/**
 * Calibration API Service
 * 
 * Isolated API module for Walk-Forward Optimization endpoints.
 * Does NOT modify existing api.ts - keeps calibration code separate.
 */

import type {
  CalibrationRequest,
  CalibrationResponse,
  CalibrationProgress,
  WeightsResponse,
  BatchWeightsResponse,
  WeightMatrix,
  CalibrationWindow,
  WeightDrift
} from '../types/calibration';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = '/api/v1/calibration';

// ============================================================================
// Core API Functions
// ============================================================================

/**
 * Start calibration for a ticker.
 * 
 * This runs the full WFO optimization pipeline:
 * 1. Loads price history from SQLite cache
 * 2. Runs two-pass coordinate descent
 * 3. Validates weight stability
 * 4. Saves results to database
 */
export async function startCalibration(
  request: CalibrationRequest
): Promise<CalibrationResponse> {
  const response = await fetch(`${API_BASE}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ticker: request.ticker.toUpperCase(),
      horizons: request.horizons ?? [3, 15]
    })
  });
  
  if (!response.ok) {
    throw new Error(`Calibration failed: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get calibrated weights for a ticker and horizon.
 * 
 * Returns default weights if no calibration exists (silent fallback).
 */
export async function getCalibrationWeights(
  ticker: string,
  horizon: number = 3
): Promise<WeightsResponse> {
  const response = await fetch(
    `${API_BASE}/weights/${ticker.toUpperCase()}?horizon=${horizon}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to get weights: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get calibrated weights for multiple tickers.
 * 
 * Useful for loading weights for entire portfolio at once.
 */
export async function getBatchWeights(
  tickers: string[],
  horizon: number = 3
): Promise<BatchWeightsResponse> {
  const tickerStr = tickers.map(t => t.toUpperCase()).join(',');
  const response = await fetch(
    `${API_BASE}/weights/batch?tickers=${tickerStr}&horizon=${horizon}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to get batch weights: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get default indicator weights.
 */
export async function getDefaultWeights(): Promise<{
  weights: WeightMatrix;
  description: string;
}> {
  const response = await fetch(`${API_BASE}/defaults`);
  
  if (!response.ok) {
    throw new Error(`Failed to get default weights: ${response.status}`);
  }
  
  return response.json();
}

// ============================================================================
// SSE Streaming
// ============================================================================

/**
 * Stream calibration progress via Server-Sent Events.
 * 
 * Usage:
 * ```ts
 * const cleanup = streamCalibrationProgress('AAPL', (progress) => {
 *   console.log(progress.stage, progress.progress);
 * });
 * 
 * // Later: cleanup();
 * ```
 */
export function streamCalibrationProgress(
  ticker: string,
  onProgress: (progress: CalibrationProgress) => void,
  onError?: (error: Error) => void,
  onComplete?: () => void
): () => void {
  const eventSource = new EventSource(
    `${API_BASE}/stream/${ticker.toUpperCase()}`
  );
  
  eventSource.onmessage = (event) => {
    try {
      const progress: CalibrationProgress = JSON.parse(event.data);
      onProgress(progress);
      
      // Close connection when complete or error
      if (progress.stage === 'complete' || progress.stage === 'error') {
        eventSource.close();
        onComplete?.();
      }
    } catch (err) {
      console.error('Failed to parse SSE data:', err);
    }
  };
  
  eventSource.onerror = (err) => {
    console.error('SSE connection error:', err);
    eventSource.close();
    onError?.(new Error('SSE connection failed'));
  };
  
  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

// ============================================================================
// Weight Drift Analysis
// ============================================================================

/**
 * Calculate weight drift between old and new weights.
 */
export function calculateWeightDrift(
  oldWeights: Partial<WeightMatrix>,
  newWeights: Partial<WeightMatrix>
): WeightDrift[] {
  const indicators = new Set([
    ...Object.keys(oldWeights),
    ...Object.keys(newWeights)
  ]);
  
  const drifts: WeightDrift[] = [];
  
  for (const indicator of indicators) {
    const oldVal = (oldWeights as Record<string, number>)[indicator] ?? 1.0;
    const newVal = (newWeights as Record<string, number>)[indicator] ?? 1.0;
    const drift = Math.abs(newVal - oldVal);
    
    let stability: 'stable' | 'moderate' | 'noisy';
    if (drift < 0.3) {
      stability = 'stable';
    } else if (drift < 0.7) {
      stability = 'moderate';
    } else {
      stability = 'noisy';
    }
    
    drifts.push({
      indicator,
      old_weight: oldVal,
      new_weight: newVal,
      drift,
      stability
    });
  }
  
  return drifts.sort((a, b) => b.drift - a.drift);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Check if calibration exists for a ticker.
 */
export async function hasCalibration(
  ticker: string,
  horizon: number = 3
): Promise<boolean> {
  try {
    const result = await getCalibrationWeights(ticker, horizon);
    return !result.is_default;
  } catch {
    return false;
  }
}

// ============================================================================
// Data Fetching for WFO
// ============================================================================

export interface DataStatus {
  ticker: string;
  available_days: number;
  required_days: number;
  has_sufficient_data: boolean;
  earliest_date: string | null;
  latest_date: string | null;
  message: string;
}

export interface FetchDataResult {
  status: 'success' | 'error';
  ticker: string;
  days_fetched: number;
  earliest_date?: string;
  latest_date?: string;
  message: string;
}

/**
 * Check how much historical data is available for WFO calibration.
 */
export async function getDataStatus(ticker: string): Promise<DataStatus> {
  const response = await fetch(
    `${API_BASE}/data-status/${ticker.toUpperCase()}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to get data status: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Fetch and cache 2 years of daily price history for calibration.
 * 
 * This fetches from Yahoo Finance and stores in SQLite.
 */
export async function fetchCalibrationData(
  ticker: string
): Promise<FetchDataResult> {
  const response = await fetch(
    `${API_BASE}/fetch-data/${ticker.toUpperCase()}`,
    { method: 'POST' }
  );
  
  if (!response.ok) {
    throw new Error(`Failed to fetch data: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get SQN interpretation label.
 */
export function getSQNLabel(sqn: number): {
  label: string;
  color: string;
  description: string;
} {
  if (sqn < 0) {
    return {
      label: 'Losing',
      color: 'text-red-500',
      description: 'Strategy loses money'
    };
  }
  if (sqn < 1.6) {
    return {
      label: 'Poor',
      color: 'text-red-400',
      description: 'Not tradeable'
    };
  }
  if (sqn < 2.0) {
    return {
      label: 'Below Average',
      color: 'text-amber-400',
      description: 'Barely tradeable'
    };
  }
  if (sqn < 2.5) {
    return {
      label: 'Average',
      color: 'text-yellow-400',
      description: 'Acceptable'
    };
  }
  if (sqn < 3.0) {
    return {
      label: 'Good',
      color: 'text-green-400',
      description: 'Solid strategy'
    };
  }
  if (sqn < 5.0) {
    return {
      label: 'Excellent',
      color: 'text-cyan-400',
      description: 'Very reliable'
    };
  }
  return {
    label: 'Superb',
    color: 'text-purple-400',
    description: 'Exceptional'
  };
}

// ============================================================================
// Portfolio Integration
// ============================================================================

/**
 * Get list of portfolio tickers for the calibration picker.
 */
export async function getPortfolioTickers(): Promise<string[]> {
  const response = await fetch(`${API_BASE}/tickers`);
  
  if (!response.ok) {
    throw new Error(`Failed to get tickers: ${response.status}`);
  }
  
  const data = await response.json();
  return data.tickers || [];
}

// ============================================================================
// Verification
// ============================================================================

export interface CalibrationVerification {
  ticker: string;
  verified: boolean;
  weights_count: number;
  windows_count: number;
  trades_count: number;
  weights: Array<{
    indicator: string;
    horizon: number;
    weight: number;
    sqn_score: number | null;
    stability_passed: boolean;
    updated_at: string | null;
  }>;
  message: string;
}

/**
 * Verify calibration data exists in database.
 * 
 * This confirms that calibration actually saved, not just returned success.
 */
export async function verifyCalibration(
  ticker: string
): Promise<CalibrationVerification> {
  const response = await fetch(
    `${API_BASE}/verify/${ticker.toUpperCase()}`
  );
  
  if (!response.ok) {
    throw new Error(`Failed to verify calibration: ${response.status}`);
  }
  
  return response.json();
}

