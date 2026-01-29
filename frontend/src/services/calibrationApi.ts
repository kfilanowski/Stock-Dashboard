/**
 * Calibration API Service
 *
 * Isolated API module for Walk-Forward Optimization endpoints.
 * Includes comprehensive error tracking with full context.
 */

import type {
  CalibrationRequest,
  CalibrationResponse,
  CalibrationProgress,
  WeightsResponse,
  BatchWeightsResponse,
  WeightMatrix,
  WeightDrift,
  ResonanceResponse
} from '../types/calibration';
import { trackError, trackApiError, type ErrorContext } from './errorTracking';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE = '/api/v1/calibration';

// ============================================================================
// Error Handling Helpers
// ============================================================================

/**
 * API error with full context
 */
export class CalibrationApiError extends Error {
  public readonly context: ErrorContext;
  public readonly httpStatus?: number;
  public readonly responseBody?: unknown;
  public readonly requestUrl: string;

  constructor(
    message: string,
    context: ErrorContext,
    httpStatus?: number,
    responseBody?: unknown,
    requestUrl?: string
  ) {
    super(message);
    this.name = 'CalibrationApiError';
    this.context = context;
    this.httpStatus = httpStatus;
    this.responseBody = responseBody;
    this.requestUrl = requestUrl || '';
  }
}

/**
 * Handle API response, tracking errors with full context
 */
async function handleResponse<T>(
  response: Response,
  context: ErrorContext
): Promise<T> {
  if (!response.ok) {
    const tracked = await trackApiError(response, context);
    throw new CalibrationApiError(
      tracked.message,
      context,
      response.status,
      tracked.responseBody,
      response.url
    );
  }

  try {
    return await response.json();
  } catch (err) {
    trackError(err, {
      ...context,
      operation: `${context.operation}:parseResponse`,
      metadata: { ...context.metadata, responseStatus: response.status }
    });
    throw new CalibrationApiError(
      'Failed to parse response JSON',
      context,
      response.status,
      undefined,
      response.url
    );
  }
}

/**
 * Wrap fetch with error tracking
 */
async function trackedFetch(
  url: string,
  options: RequestInit,
  context: ErrorContext
): Promise<Response> {
  try {
    const response = await fetch(url, options);
    return response;
  } catch (err) {
    // Network error, CORS, etc.
    trackError(err, {
      ...context,
      metadata: {
        ...context.metadata,
        url,
        method: options.method || 'GET',
        errorType: 'network'
      }
    });
    throw new CalibrationApiError(
      err instanceof Error ? err.message : 'Network request failed',
      context,
      undefined,
      undefined,
      url
    );
  }
}

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
  const ticker = request.ticker.toUpperCase();
  const horizons = request.horizons ?? [3, 15];
  const optimizer = request.optimizer ?? 'coordinate_descent';

  const context: ErrorContext = {
    operation: 'startCalibration',
    source: 'calibrationApi',
    ticker,
    params: { horizons, optimizer }
  };

  const response = await trackedFetch(
    `${API_BASE}/start`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, horizons, optimizer })
    },
    context
  );

  return handleResponse<CalibrationResponse>(response, context);
}

/**
 * Get calibrated weights for a ticker, horizon, and strategy class.
 */
export async function getCalibrationWeights(
  ticker: string,
  horizon: number = 3,
  strategyClass: string = 'directional'
): Promise<WeightsResponse> {
  const normalizedTicker = ticker.toUpperCase();

  const context: ErrorContext = {
    operation: 'getCalibrationWeights',
    source: 'calibrationApi',
    ticker: normalizedTicker,
    params: { horizon, strategyClass }
  };

  const response = await trackedFetch(
    `${API_BASE}/weights/${normalizedTicker}?horizon=${horizon}&strategy_class=${strategyClass}`,
    {},
    context
  );

  return handleResponse<WeightsResponse>(response, context);
}

/**
 * Get calibrated weights for multiple tickers.
 */
export async function getBatchWeights(
  tickers: string[],
  horizon: number = 3,
  strategyClass: string = 'all'
): Promise<BatchWeightsResponse> {
  const normalizedTickers = tickers.map(t => t.toUpperCase());
  const tickerStr = normalizedTickers.join(',');

  const context: ErrorContext = {
    operation: 'getBatchWeights',
    source: 'calibrationApi',
    params: { tickers: normalizedTickers, horizon, strategyClass }
  };

  const response = await trackedFetch(
    `${API_BASE}/weights/batch?tickers=${tickerStr}&horizon=${horizon}&strategy_class=${strategyClass}`,
    {},
    context
  );

  return handleResponse<BatchWeightsResponse>(response, context);
}

/**
 * Get default indicator weights.
 */
export async function getDefaultWeights(): Promise<{
  weights: WeightMatrix;
  description: string;
}> {
  const context: ErrorContext = {
    operation: 'getDefaultWeights',
    source: 'calibrationApi'
  };

  const response = await trackedFetch(`${API_BASE}/defaults`, {}, context);
  return handleResponse<{ weights: WeightMatrix; description: string }>(response, context);
}

// ============================================================================
// SSE Streaming
// ============================================================================

/**
 * Stream calibration progress via Server-Sent Events.
 */
export function streamCalibrationProgress(
  ticker: string,
  onProgress: (progress: CalibrationProgress) => void,
  onError?: (error: Error) => void,
  onComplete?: () => void,
  horizons?: number[]
): () => void {
  const normalizedTicker = ticker.toUpperCase();

  // Build URL with optional horizons query param
  let url = `${API_BASE}/stream/${normalizedTicker}`;
  if (horizons && horizons.length > 0) {
    url += `?horizons=${horizons.join(',')}`;
  }

  const context: ErrorContext = {
    operation: 'streamCalibrationProgress',
    source: 'calibrationApi',
    ticker: normalizedTicker,
    params: { horizons, url }
  };

  const eventSource = new EventSource(url);
  let messageCount = 0;

  eventSource.onmessage = (event) => {
    messageCount++;
    try {
      const progress: CalibrationProgress = JSON.parse(event.data);
      onProgress(progress);

      if (progress.stage === 'error') {
        trackError(progress.message || 'SSE reported error stage', {
          ...context,
          operation: 'streamCalibrationProgress:serverError',
          metadata: { messageCount, progress }
        }, 'warning');
        eventSource.close();
        onComplete?.();
      }
    } catch (err) {
      // Track parse errors
      trackError(err, {
        ...context,
        operation: 'streamCalibrationProgress:parseError',
        metadata: { messageCount, rawData: event.data?.substring(0, 500) }
      });

      onProgress({
        ticker: normalizedTicker,
        horizon: 0,
        stage: 'error',
        progress: 0,
        message: `SSE parse error (message #${messageCount}): ${err instanceof Error ? err.message : 'Invalid JSON'}`
      });
    }
  };

  eventSource.onerror = (_event) => {
    const readyState = eventSource.readyState;
    const errorMsg = readyState === EventSource.CLOSED
      ? 'SSE connection closed unexpectedly'
      : readyState === EventSource.CONNECTING
        ? 'SSE connection failed while reconnecting'
        : 'SSE connection failed - server may be unavailable';

    trackError(errorMsg, {
      ...context,
      operation: 'streamCalibrationProgress:connectionError',
      metadata: {
        messageCount,
        readyState,
        readyStateLabel: ['CONNECTING', 'OPEN', 'CLOSED'][readyState] || 'UNKNOWN'
      }
    });

    eventSource.close();
    onError?.(new CalibrationApiError(errorMsg, context, undefined, undefined, url));
  };

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
 * Returns { exists: boolean, error?: string } to distinguish "no calibration" from "error occurred".
 */
export async function hasCalibration(
  ticker: string,
  horizon: number = 3
): Promise<{ exists: boolean; error?: string }> {
  try {
    const result = await getCalibrationWeights(ticker, horizon);
    return { exists: !result.is_default };
  } catch (err) {
    return {
      exists: false,
      error: err instanceof Error ? err.message : 'Unknown error checking calibration'
    };
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
  const normalizedTicker = ticker.toUpperCase();

  const context: ErrorContext = {
    operation: 'getDataStatus',
    source: 'calibrationApi',
    ticker: normalizedTicker
  };

  const response = await trackedFetch(
    `${API_BASE}/data-status/${normalizedTicker}`,
    {},
    context
  );

  return handleResponse<DataStatus>(response, context);
}

/**
 * Get resonant horizons for a ticker.
 */
export async function getResonance(ticker: string): Promise<ResonanceResponse> {
  const normalizedTicker = ticker.toUpperCase();

  const context: ErrorContext = {
    operation: 'getResonance',
    source: 'calibrationApi',
    ticker: normalizedTicker
  };

  const response = await trackedFetch(
    `${API_BASE}/resonance/${normalizedTicker}`,
    {},
    context
  );

  return handleResponse<ResonanceResponse>(response, context);
}

/**
 * Fetch and cache price history for calibration.
 */
export async function fetchCalibrationData(ticker: string): Promise<FetchDataResult> {
  const normalizedTicker = ticker.toUpperCase();

  const context: ErrorContext = {
    operation: 'fetchCalibrationData',
    source: 'calibrationApi',
    ticker: normalizedTicker,
    metadata: { method: 'POST' }
  };

  const response = await trackedFetch(
    `${API_BASE}/fetch-data/${normalizedTicker}`,
    { method: 'POST' },
    context
  );

  return handleResponse<FetchDataResult>(response, context);
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
  const context: ErrorContext = {
    operation: 'getPortfolioTickers',
    source: 'calibrationApi'
  };

  const response = await trackedFetch(`${API_BASE}/tickers`, {}, context);
  const data = await handleResponse<{ tickers: string[] }>(response, context);
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
    strategy_class: string;
    weight: number;
    sqn_score: number | null;
    stability_passed: boolean;
    updated_at: string | null;
  }>;
  message: string;
}

/**
 * Verify calibration data exists in database.
 */
export async function verifyCalibration(ticker: string): Promise<CalibrationVerification> {
  const normalizedTicker = ticker.toUpperCase();

  const context: ErrorContext = {
    operation: 'verifyCalibration',
    source: 'calibrationApi',
    ticker: normalizedTicker
  };

  const response = await trackedFetch(
    `${API_BASE}/verify/${normalizedTicker}`,
    {},
    context
  );

  return handleResponse<CalibrationVerification>(response, context);
}

