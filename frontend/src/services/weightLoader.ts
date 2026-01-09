/**
 * Weight Loader Service
 * 
 * Isolated weight loading with caching.
 * Silent fallback to defaults - never breaks existing flow.
 */

import * as calibrationApi from './calibrationApi';
import type { WeightMatrix } from '../types/calibration';

// ============================================================================
// Default Weights
// ============================================================================

/**
 * Default weights used when no calibration exists.
 * All indicators weighted equally at 1.0.
 */
export const DEFAULT_WEIGHTS: WeightMatrix = {
  rsi: 1.0,
  macd: 1.0,
  bollinger: 1.0,
  adx: 1.0,
  cmf: 1.0,
  momentum: 1.0,
  volume: 1.0,
  rvol: 1.0,
  sma: 1.0,
  position: 1.0,
  squeeze: 1.0,
  cross: 1.0
};

// ============================================================================
// Weight Cache
// ============================================================================

interface CacheEntry {
  weights: WeightMatrix;
  isDefault: boolean;
  expiry: number;
}

// Cache with 5-minute TTL
const CACHE_TTL_MS = 5 * 60 * 1000;
const weightCache = new Map<string, CacheEntry>();

/**
 * Generate cache key for ticker/horizon combination.
 */
function getCacheKey(ticker: string, horizon: number): string {
  return `${ticker.toUpperCase()}_${horizon}`;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Get weights for a stock, with silent fallback to defaults.
 * 
 * This function NEVER throws - it always returns valid weights.
 * If calibration fails or doesn't exist, it silently uses defaults.
 * 
 * @param ticker - Stock ticker symbol
 * @param horizon - Holding period (3 for swing, 15 for trend)
 * @returns Weight matrix (either calibrated or default)
 */
export async function getWeightsForStock(
  ticker: string,
  horizon: number = 3
): Promise<{ weights: WeightMatrix; isCalibrated: boolean }> {
  const cacheKey = getCacheKey(ticker, horizon);
  
  // Check cache first
  const cached = weightCache.get(cacheKey);
  if (cached && Date.now() < cached.expiry) {
    return {
      weights: cached.weights,
      isCalibrated: !cached.isDefault
    };
  }
  
  // Try to fetch calibrated weights
  try {
    const response = await calibrationApi.getCalibrationWeights(ticker, horizon);
    
    if (response && !response.is_default) {
      // Merge with defaults to ensure all keys exist
      const merged = mergeWeights(DEFAULT_WEIGHTS, response.weights);
      
      // Cache the result
      weightCache.set(cacheKey, {
        weights: merged,
        isDefault: false,
        expiry: Date.now() + CACHE_TTL_MS
      });
      
      return {
        weights: merged,
        isCalibrated: true
      };
    }
  } catch (err) {
    // Silent fallback - log but don't throw
    console.debug(`[weightLoader] Using defaults for ${ticker}:`, err);
  }
  
  // Return defaults (and cache them)
  weightCache.set(cacheKey, {
    weights: DEFAULT_WEIGHTS,
    isDefault: true,
    expiry: Date.now() + CACHE_TTL_MS
  });
  
  return {
    weights: DEFAULT_WEIGHTS,
    isCalibrated: false
  };
}

/**
 * Get weights for multiple stocks.
 * 
 * Efficient batch loading with fallback to individual requests.
 */
export async function getWeightsForStocks(
  tickers: string[],
  horizon: number = 3
): Promise<Map<string, { weights: WeightMatrix; isCalibrated: boolean }>> {
  const results = new Map<string, { weights: WeightMatrix; isCalibrated: boolean }>();
  
  // Check cache first
  const uncached: string[] = [];
  for (const ticker of tickers) {
    const cacheKey = getCacheKey(ticker, horizon);
    const cached = weightCache.get(cacheKey);
    
    if (cached && Date.now() < cached.expiry) {
      results.set(ticker.toUpperCase(), {
        weights: cached.weights,
        isCalibrated: !cached.isDefault
      });
    } else {
      uncached.push(ticker);
    }
  }
  
  // Fetch uncached weights
  if (uncached.length > 0) {
    try {
      const batchResponse = await calibrationApi.getBatchWeights(uncached, horizon);
      
      for (const [ticker, data] of Object.entries(batchResponse.tickers)) {
        const merged = mergeWeights(DEFAULT_WEIGHTS, data.weights);
        
        weightCache.set(getCacheKey(ticker, horizon), {
          weights: merged,
          isDefault: data.is_default,
          expiry: Date.now() + CACHE_TTL_MS
        });
        
        results.set(ticker, {
          weights: merged,
          isCalibrated: !data.is_default
        });
      }
    } catch (err) {
      // Fallback: use defaults for all uncached
      console.debug('[weightLoader] Batch fetch failed, using defaults:', err);
      
      for (const ticker of uncached) {
        results.set(ticker.toUpperCase(), {
          weights: DEFAULT_WEIGHTS,
          isCalibrated: false
        });
      }
    }
  }
  
  return results;
}

/**
 * Invalidate cache for a ticker (call after calibration).
 */
export function invalidateCache(ticker: string, horizon?: number): void {
  if (horizon !== undefined) {
    weightCache.delete(getCacheKey(ticker, horizon));
  } else {
    // Invalidate all horizons
    for (const h of [3, 15]) {
      weightCache.delete(getCacheKey(ticker, h));
    }
  }
}

/**
 * Clear entire weight cache.
 */
export function clearCache(): void {
  weightCache.clear();
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Merge calibrated weights with defaults.
 * Ensures all indicator keys exist in the result.
 */
function mergeWeights(
  defaults: WeightMatrix,
  calibrated: Partial<WeightMatrix>
): WeightMatrix {
  return {
    ...defaults,
    ...calibrated
  };
}

/**
 * Check if a ticker has been calibrated (without fetching full weights).
 */
export async function isCalibrated(
  ticker: string,
  horizon: number = 3
): Promise<boolean> {
  const { isCalibrated } = await getWeightsForStock(ticker, horizon);
  return isCalibrated;
}

