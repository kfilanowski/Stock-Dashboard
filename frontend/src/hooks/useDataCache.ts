/**
 * Data Cache Hooks
 * 
 * Convenient hooks for accessing cached data in components.
 * All hooks consume from the centralized DataCacheContext.
 */

import { useMemo, useState, useEffect } from 'react';
import { useDataCacheContext } from '../context/DataCacheContext';
import type { PriceData, HistoryCacheEntry } from '../services/dataCache';
import type { HistoryPoint } from '../types';

// ============================================================================
// Price Hooks
// ============================================================================

/**
 * Get cached price data for a single ticker.
 */
export function usePrice(ticker: string | null): PriceData | null {
  const { prices } = useDataCacheContext();
  return ticker ? prices.get(ticker) ?? null : null;
}

/**
 * Get cached prices for multiple tickers.
 */
export function usePrices(tickers: string[]): Map<string, PriceData> {
  const { prices } = useDataCacheContext();
  
  return useMemo(() => {
    const result = new Map<string, PriceData>();
    for (const ticker of tickers) {
      const price = prices.get(ticker);
      if (price) result.set(ticker, price);
    }
    return result;
  }, [prices, tickers.join(',')]);
}

/**
 * Get all cached prices.
 */
export function useAllPrices(): Map<string, PriceData> {
  const { prices } = useDataCacheContext();
  return prices;
}

// ============================================================================
// History Hooks
// ============================================================================

/**
 * Chart data shape expected by components (matches HoldingChartData).
 */
export interface ChartData {
  history: HistoryPoint[];
  referenceClose: number | null;
  isComplete: boolean;
  expectedStart: string | null;
  actualStart: string | null;
}

/**
 * Get cached history for a single ticker.
 * Returns the data in the component-friendly ChartData format.
 */
export function useHistory(ticker: string | null): ChartData | null {
  const { historyData, chartPeriod } = useDataCacheContext();
  
  return useMemo(() => {
    if (!ticker) return null;
    const entry = historyData.get(ticker);
    if (!entry || entry.period !== chartPeriod) return null;
    
    return {
      history: entry.history,
      referenceClose: entry.referenceClose,
      isComplete: entry.isComplete,
      expectedStart: entry.expectedStart,
      actualStart: entry.actualStart
    };
  }, [ticker, historyData, chartPeriod]);
}

/**
 * Get cached history for multiple tickers.
 */
export function useHistoryBatch(tickers: string[]): Map<string, ChartData> {
  const { historyData, chartPeriod } = useDataCacheContext();
  
  return useMemo(() => {
    const result = new Map<string, ChartData>();
    for (const ticker of tickers) {
      const entry = historyData.get(ticker);
      if (entry && entry.period === chartPeriod) {
        result.set(ticker, {
          history: entry.history,
          referenceClose: entry.referenceClose,
          isComplete: entry.isComplete,
          expectedStart: entry.expectedStart,
          actualStart: entry.actualStart
        });
      }
    }
    return result;
  }, [historyData, chartPeriod, tickers.join(',')]);
}

/**
 * Get all cached history data.
 */
export function useAllHistory(): Map<string, HistoryCacheEntry> {
  const { historyData } = useDataCacheContext();
  return historyData;
}

// ============================================================================
// History for Different Periods (Modals)
// ============================================================================

/**
 * Get history for a specific period (may differ from global chart period).
 * Used by modals that need to view different time periods.
 * 
 * This hook will fetch data if not cached for the requested period.
 */
export function useHistoryForPeriod(
  ticker: string | null, 
  period: string
): { data: ChartData | null; isLoading: boolean } {
  const { cache, historyData, chartPeriod } = useDataCacheContext();
  const [localData, setLocalData] = useState<HistoryCacheEntry | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  useEffect(() => {
    if (!ticker) {
      setLocalData(null);
      return;
    }
    
    // If period matches global period, use cached data
    if (period === chartPeriod) {
      const cached = historyData.get(ticker);
      if (cached && cached.period === period) {
        setLocalData(cached);
        return;
      }
    }
    
    // Check if we already have local data for this period
    if (localData && localData.period === period) {
      return;
    }
    
    // Fetch data for this specific period
    setIsLoading(true);
    cache.fetchSingleHistory(ticker, period)
      .then(entry => {
        if (entry) {
          setLocalData(entry);
        }
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [ticker, period, chartPeriod, cache, historyData, localData?.period]);
  
  // Clear local data when ticker changes
  useEffect(() => {
    setLocalData(null);
  }, [ticker]);
  
  const data = useMemo((): ChartData | null => {
    // Prefer cached data if period matches
    if (period === chartPeriod && ticker) {
      const cached = historyData.get(ticker);
      if (cached && cached.period === period) {
        return {
          history: cached.history,
          referenceClose: cached.referenceClose,
          isComplete: cached.isComplete,
          expectedStart: cached.expectedStart,
          actualStart: cached.actualStart
        };
      }
    }
    
    // Fall back to locally fetched data
    if (localData && localData.period === period) {
      return {
        history: localData.history,
        referenceClose: localData.referenceClose,
        isComplete: localData.isComplete,
        expectedStart: localData.expectedStart,
        actualStart: localData.actualStart
      };
    }
    
    return null;
  }, [ticker, period, chartPeriod, historyData, localData]);
  
  return { data, isLoading };
}

// ============================================================================
// Chart Period Hook
// ============================================================================

/**
 * Access and control the global chart period.
 */
export function useChartPeriod(): {
  chartPeriod: string;
  setChartPeriod: (period: string) => Promise<void>;
} {
  const { chartPeriod, setChartPeriod } = useDataCacheContext();
  return { chartPeriod, setChartPeriod };
}

// ============================================================================
// Ticker Management Hook
// ============================================================================

/**
 * Manage tracked tickers.
 */
export function useTickers(): {
  setTickers: (tickers: string[]) => void;
  refreshHistory: (ticker: string) => Promise<void>;
} {
  const { setTickers, refreshHistory } = useDataCacheContext();
  return { setTickers, refreshHistory };
}

// ============================================================================
// Status Hooks
// ============================================================================

/**
 * Get cache status and timestamps.
 */
export function useCacheStatus(): {
  lastPricesFetched: Date | null;
  lastHistoryFetched: Date | null;
  isRefreshing: boolean;
} {
  const { lastPricesFetched, lastHistoryFetched, isRefreshing } = useDataCacheContext();
  return { lastPricesFetched, lastHistoryFetched, isRefreshing };
}

// ============================================================================
// Combined Hook for Components
// ============================================================================

/**
 * Get all data for a single ticker (price + history).
 * Convenient for HoldingCard-style components.
 */
export function useTickerData(ticker: string | null): {
  price: PriceData | null;
  chartData: ChartData | null;
  isHistoryLoading: boolean;
} {
  const { prices, historyData, chartPeriod } = useDataCacheContext();
  
  const price = ticker ? prices.get(ticker) ?? null : null;
  
  const chartData = useMemo((): ChartData | null => {
    if (!ticker) return null;
    const entry = historyData.get(ticker);
    if (!entry || entry.period !== chartPeriod) return null;
    
    return {
      history: entry.history,
      referenceClose: entry.referenceClose,
      isComplete: entry.isComplete,
      expectedStart: entry.expectedStart,
      actualStart: entry.actualStart
    };
  }, [ticker, historyData, chartPeriod]);
  
  // History is loading if we don't have data yet
  const isHistoryLoading = ticker !== null && chartData === null;
  
  return { price, chartData, isHistoryLoading };
}

/**
 * Calculate period gain for a ticker based on cached history.
 */
export function usePeriodGain(ticker: string | null): number | null {
  const { chartData } = useTickerData(ticker);
  
  return useMemo(() => {
    if (!chartData?.history?.length || chartData.referenceClose === null || chartData.referenceClose === 0) {
      return null;
    }
    const latestClose = chartData.history[chartData.history.length - 1]?.close ?? 0;
    return ((latestClose - chartData.referenceClose) / chartData.referenceClose) * 100;
  }, [chartData]);
}

