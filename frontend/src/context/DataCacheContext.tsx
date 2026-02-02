/**
 * Data Cache Context
 * 
 * React Context provider for the centralized data cache.
 * Manages the cache lifecycle and exposes data to components.
 */

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import {
  DataCacheService,
  getDataCache,
  type PriceData,
  type HistoryCacheEntry,
  type ExtendedHistoryEntry
} from '../services/dataCache';

// ============================================================================
// Context Types
// ============================================================================

interface DataCacheContextValue {
  // The cache service instance
  cache: DataCacheService;

  // Price data (reactive state for components)
  prices: Map<string, PriceData>;

  // History data (reactive state for components)
  historyData: Map<string, HistoryCacheEntry>;

  // Extended history for S/R calculations (always 3-month daily data)
  extendedHistoryData: Map<string, ExtendedHistoryEntry>;

  // Timestamps for UI indicators
  lastPricesFetched: Date | null;
  lastHistoryFetched: Date | null;

  // Current chart period
  chartPeriod: string;

  // Actions
  setChartPeriod: (period: string) => Promise<void>;
  setTickers: (tickers: string[]) => void;
  refreshHistory: (ticker: string) => Promise<void>;

  // Status
  isRefreshing: boolean;
}

const DataCacheContext = createContext<DataCacheContextValue | null>(null);

// ============================================================================
// Provider Component
// ============================================================================

interface DataCacheProviderProps {
  children: React.ReactNode;
  /** Initial tickers to track (optional, can be set later) */
  initialTickers?: string[];
  /** Initial chart period (default: '1d') */
  initialChartPeriod?: string;
}

export function DataCacheProvider({ 
  children, 
  initialTickers = [],
  initialChartPeriod = '1d'
}: DataCacheProviderProps) {
  // Get singleton cache instance
  const cacheRef = useRef<DataCacheService>(getDataCache());
  const cache = cacheRef.current;
  
  // Reactive state for components
  const [prices, setPrices] = useState<Map<string, PriceData>>(new Map());
  const [historyData, setHistoryData] = useState<Map<string, HistoryCacheEntry>>(new Map());
  const [extendedHistoryData, setExtendedHistoryData] = useState<Map<string, ExtendedHistoryEntry>>(new Map());
  const [lastPricesFetched, setLastPricesFetched] = useState<Date | null>(null);
  const [lastHistoryFetched, setLastHistoryFetched] = useState<Date | null>(null);
  const [chartPeriod, setChartPeriodState] = useState(initialChartPeriod);
  const [isRefreshing, setIsRefreshing] = useState(false);
  
  // Track if mounted for cleanup
  const mountedRef = useRef(true);
  // Track if we've already started the loop (handles StrictMode double-mount)
  const loopStartedRef = useRef(false);
  
  // Initialize cache with initial values
  useEffect(() => {
    mountedRef.current = true;
    
    // Set initial period (don't trigger fetch yet)
    if (cache.getChartPeriod() !== initialChartPeriod) {
      // Directly set period without fetching (we'll fetch when refresh starts)
      (cache as any).chartPeriod = initialChartPeriod;
    }
    
    // Set initial tickers if provided
    if (initialTickers.length > 0) {
      cache.setTickers(initialTickers);
    }
    
    // Subscribe to price updates
    const unsubPrice = cache.onPriceUpdate((ticker, data) => {
      if (!mountedRef.current) return;
      setPrices(prev => new Map(prev).set(ticker, data));
      setLastPricesFetched(cache.getLastPricesFetchedAt());
    });
    
    // Subscribe to history updates
    const unsubHistory = cache.onHistoryUpdate((ticker, data) => {
      if (!mountedRef.current) return;
      setHistoryData(prev => new Map(prev).set(ticker, data));
      setLastHistoryFetched(cache.getLastHistoryFetchedAt());
    });
    
    // Subscribe to batch history completions
    const unsubBatch = cache.onBatchHistoryComplete(() => {
      if (!mountedRef.current) return;
      // Sync all history data
      setHistoryData(new Map(cache.getAllHistory()));
      setLastHistoryFetched(cache.getLastHistoryFetchedAt());
    });

    // Subscribe to extended history updates (for S/R calculations)
    const unsubExtended = cache.onExtendedHistoryUpdate((ticker, data) => {
      if (!mountedRef.current) return;
      setExtendedHistoryData(prev => new Map(prev).set(ticker, data));
    });

    // Start the refresh loop (only once, handles StrictMode double-mount)
    // The cache is a singleton - we don't stop it on unmount because other
    // components may still be using it. The loop runs for the app's lifetime.
    if (!loopStartedRef.current && !cache.isRefreshing()) {
      loopStartedRef.current = true;
      setIsRefreshing(true);
      cache.startRefreshLoop().catch(console.error);
    } else if (cache.isRefreshing()) {
      setIsRefreshing(true);
    }
    
    return () => {
      mountedRef.current = false;
      unsubPrice();
      unsubHistory();
      unsubBatch();
      unsubExtended();
      // NOTE: We intentionally do NOT stop the refresh loop on unmount.
      // The cache is a singleton shared across the app, and stopping it
      // would break other components. React StrictMode would also cause
      // issues by stopping/starting the loop during its double-mount cycle.
    };
  }, []); // Only run once on mount
  
  // Handle chart period change
  const handleSetChartPeriod = useCallback(async (period: string) => {
    setChartPeriodState(period);
    await cache.setChartPeriod(period);
    // Sync history data after period change
    if (mountedRef.current) {
      setHistoryData(new Map(cache.getAllHistory()));
    }
  }, [cache]);
  
  // Handle tickers change
  const handleSetTickers = useCallback((tickers: string[]) => {
    cache.setTickers(tickers);

    // Clean up removed tickers from local state
    const tickerSet = new Set(tickers);

    setPrices(prev => {
      const next = new Map<string, PriceData>();
      for (const [t, data] of prev) {
        if (tickerSet.has(t)) next.set(t, data);
      }
      return next;
    });

    setHistoryData(prev => {
      const next = new Map<string, HistoryCacheEntry>();
      for (const [t, data] of prev) {
        if (tickerSet.has(t)) next.set(t, data);
      }
      return next;
    });

    setExtendedHistoryData(prev => {
      const next = new Map<string, ExtendedHistoryEntry>();
      for (const [t, data] of prev) {
        if (tickerSet.has(t)) next.set(t, data);
      }
      return next;
    });
  }, [cache]);
  
  // Refresh history for a specific ticker (user action)
  const handleRefreshHistory = useCallback(async (ticker: string) => {
    cache.clearTickerHistory(ticker);
    
    // Remove from local state to show loading
    setHistoryData(prev => {
      const next = new Map(prev);
      next.delete(ticker);
      return next;
    });
    
    // Fetch fresh data
    await cache.fetchBatchHistory([ticker]);
    
    // Sync state
    if (mountedRef.current) {
      const entry = cache.getHistory(ticker);
      if (entry) {
        setHistoryData(prev => new Map(prev).set(ticker, entry));
      }
    }
  }, [cache]);
  
  const value: DataCacheContextValue = {
    cache,
    prices,
    historyData,
    extendedHistoryData,
    lastPricesFetched,
    lastHistoryFetched,
    chartPeriod,
    setChartPeriod: handleSetChartPeriod,
    setTickers: handleSetTickers,
    refreshHistory: handleRefreshHistory,
    isRefreshing
  };
  
  return (
    <DataCacheContext.Provider value={value}>
      {children}
    </DataCacheContext.Provider>
  );
}

// ============================================================================
// Hook to access the context
// ============================================================================

/**
 * Access the data cache context.
 * Must be used within a DataCacheProvider.
 */
export function useDataCacheContext(): DataCacheContextValue {
  const context = useContext(DataCacheContext);
  if (!context) {
    throw new Error('useDataCacheContext must be used within a DataCacheProvider');
  }
  return context;
}

