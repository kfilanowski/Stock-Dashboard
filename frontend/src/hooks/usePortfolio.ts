/**
 * Portfolio Hook
 * 
 * Manages portfolio state (holdings, add/remove/update) and integrates
 * with the centralized DataCacheService for price and history data.
 * 
 * The hook:
 * - Fetches portfolio structure from API
 * - Delegates price/history fetching to DataCacheService
 * - Updates portfolio prices when cache updates
 * - Provides CRUD operations for holdings
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { Portfolio, Holding, HistoryPoint } from '../types';
import * as api from '../services/api';
import { getDataCache, type PriceData, type HistoryCacheEntry } from '../services/dataCache';

// Re-export HoldingChartData for backwards compatibility
export interface HoldingChartData {
  history: HistoryPoint[];
  referenceClose: number | null;
  isComplete: boolean;
  expectedStart: string | null;
  actualStart: string | null;
}

interface UsePortfolioOptions {
  chartPeriod?: string;
  onHistoryUpdate?: (ticker: string, data: HoldingChartData) => void;
}

export function usePortfolio(options: UsePortfolioOptions = {}) {
  const { 
    chartPeriod = '1d',
    onHistoryUpdate 
  } = options;

  // Core state
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const [lastPricesFetched, setLastPricesFetched] = useState<Date | null>(null);
  
  // Refs
  const isInitialLoad = useRef(true);
  const isMounted = useRef(true);
  const portfolioRef = useRef<Portfolio | null>(null);
  const chartPeriodRef = useRef(chartPeriod);
  const onHistoryUpdateRef = useRef(onHistoryUpdate);
  const cacheInitializedRef = useRef(false);
  
  // Get the singleton cache instance
  const cache = getDataCache();

  // Sync portfolio ref
  useEffect(() => {
    portfolioRef.current = portfolio;
  }, [portfolio]);
  
  // Sync callback ref
  useEffect(() => {
    onHistoryUpdateRef.current = onHistoryUpdate;
  }, [onHistoryUpdate]);
  
  // Handle chart period changes
  useEffect(() => {
    const prevPeriod = chartPeriodRef.current;
    chartPeriodRef.current = chartPeriod;
    
    // Skip initial mount
    if (prevPeriod === chartPeriod) return;
    
    // Update cache period (this triggers batch history refetch internally)
    cache.setChartPeriod(chartPeriod).catch(console.error);
  }, [chartPeriod, cache]);
  
  // Update portfolio prices from cache data
  const updatePortfolioPrices = useCallback((priceData: Map<string, PriceData>) => {
    setPortfolio((prev: Portfolio | null) => {
      if (!prev) return prev;
      
      // First pass: calculate market values for allocation percentages
      let totalMarketValue = 0;
      let totalCostBasis = 0;
      
      const holdingsWithPrices = prev.holdings.map((h: Holding) => {
        const price = priceData.get(h.ticker);
        const currentPrice = price?.currentPrice ?? h.current_price ?? 0;
        const marketValue = h.shares * currentPrice;
        const costBasis = h.shares && h.avg_cost ? h.shares * h.avg_cost : null;
        
        totalMarketValue += marketValue;
        if (costBasis !== null) totalCostBasis += costBasis;
        
        return {
          ...h,
          current_price: currentPrice,
          market_value: Math.round(marketValue * 100) / 100,
          cost_basis: costBasis ? Math.round(costBasis * 100) / 100 : null,
          // Include 52-week data for analysis (only update if we have new data)
          high_52w: price?.high52w ?? h.high_52w ?? null,
          low_52w: price?.low52w ?? h.low_52w ?? null,
        };
      });
      
      // Second pass: calculate allocations and gain/loss
      const updatedHoldings = holdingsWithPrices.map((h) => {
        const allocationPct = totalMarketValue > 0 ? (h.market_value! / totalMarketValue) * 100 : 0;
        let gainLoss = null;
        let gainLossPct = null;
        
        if (h.market_value !== null && h.cost_basis !== null) {
          gainLoss = h.market_value - h.cost_basis;
          if (h.avg_cost && h.avg_cost > 0) {
            gainLossPct = ((h.current_price! - h.avg_cost) / h.avg_cost) * 100;
          }
        }
        
        return {
          ...h,
          allocation_pct: Math.round(allocationPct * 100) / 100,
          gain_loss: gainLoss !== null ? Math.round(gainLoss * 100) / 100 : null,
          gain_loss_pct: gainLossPct !== null ? Math.round(gainLossPct * 100) / 100 : null,
        };
      });
      
      const totalGainLoss = totalCostBasis > 0 ? totalMarketValue - totalCostBasis : null;
      const totalGainLossPct = totalCostBasis > 0 && totalGainLoss !== null 
        ? (totalGainLoss / totalCostBasis) * 100 
        : null;
      
      return {
        ...prev,
        holdings: updatedHoldings,
        total_market_value: Math.round(totalMarketValue * 100) / 100,
        total_cost_basis: totalCostBasis > 0 ? Math.round(totalCostBasis * 100) / 100 : null,
        total_gain_loss: totalGainLoss !== null ? Math.round(totalGainLoss * 100) / 100 : null,
        total_gain_loss_pct: totalGainLossPct !== null ? Math.round(totalGainLossPct * 100) / 100 : null,
      };
    });
    
    const now = new Date();
    setLastFetched(now);
    setLastPricesFetched(now);
  }, []);
  
  // Convert cache entry to HoldingChartData format
  const toChartData = useCallback((entry: HistoryCacheEntry): HoldingChartData => ({
    history: entry.history,
    referenceClose: entry.referenceClose,
    isComplete: entry.isComplete,
    expectedStart: entry.expectedStart,
    actualStart: entry.actualStart
  }), []);
  
  // Subscribe to cache updates
  useEffect(() => {
    isMounted.current = true;
    
    // Subscribe to price updates
    const unsubPrice = cache.onPriceUpdate((_ticker, _data) => {
      if (!isMounted.current) return;
      // Update portfolio with all current prices
      updatePortfolioPrices(cache.getAllPrices());
    });
    
    // Subscribe to history updates
    const unsubHistory = cache.onHistoryUpdate((ticker, entry) => {
      if (!isMounted.current) return;
      // Notify parent component of history update
      if (onHistoryUpdateRef.current) {
        onHistoryUpdateRef.current(ticker, toChartData(entry));
      }
    });
    
    // Subscribe to batch history completions
    const unsubBatch = cache.onBatchHistoryComplete(() => {
      if (!isMounted.current) return;
      // Notify parent of all history updates
      for (const [ticker, entry] of cache.getAllHistory()) {
        if (onHistoryUpdateRef.current) {
          onHistoryUpdateRef.current(ticker, toChartData(entry));
        }
      }
    });
    
    return () => {
      isMounted.current = false;
      unsubPrice();
      unsubHistory();
      unsubBatch();
    };
  }, [cache, updatePortfolioPrices, toChartData]);
  
  // Fetch portfolio from API
  const fetchPortfolio = useCallback(async (showLoading = true, useLiteMode = false) => {
    if (showLoading && isInitialLoad.current) {
      setLoading(true);
    }
    
    try {
      const data = await api.getPortfolio(useLiteMode);
      setPortfolio(data);
      setError(null);
      setLastFetched(new Date());
      isInitialLoad.current = false;
      
      // Update cache with current tickers - this triggers fetch for new tickers
      const tickers = data.holdings.map(h => h.ticker);
      cache.setTickers(tickers);
      
      // Set chart period on first load (the provider starts the refresh loop)
      if (!cacheInitializedRef.current) {
        cacheInitializedRef.current = true;
        // Sync chart period with cache
        if (cache.getChartPeriod() !== chartPeriodRef.current) {
          cache.setChartPeriod(chartPeriodRef.current).catch(console.error);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio');
    } finally {
      setLoading(false);
    }
  }, [cache]);
  
  // Initial load
  useEffect(() => {
    fetchPortfolio(true, true); // Lite mode for instant load
    
    return () => {
      // Don't stop the cache on unmount - it's shared across the app
      // The DataCacheProvider handles lifecycle
    };
  }, [fetchPortfolio]);
  
  // Update portfolio value
  const updatePortfolioValue = async (total_value: number) => {
    try {
      const updated = await api.updatePortfolio({ total_value });
      setPortfolio((prev: Portfolio | null) => prev ? { ...prev, total_value: updated.total_value } : prev);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update portfolio');
    }
  };
  
  // Add a new holding
  const addHolding = async (ticker: string, shares: number = 0, avgCost?: number) => {
    await api.addHolding({ ticker, shares, avg_cost: avgCost });
    
    const upperTicker = ticker.toUpperCase();
    
    // Create a temporary holding object
    const tempHolding: Holding = {
      id: Date.now(),
      portfolio_id: 0,
      ticker: upperTicker,
      shares,
      avg_cost: avgCost,
      added_at: new Date().toISOString(),
    };
    
    setPortfolio((prev: Portfolio | null) => {
      if (!prev) return prev;
      return {
        ...prev,
        holdings: [
          ...prev.holdings,
          tempHolding
        ]
      };
    });
    
    // Update cache tickers (will trigger fetch for new ticker)
    const currentTickers = cache.getTickers();
    if (!currentTickers.includes(upperTicker)) {
      cache.setTickers([...currentTickers, upperTicker]);
    }
    
    // Also refresh portfolio data from server
    fetchPortfolio(false);
  };
  
  // Remove a holding
  const removeHolding = async (holdingId: number) => {
    try {
      const holding = portfolioRef.current?.holdings.find((h: Holding) => h.id === holdingId);
      
      await api.deleteHolding(holdingId);
      
      setPortfolio((prev: Portfolio | null) => {
        if (!prev) return prev;
        return {
          ...prev,
          holdings: prev.holdings.filter((h: Holding) => h.id !== holdingId)
        };
      });
      
      // Update cache tickers (will clean up removed ticker)
      if (holding) {
        const remainingTickers = portfolioRef.current?.holdings
          .filter(h => h.id !== holdingId)
          .map(h => h.ticker) ?? [];
        cache.setTickers(remainingTickers);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove holding');
    }
  };
  
  // Update a holding
  const updateHolding = async (holdingId: number, data: { shares?: number; avg_cost?: number; is_pinned?: boolean }) => {
    await api.updateHolding(holdingId, data);
    setPortfolio((prev: Portfolio | null) => {
      if (!prev) return prev;
      
      // First pass: update the holding and calculate market values
      let totalMarketValue = 0;
      let totalCostBasis = 0;
      
      const holdingsWithUpdates = prev.holdings.map((h: Holding) => {
        const isUpdatedHolding = h.id === holdingId;
        const shares = isUpdatedHolding && data.shares !== undefined ? data.shares : h.shares;
        const avgCost = isUpdatedHolding && data.avg_cost !== undefined ? data.avg_cost : h.avg_cost;
        const isPinned = isUpdatedHolding && data.is_pinned !== undefined ? data.is_pinned : h.is_pinned;
        const currentPrice = h.current_price ?? 0;
        
        const marketValue = shares * currentPrice;
        const costBasis = shares && avgCost ? shares * avgCost : null;
        
        totalMarketValue += marketValue;
        if (costBasis !== null) totalCostBasis += costBasis;
        
        return {
          ...h,
          shares,
          avg_cost: avgCost,
          is_pinned: isPinned,
          market_value: Math.round(marketValue * 100) / 100,
          cost_basis: costBasis ? Math.round(costBasis * 100) / 100 : null,
        };
      });
      
      // Second pass: calculate allocations and gain/loss
      const updatedHoldings = holdingsWithUpdates.map((h) => {
        const allocationPct = totalMarketValue > 0 ? ((h.market_value ?? 0) / totalMarketValue) * 100 : 0;
        let gainLoss = null;
        let gainLossPct = null;
        
        if (h.market_value !== null && h.cost_basis !== null) {
          gainLoss = (h.market_value ?? 0) - (h.cost_basis ?? 0);
          if (h.avg_cost && h.avg_cost > 0 && h.current_price) {
            gainLossPct = ((h.current_price - h.avg_cost) / h.avg_cost) * 100;
          }
        }
        
        return {
          ...h,
          allocation_pct: Math.round(allocationPct * 100) / 100,
          gain_loss: gainLoss !== null ? Math.round(gainLoss * 100) / 100 : null,
          gain_loss_pct: gainLossPct !== null ? Math.round(gainLossPct * 100) / 100 : null,
        };
      });
      
      const totalGainLoss = totalCostBasis > 0 ? totalMarketValue - totalCostBasis : null;
      const totalGainLossPct = totalCostBasis > 0 && totalGainLoss !== null 
        ? (totalGainLoss / totalCostBasis) * 100 
        : null;
      
      return {
        ...prev,
        holdings: updatedHoldings,
        total_market_value: Math.round(totalMarketValue * 100) / 100,
        total_cost_basis: totalCostBasis > 0 ? Math.round(totalCostBasis * 100) / 100 : null,
        total_gain_loss: totalGainLoss !== null ? Math.round(totalGainLoss * 100) / 100 : null,
        total_gain_loss_pct: totalGainLossPct !== null ? Math.round(totalGainLossPct * 100) / 100 : null,
      };
    });
  };

  return {
    portfolio,
    loading,
    error,
    lastFetched,
    lastPricesFetched,
    refresh: () => fetchPortfolio(false),
    updatePortfolioValue,
    addHolding,
    removeHolding,
    updateHolding,
  };
}
