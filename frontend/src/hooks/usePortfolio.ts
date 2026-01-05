import { useState, useEffect, useCallback, useRef } from 'react';
import type { Portfolio, Holding, HistoryPoint } from '../types';
import * as api from '../services/api';

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

  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const isInitialLoad = useRef(true);
  const currentRefreshIndex = useRef(0);
  const isMounted = useRef(true);
  
  // Keep refs for the rolling refresh loop
  const portfolioRef = useRef<Portfolio | null>(null);
  const chartPeriodRef = useRef(chartPeriod);
  const onHistoryUpdateRef = useRef(onHistoryUpdate);

  useEffect(() => {
    portfolioRef.current = portfolio;
  }, [portfolio]);

  useEffect(() => {
    chartPeriodRef.current = chartPeriod;
    // Reset index when period changes so we start fresh
    currentRefreshIndex.current = 0;
  }, [chartPeriod]);

  useEffect(() => {
    onHistoryUpdateRef.current = onHistoryUpdate;
  }, [onHistoryUpdate]);

  const fetchPortfolio = useCallback(async (showLoading = true, useLiteMode = false) => {
    if (showLoading && isInitialLoad.current) {
      setLoading(true);
    } else if (!useLiteMode) {
      setRefreshing(true);
    }
    
    try {
      const data = await api.getPortfolio(useLiteMode);
      setPortfolio(data);
      setError(null);
      setLastFetched(new Date());
      isInitialLoad.current = false;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Refresh a single stock - fetches BOTH quote and history sequentially
  const refreshSingleStock = useCallback(async (holding: Holding) => {
    if (!isMounted.current) return;
    
    try {
      // First fetch quote
      const quote = await api.getStockQuote(holding.ticker);
      
      if (!isMounted.current) return;
      
      // Update holding with quote data
      setPortfolio(prev => {
        if (!prev) return prev;
        
        const allocatedValue = prev.total_value * (holding.allocation_pct / 100);
        const currentValue = allocatedValue * (1 + quote.ytd_return / 100);
        
        const updatedHoldings = prev.holdings.map(h => 
          h.ticker === holding.ticker 
            ? {
                ...h,
                current_price: quote.current_price,
                current_value: Math.round(currentValue * 100) / 100,
                ytd_return: quote.ytd_return,
                sma_200: quote.sma_200 ?? undefined,
                price_vs_sma: quote.price_vs_sma ?? undefined,
              }
            : h
        );
        
        const newTotalValue = updatedHoldings.reduce((sum, h) => sum + (h.current_value ?? 0), 0);
        const totalAllocated = updatedHoldings.reduce((sum, h) => sum + h.allocation_pct, 0);
        const baseInvested = prev.total_value * (totalAllocated / 100);
        const totalGainLoss = baseInvested > 0 ? newTotalValue - baseInvested : 0;
        const totalGainLossPct = baseInvested > 0 ? (totalGainLoss / baseInvested) * 100 : 0;
        
        return {
          ...prev,
          holdings: updatedHoldings,
          current_total_value: Math.round(newTotalValue * 100) / 100,
          total_gain_loss: Math.round(totalGainLoss * 100) / 100,
          total_gain_loss_pct: Math.round(totalGainLossPct * 100) / 100,
        };
      });

      // Then fetch history (sequential, not parallel)
      const historyData = await api.getStockHistory(holding.ticker, chartPeriodRef.current);
      
      if (!isMounted.current) return;
      
      // Notify parent of history update
      if (onHistoryUpdateRef.current) {
        onHistoryUpdateRef.current(holding.ticker, {
          history: historyData.history || [],
          referenceClose: historyData.reference_close,
          isComplete: historyData.is_complete ?? false,
          expectedStart: historyData.expected_start,
          actualStart: historyData.actual_start
        });
      }
      
      setLastFetched(new Date());
    } catch (err) {
      console.error(`Failed to refresh ${holding.ticker}:`, err);
    }
  }, []);

  // Rolling refresh - one stock at a time, fetching quote + history
  // Moves to next stock immediately after completing current one
  const runRollingRefresh = useCallback(async () => {
    // Small delay before starting to let initial render settle
    await new Promise(resolve => setTimeout(resolve, 300));
    
    while (isMounted.current) {
      const currentPortfolio = portfolioRef.current;
      
      if (!currentPortfolio?.holdings.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        continue;
      }
      
      // Capture the period before fetching
      const periodBeforeFetch = chartPeriodRef.current;
      
      const index = currentRefreshIndex.current % currentPortfolio.holdings.length;
      const holding = currentPortfolio.holdings[index];
      
      if (holding) {
        await refreshSingleStock(holding);
      }
      
      // Only move to next stock if period didn't change during fetch
      // If period changed, index was already reset to 0 - don't increment it
      if (chartPeriodRef.current === periodBeforeFetch) {
        currentRefreshIndex.current = (index + 1) % currentPortfolio.holdings.length;
      }
    }
  }, [refreshSingleStock]);

  // Initial load and start rolling refresh
  useEffect(() => {
    isMounted.current = true;
    fetchPortfolio(true, true); // Lite mode for instant load
    runRollingRefresh();
    
    return () => {
      isMounted.current = false;
    };
  }, [fetchPortfolio, runRollingRefresh]);

  const updatePortfolioValue = async (total_value: number) => {
    try {
      const updated = await api.updatePortfolio({ total_value });
      setPortfolio(prev => prev ? { ...prev, total_value: updated.total_value } : prev);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update portfolio');
    }
  };

  const addHolding = async (ticker: string, allocation_pct: number, investment_date?: string, investment_price?: number) => {
    await api.addHolding({ ticker, allocation_pct, investment_date, investment_price });
    setPortfolio(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        holdings: [
          ...prev.holdings,
          {
            id: Date.now(),
            portfolio_id: prev.id,
            ticker: ticker.toUpperCase(),
            allocation_pct,
            added_at: new Date().toISOString(),
            investment_date,
            investment_price,
            current_price: undefined,
            current_value: undefined,
            ytd_return: undefined,
          }
        ]
      };
    });
    fetchPortfolio(false);
  };

  const removeHolding = async (holdingId: number) => {
    try {
      await api.deleteHolding(holdingId);
      setPortfolio(prev => {
        if (!prev) return prev;
        return {
          ...prev,
          holdings: prev.holdings.filter(h => h.id !== holdingId)
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove holding');
    }
  };

  const updateHolding = async (holdingId: number, data: { allocation_pct?: number; investment_date?: string; investment_price?: number }) => {
    await api.updateHolding(holdingId, data);
    setPortfolio(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        holdings: prev.holdings.map(h => 
          h.id === holdingId ? { ...h, ...data } : h
        )
      };
    });
  };

  return {
    portfolio,
    loading,
    refreshing,
    error,
    lastFetched,
    refresh: () => fetchPortfolio(false),
    updatePortfolioValue,
    addHolding,
    removeHolding,
    updateHolding,
  };
}
