import { useState, useEffect, useCallback, useRef } from 'react';
import type { Portfolio } from '../types';
import * as api from '../services/api';

export function usePortfolio(refreshInterval = 10000) {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const isInitialLoad = useRef(true);

  const fetchPortfolio = useCallback(async (showLoading = true) => {
    if (showLoading && isInitialLoad.current) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    
    try {
      const data = await api.getPortfolio();
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

  useEffect(() => {
    fetchPortfolio(true);
    const interval = setInterval(() => fetchPortfolio(false), refreshInterval);
    return () => clearInterval(interval);
  }, [fetchPortfolio, refreshInterval]);

  const updatePortfolioValue = async (total_value: number) => {
    try {
      const updated = await api.updatePortfolio({ total_value });
      setPortfolio(prev => prev ? { ...prev, total_value: updated.total_value } : prev);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update portfolio');
    }
  };

  const addHolding = async (ticker: string, allocation_pct: number) => {
    await api.addHolding(ticker, allocation_pct);
    await fetchPortfolio(false);
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

  const updateHolding = async (holdingId: number, allocation_pct: number) => {
    await api.updateHolding(holdingId, allocation_pct);
    setPortfolio(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        holdings: prev.holdings.map(h => 
          h.id === holdingId ? { ...h, allocation_pct } : h
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
