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

// Intraday periods
// - 1d: Uses live chart_point from batch prices (built in real-time, 5m intervals)
// - 3d, 1w, 1mo: Uses incremental history fetch every 15 seconds
const INTRADAY_PERIODS = ['1d', '3d', '1w', '1mo'];

// Map period to interval for incremental fetching and live chart_point tracking
// Must match backend configuration in stock_fetcher.py
const PERIOD_TO_INTERVAL: Record<string, string> = {
  '1d': '5m',    // Every 5 minutes
  '3d': '15m',   // Every 15 minutes
  '1w': '30m',   // Every 30 minutes
  '1mo': '60m',  // Every hour
};

/**
 * Convert chart date format "YYYY-MM-DD HH:MM" to ISO format "YYYY-MM-DDTHH:MM:00"
 * for use with the incremental history endpoint.
 */
function chartDateToISO(dateStr: string): string {
  if (!dateStr) return '';
  // Handle "2024-01-15 10:30" -> "2024-01-15T10:30:00"
  if (dateStr.includes(' ') && !dateStr.includes('T')) {
    return dateStr.replace(' ', 'T') + ':00';
  }
  // Already ISO format or different format
  return dateStr;
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
  const [lastPricesFetched, setLastPricesFetched] = useState<Date | null>(null);
  const isInitialLoad = useRef(true);
  const isMounted = useRef(true);
  
  // Track latest timestamps for incremental fetching
  // Key: `${ticker}:${interval}` -> ISO timestamp string
  const latestTimestampsRef = useRef<Record<string, string>>({});
  
  // Track chart data for each ticker (for appending incremental updates)
  const chartDataRef = useRef<Record<string, HoldingChartData>>({});
  
  // Keep refs for the refresh loop
  const portfolioRef = useRef<Portfolio | null>(null);
  const chartPeriodRef = useRef(chartPeriod);
  const onHistoryUpdateRef = useRef(onHistoryUpdate);
  
  // Prevent duplicate refresh loops (React StrictMode can double-mount)
  const refreshLoopRunningRef = useRef(false);

  useEffect(() => {
    portfolioRef.current = portfolio;
  }, [portfolio]);

  // Track if we need to refetch history due to period change
  const needsHistoryRefetchRef = useRef(false);
  
  useEffect(() => {
    const prevPeriod = chartPeriodRef.current;
    chartPeriodRef.current = chartPeriod;
    
    // If period changed, clear cached chart data and trigger refetch
    if (prevPeriod !== chartPeriod) {
      chartDataRef.current = {};
      latestTimestampsRef.current = {};
      needsHistoryRefetchRef.current = true;
      
      // Immediately trigger history refetch for all holdings
      const currentPortfolio = portfolioRef.current;
      if (currentPortfolio?.holdings.length && isMounted.current) {
        console.log(`Period changed from ${prevPeriod} to ${chartPeriod}, refetching history...`);
        
        // Fetch history for all holdings with new period
        (async () => {
          for (const holding of currentPortfolio.holdings) {
            if (!isMounted.current) break;
            try {
              // Only fetch history - quote data comes from batch price refresh
              const historyData = await api.getStockHistory(holding.ticker, chartPeriod);
              
              if (!isMounted.current) break;
              
              // Store chart data
              const chartData: HoldingChartData = {
                history: historyData.history || [],
                referenceClose: historyData.reference_close,
                isComplete: historyData.is_complete ?? false,
                expectedStart: historyData.expected_start,
                actualStart: historyData.actual_start
              };
              
              chartDataRef.current[holding.ticker] = chartData;
              
              // Track latest timestamp for incremental updates
              if (historyData.history?.length && INTRADAY_PERIODS.includes(chartPeriod)) {
                const interval = PERIOD_TO_INTERVAL[chartPeriod];
                const lastPoint = historyData.history[historyData.history.length - 1];
                if (lastPoint?.date) {
                  latestTimestampsRef.current[`${holding.ticker}:${interval}`] = chartDateToISO(lastPoint.date);
                }
              }
              
              // Notify parent of history update
              if (onHistoryUpdateRef.current) {
                onHistoryUpdateRef.current(holding.ticker, chartData);
              }
            } catch (err) {
              console.error(`Failed to fetch history for ${holding.ticker} after period change:`, err);
            }
          }
          needsHistoryRefetchRef.current = false;
        })();
      }
    }
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

  // Batch refresh all prices at once (much faster than individual calls)
  // Also appends chart points to 1d charts for real-time updates
  const refreshAllPrices = useCallback(async () => {
    const currentPortfolio = portfolioRef.current;
    if (!currentPortfolio?.holdings.length || !isMounted.current) return;
    
    const tickers = currentPortfolio.holdings.map((h: Holding) => h.ticker);
    const period = chartPeriodRef.current;
    
    try {
      const priceData = await api.getBatchPrices(tickers);
      
      if (!isMounted.current) return;
      
      setPortfolio((prev: Portfolio | null) => {
        if (!prev) return prev;
        
        const updatedHoldings = prev.holdings.map((h: Holding) => {
          const price = priceData[h.ticker];
          if (!price || price.current_price === 0) return h;
          
          // We need ytd_return from the quote endpoint for value calculation
          // For now, keep existing ytd_return if we have it
          const ytdReturn = h.ytd_return ?? 0;
          const allocatedValue = prev.total_value * (h.allocation_pct / 100);
          const currentValue = allocatedValue * (1 + ytdReturn / 100);
          
          return {
            ...h,
            current_price: price.current_price,
            current_value: Math.round(currentValue * 100) / 100,
          };
        });
        
        const newTotalValue = updatedHoldings.reduce((sum: number, h: Holding) => sum + (h.current_value ?? 0), 0);
        const totalAllocated = updatedHoldings.reduce((sum: number, h: Holding) => sum + h.allocation_pct, 0);
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
      
      // For 1d period, append chart points from the batch price response
      // This builds the chart in real-time from live price updates
      if (period === '1d') {
        for (const ticker of tickers) {
          const price = priceData[ticker];
          if (!price?.chart_point) continue;
          
          const existingData = chartDataRef.current[ticker];
          if (!existingData?.history) continue;
          
          const chartPoint = price.chart_point;
          const existingHistory = existingData.history;
          
          // Check if this is a new point or update to existing
          // Points are keyed by date string "YYYY-MM-DD HH:MM"
          const lastPoint = existingHistory[existingHistory.length - 1];
          
          if (lastPoint?.date === chartPoint.date) {
            // Same minute - update the last point (OHLC update)
            const updatedHistory = [...existingHistory.slice(0, -1), chartPoint];
            const updatedChartData: HoldingChartData = {
              ...existingData,
              history: updatedHistory
            };
            chartDataRef.current[ticker] = updatedChartData;
            
            // Notify parent of history update
            if (onHistoryUpdateRef.current) {
              onHistoryUpdateRef.current(ticker, updatedChartData);
            }
          } else if (!lastPoint || chartPoint.date > lastPoint.date) {
            // New minute - append the point
            const updatedHistory = [...existingHistory, chartPoint];
            const updatedChartData: HoldingChartData = {
              ...existingData,
              history: updatedHistory
            };
            chartDataRef.current[ticker] = updatedChartData;
            
            // Update latest timestamp for this ticker
            latestTimestampsRef.current[`${ticker}:1m`] = chartDateToISO(chartPoint.date);
            
            // Notify parent of history update
            if (onHistoryUpdateRef.current) {
              onHistoryUpdateRef.current(ticker, updatedChartData);
            }
          }
        }
      }
      
      const now = new Date();
      setLastFetched(now);
      setLastPricesFetched(now);
    } catch (err) {
      console.error('Failed to batch refresh prices:', err);
    }
  }, []);

  // Fetch full history for a stock (used on initial load or period change)
  const fetchFullHistory = useCallback(async (holding: Holding) => {
    if (!isMounted.current) return;
    
    const period = chartPeriodRef.current;
    
    try {
      // Also get the quote for ytd_return and SMA
      const [quote, historyData] = await Promise.all([
        api.getStockQuote(holding.ticker),
        api.getStockHistory(holding.ticker, period)
      ]);
      
      if (!isMounted.current) return;
      
      // Update holding with quote data
      setPortfolio((prev: Portfolio | null) => {
        if (!prev) return prev;
        
        const allocatedValue = prev.total_value * (holding.allocation_pct / 100);
        const currentValue = allocatedValue * (1 + quote.ytd_return / 100);
        
        const updatedHoldings = prev.holdings.map((h: Holding) => 
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
        
        const newTotalValue = updatedHoldings.reduce((sum: number, h: Holding) => sum + (h.current_value ?? 0), 0);
        const totalAllocated = updatedHoldings.reduce((sum: number, h: Holding) => sum + h.allocation_pct, 0);
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
      
      // Store chart data and latest timestamp
      const chartData: HoldingChartData = {
        history: historyData.history || [],
        referenceClose: historyData.reference_close,
        isComplete: historyData.is_complete ?? false,
        expectedStart: historyData.expected_start,
        actualStart: historyData.actual_start
      };
      
      chartDataRef.current[holding.ticker] = chartData;
      
      // Track latest timestamp for incremental updates (convert to ISO format)
      if (historyData.history?.length && INTRADAY_PERIODS.includes(period)) {
        const interval = PERIOD_TO_INTERVAL[period];
        const lastPoint = historyData.history[historyData.history.length - 1];
        if (lastPoint?.date) {
          // Convert "2024-01-15 10:30" to "2024-01-15T10:30:00" for backend
          latestTimestampsRef.current[`${holding.ticker}:${interval}`] = chartDateToISO(lastPoint.date);
        }
      }
      
      // Notify parent of history update
      if (onHistoryUpdateRef.current) {
        onHistoryUpdateRef.current(holding.ticker, chartData);
      }
      
      setLastFetched(new Date());
    } catch (err) {
      console.error(`Failed to fetch full history for ${holding.ticker}:`, err);
    }
  }, []);

  // Main refresh loop - batch prices only
  // History is fetched once on initial load and when period changes
  // For 1d period, chart_point in batch response builds chart in real-time
  const runRefreshLoop = useCallback(async () => {
    // Prevent duplicate loops (React StrictMode double-mounts)
    if (refreshLoopRunningRef.current) {
      console.log('Refresh loop already running, skipping duplicate');
      return;
    }
    refreshLoopRunningRef.current = true;
    
    try {
      // Wait for initial portfolio data to be available
      // Poll every 100ms for up to 10 seconds
      let waitCount = 0;
      while (!portfolioRef.current && isMounted.current && waitCount < 100) {
        await new Promise(resolve => setTimeout(resolve, 100));
        waitCount++;
      }
      
      if (!isMounted.current) return;
      
      // First, do initial full history fetch for all holdings
      const currentPortfolio = portfolioRef.current;
      if (currentPortfolio?.holdings.length) {
        console.log(`Starting full history fetch for ${currentPortfolio.holdings.length} holdings...`);
        for (const holding of currentPortfolio.holdings) {
          if (!isMounted.current) return;
          await fetchFullHistory(holding);
        }
        console.log('Full history fetch complete, starting price refresh loop...');
      }
      
      // Price refresh loop - minimum interval between fetches
      const MIN_PRICE_INTERVAL = 3000;  // Minimum 3 seconds between price fetches
      
      while (isMounted.current) {
        const currentPortfolio = portfolioRef.current;
        
        // Record start time to ensure minimum interval between fetches
        const cycleStartTime = Date.now();
        
        // Batch refresh all prices (fast, single API call)
        // For 1d period, this also appends chart_point to build chart in real-time
        await refreshAllPrices();
        
        if (!isMounted.current) break;
        
        // Check for holdings missing chart data (e.g., newly added stocks)
        if (currentPortfolio?.holdings.length) {
          const holdingsMissingData = currentPortfolio.holdings.filter(
            (h: Holding) => !chartDataRef.current[h.ticker]?.history?.length
          );
          
          if (holdingsMissingData.length > 0) {
            console.log(`Fetching history for ${holdingsMissingData.length} holdings missing chart data...`);
            for (const holding of holdingsMissingData) {
              if (!isMounted.current) break;
              await fetchFullHistory(holding);
            }
          }
        }
        
        if (!isMounted.current) break;
        
        // Wait only the remaining time to ensure minimum interval between fetches
        // This prevents request buildup if API calls are slow
        const elapsed = Date.now() - cycleStartTime;
        const remainingWait = MIN_PRICE_INTERVAL - elapsed;
        if (remainingWait > 0) {
          await new Promise(resolve => setTimeout(resolve, remainingWait));
        }
      }
    } finally {
      refreshLoopRunningRef.current = false;
    }
  }, [refreshAllPrices, fetchFullHistory]);

  // Initial load and start refresh loop
  useEffect(() => {
    isMounted.current = true;
    fetchPortfolio(true, true); // Lite mode for instant load
    runRefreshLoop();
    
    return () => {
      isMounted.current = false;
    };
  }, [fetchPortfolio, runRefreshLoop]);

  const updatePortfolioValue = async (total_value: number) => {
    try {
      const updated = await api.updatePortfolio({ total_value });
      setPortfolio((prev: Portfolio | null) => prev ? { ...prev, total_value: updated.total_value } : prev);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update portfolio');
    }
  };

  const addHolding = async (ticker: string, allocation_pct: number, investment_date?: string, investment_price?: number) => {
    await api.addHolding({ ticker, allocation_pct, investment_date, investment_price });
    
    const upperTicker = ticker.toUpperCase();
    
    // Clear cached data for fresh fetch
    delete chartDataRef.current[upperTicker];
    const interval = PERIOD_TO_INTERVAL[chartPeriodRef.current];
    if (interval) {
      delete latestTimestampsRef.current[`${upperTicker}:${interval}`];
    }
    
    // Create a temporary holding object for fetching history
    const tempHolding: Holding = {
      id: Date.now(),
      portfolio_id: 0,
      ticker: upperTicker,
      allocation_pct,
      added_at: new Date().toISOString(),
      investment_date,
      investment_price,
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
    
    // Fetch full history for the new holding (critical for chart to display)
    // This runs in background - don't await to keep UI responsive
    fetchFullHistory(tempHolding).catch((err: Error) => {
      console.error(`Failed to fetch history for new holding ${upperTicker}:`, err);
    });
    
    // Also refresh portfolio data from server
    fetchPortfolio(false);
  };

  const removeHolding = async (holdingId: number) => {
    try {
      const holding = portfolioRef.current?.holdings.find((h: Holding) => h.id === holdingId);
      if (holding) {
        // Clean up cached data
        delete chartDataRef.current[holding.ticker];
        const interval = PERIOD_TO_INTERVAL[chartPeriodRef.current];
        if (interval) {
          delete latestTimestampsRef.current[`${holding.ticker}:${interval}`];
        }
      }
      
      await api.deleteHolding(holdingId);
      setPortfolio((prev: Portfolio | null) => {
        if (!prev) return prev;
        return {
          ...prev,
          holdings: prev.holdings.filter((h: Holding) => h.id !== holdingId)
        };
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove holding');
    }
  };

  const updateHolding = async (holdingId: number, data: { allocation_pct?: number; investment_date?: string; investment_price?: number }) => {
    await api.updateHolding(holdingId, data);
    setPortfolio((prev: Portfolio | null) => {
      if (!prev) return prev;
      
      // Update the holding with new data and recalculate current_value if allocation changed
      const updatedHoldings = prev.holdings.map((h: Holding) => {
        if (h.id !== holdingId) return h;
        
        const updatedHolding = { ...h, ...data };
        
        // Recalculate current_value if allocation changed and we have ytd_return data
        if (data.allocation_pct !== undefined && h.ytd_return !== undefined) {
          const allocatedValue = prev.total_value * (data.allocation_pct / 100);
          const currentValue = allocatedValue * (1 + h.ytd_return / 100);
          updatedHolding.current_value = Math.round(currentValue * 100) / 100;
        }
        
        return updatedHolding;
      });
      
      // Recalculate portfolio totals
      const newTotalValue = updatedHoldings.reduce((sum: number, h: Holding) => sum + (h.current_value ?? 0), 0);
      const totalAllocated = updatedHoldings.reduce((sum: number, h: Holding) => sum + h.allocation_pct, 0);
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
  };

  return {
    portfolio,
    loading,
    refreshing,
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
