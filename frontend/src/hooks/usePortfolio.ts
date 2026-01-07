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
        
        // First pass: calculate market values for allocation percentages
        let totalMarketValue = 0;
        let totalCostBasis = 0;
        
        const holdingsWithPrices = prev.holdings.map((h: Holding) => {
          const price = priceData[h.ticker];
          const currentPrice = price?.current_price ?? h.current_price ?? 0;
          const marketValue = h.shares * currentPrice;
          const costBasis = h.shares && h.avg_cost ? h.shares * h.avg_cost : null;
          
          totalMarketValue += marketValue;
          if (costBasis !== null) totalCostBasis += costBasis;
          
          return {
            ...h,
            current_price: currentPrice,
            market_value: Math.round(marketValue * 100) / 100,
            cost_basis: costBasis ? Math.round(costBasis * 100) / 100 : null,
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
      
      // Update holding with quote data and recalculate portfolio totals
      setPortfolio((prev: Portfolio | null) => {
        if (!prev) return prev;
        
        // First, update the holding with new price data
        const updatedHoldings = prev.holdings.map((h: Holding) => {
          if (h.ticker !== holding.ticker) return h;
          
          const marketValue = h.shares * quote.current_price;
          const costBasis = h.shares && h.avg_cost ? h.shares * h.avg_cost : null;
          
          return {
            ...h,
            current_price: quote.current_price,
            market_value: Math.round(marketValue * 100) / 100,
            cost_basis: costBasis ? Math.round(costBasis * 100) / 100 : null,
            ytd_return: quote.ytd_return,
            sma_200: quote.sma_200 ?? undefined,
            price_vs_sma: quote.price_vs_sma ?? undefined,
          };
        });
        
        // Calculate portfolio totals
        let totalMarketValue = 0;
        let totalCostBasis = 0;
        
        updatedHoldings.forEach((h) => {
          totalMarketValue += h.market_value ?? 0;
          totalCostBasis += h.cost_basis ?? 0;
        });
        
        // Calculate allocations and gain/loss
        const finalHoldings = updatedHoldings.map((h) => {
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
          holdings: finalHoldings,
          total_market_value: Math.round(totalMarketValue * 100) / 100,
          total_cost_basis: totalCostBasis > 0 ? Math.round(totalCostBasis * 100) / 100 : null,
          total_gain_loss: totalGainLoss !== null ? Math.round(totalGainLoss * 100) / 100 : null,
          total_gain_loss_pct: totalGainLossPct !== null ? Math.round(totalGainLossPct * 100) / 100 : null,
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

  const addHolding = async (ticker: string, shares: number = 0, avgCost?: number) => {
    await api.addHolding({ ticker, shares, avg_cost: avgCost });
    
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

  const updateHolding = async (holdingId: number, data: { shares?: number; avg_cost?: number }) => {
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
        const currentPrice = h.current_price ?? 0;
        
        const marketValue = shares * currentPrice;
        const costBasis = shares && avgCost ? shares * avgCost : null;
        
        totalMarketValue += marketValue;
        if (costBasis !== null) totalCostBasis += costBasis;
        
        return {
          ...h,
          shares,
          avg_cost: avgCost,
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
