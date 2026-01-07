/**
 * Stock Analysis Hook
 * 
 * Manages background analysis of stocks with caching.
 * Analysis runs asynchronously to avoid blocking UI.
 * Results are cached for 15 minutes per stock.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { StockAnalysis } from '../types';
import { getStockHistory, getStock, getStockAnalysis as fetchStockAnalysisData } from '../services/api';
import { analyzeStock, type AdditionalAnalysisData } from '../services/stockScoring';

// ============================================================================
// Types
// ============================================================================

export interface CachedAnalysis {
  analysis: StockAnalysis;
  fetchedAt: Date;
  expiresAt: Date;
}

export interface AnalysisState {
  /** Cached analysis result (may be stale but still shown) */
  analysis: StockAnalysis | null;
  /** Whether analysis is currently being calculated */
  isLoading: boolean;
  /** Whether the cached data is stale (past expiry) */
  isStale: boolean;
  /** Error message if analysis failed */
  error: string | null;
  /** When the analysis was last calculated */
  lastUpdated: Date | null;
}

interface AnalysisRequest {
  ticker: string;
  currentPrice: number;
  high52w: number | null;
  low52w: number | null;
}

// ============================================================================
// Cache Configuration
// ============================================================================

const CACHE_DURATION_MS = 15 * 60 * 1000; // 15 minutes
const MAX_CONCURRENT_REQUESTS = 3;
const REQUEST_DELAY_MS = 200; // Delay between requests to avoid rate limiting

// ============================================================================
// Global Cache (shared across all component instances)
// ============================================================================

const analysisCache = new Map<string, CachedAnalysis>();
const pendingRequests = new Set<string>();
const requestQueue: AnalysisRequest[] = [];
let isProcessingQueue = false;

/**
 * Get cached analysis if it exists and is not expired.
 */
function getCachedAnalysis(ticker: string): CachedAnalysis | null {
  const cached = analysisCache.get(ticker);
  if (!cached) return null;
  return cached;
}

/**
 * Check if cached analysis is still valid (not expired).
 */
function isCacheValid(cached: CachedAnalysis): boolean {
  return new Date() < cached.expiresAt;
}

/**
 * Store analysis in cache.
 */
function setCachedAnalysis(ticker: string, analysis: StockAnalysis): void {
  const now = new Date();
  analysisCache.set(ticker, {
    analysis,
    fetchedAt: now,
    expiresAt: new Date(now.getTime() + CACHE_DURATION_MS)
  });
}

// ============================================================================
// Background Analysis Processor
// ============================================================================

/**
 * Process the request queue, fetching analysis for stocks in background.
 */
async function processQueue(
  onUpdate: (ticker: string, analysis: StockAnalysis) => void,
  onError: (ticker: string, error: string) => void,
  onStart: (ticker: string) => void
): Promise<void> {
  if (isProcessingQueue) return;
  isProcessingQueue = true;
  
  while (requestQueue.length > 0) {
    // Process up to MAX_CONCURRENT_REQUESTS at a time
    const batch = requestQueue.splice(0, MAX_CONCURRENT_REQUESTS);
    
    await Promise.all(
      batch.map(async (request) => {
        const { ticker, currentPrice, high52w, low52w } = request;
        
        // Skip if already being processed
        if (pendingRequests.has(ticker)) return;
        pendingRequests.add(ticker);
        onStart(ticker);
        
        try {
          // Fetch 1-year history and stock data in parallel
          // Stock data provides: current price (fallback), 52-week high/low
          // History provides: price data for indicators (need 200+ days for SMA200)
          const [historyResponse, stockData] = await Promise.all([
            getStockHistory(ticker, '1y'),
            getStock(ticker).catch(() => null) // Don't fail if stock data unavailable
          ]);
          
          const history = historyResponse.history;
          
          // Use provided values or fall back to fetched stock data
          const finalHigh52w = high52w ?? stockData?.high_52w ?? null;
          const finalLow52w = low52w ?? stockData?.low_52w ?? null;
          const finalPrice = currentPrice || stockData?.current_price || 0;
          
          if (!finalPrice) {
            throw new Error('Unable to determine current price');
          }
          
          // Try to fetch additional analysis data (fundamentals, options)
          let additionalData: AdditionalAnalysisData | undefined;
          try {
            const analysisData = await fetchStockAnalysisData(ticker);
            additionalData = {
              roic: analysisData.fundamentals?.roic,
              roe: analysisData.fundamentals?.roe,
              sector: analysisData.fundamentals?.sector,
              industry: analysisData.fundamentals?.industry,
              beta: analysisData.fundamentals?.beta,
              callPutRatioOI: analysisData.options?.call_put_ratio_oi,
              callPutRatioVolume: analysisData.options?.call_put_ratio_volume,
              optionsSentiment: analysisData.options?.options_sentiment,
              avgImpliedVolatility: analysisData.options?.avg_implied_volatility,
              ivPercentile: analysisData.options?.iv_percentile,
              hasOptions: analysisData.options?.has_options ?? false
            };
          } catch {
            // Continue without additional data if fetch fails
            console.debug(`[Analysis] Additional data not available for ${ticker}`);
          }
          
          // Run the analysis with complete data
          const analysis = analyzeStock(
            ticker,
            history,
            finalPrice,
            finalHigh52w,
            finalLow52w,
            undefined, // Use default weights
            additionalData
          );
          
          // Cache the result
          setCachedAnalysis(ticker, analysis);
          onUpdate(ticker, analysis);
          
        } catch (err) {
          const message = err instanceof Error ? err.message : 'Analysis failed';
          console.error(`[Analysis] Failed for ${ticker}:`, message);
          onError(ticker, message);
        } finally {
          pendingRequests.delete(ticker);
        }
      })
    );
    
    // Small delay between batches to be nice to the API
    if (requestQueue.length > 0) {
      await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY_MS));
    }
  }
  
  isProcessingQueue = false;
}

// ============================================================================
// Hook: useStockAnalysis
// ============================================================================

/**
 * Hook to get analysis for a single stock.
 * Manages caching and background fetching automatically.
 */
export function useStockAnalysis(
  ticker: string | null,
  currentPrice?: number,
  high52w?: number | null,
  low52w?: number | null
): AnalysisState {
  const [state, setState] = useState<AnalysisState>({
    analysis: null,
    isLoading: false,
    isStale: false,
    error: null,
    lastUpdated: null
  });
  
  // Ref to track mounted state
  const mountedRef = useRef(true);
  
  // Check cache on mount and when ticker changes
  useEffect(() => {
    if (!ticker || !currentPrice) {
      setState(prev => ({ ...prev, analysis: null, isLoading: false }));
      return;
    }
    
    const cached = getCachedAnalysis(ticker);
    
    if (cached) {
      const isValid = isCacheValid(cached);
      setState({
        analysis: cached.analysis,
        isLoading: !isValid && !pendingRequests.has(ticker),
        isStale: !isValid,
        error: null,
        lastUpdated: cached.fetchedAt
      });
      
      // If cache is stale, queue a refresh
      if (!isValid && !pendingRequests.has(ticker)) {
        queueAnalysis(ticker, currentPrice, high52w ?? null, low52w ?? null);
      }
    } else {
      // No cache - queue analysis
      setState(prev => ({ ...prev, isLoading: true, analysis: null }));
      queueAnalysis(ticker, currentPrice, high52w ?? null, low52w ?? null);
    }
    
    return () => {
      mountedRef.current = false;
    };
  }, [ticker, currentPrice, high52w, low52w]);
  
  // Queue analysis request
  const queueAnalysis = useCallback((
    t: string,
    price: number,
    high: number | null,
    low: number | null
  ) => {
    // Don't queue if already pending or in queue
    if (pendingRequests.has(t)) return;
    if (requestQueue.some(r => r.ticker === t)) return;
    
    requestQueue.push({
      ticker: t,
      currentPrice: price,
      high52w: high,
      low52w: low
    });
    
    // Start processing queue
    processQueue(
      // onUpdate
      (updatedTicker, analysis) => {
        if (mountedRef.current && updatedTicker === ticker) {
          setState({
            analysis,
            isLoading: false,
            isStale: false,
            error: null,
            lastUpdated: new Date()
          });
        }
      },
      // onError
      (errorTicker, error) => {
        if (mountedRef.current && errorTicker === ticker) {
          setState(prev => ({
            ...prev,
            isLoading: false,
            error
          }));
        }
      },
      // onStart
      (startTicker) => {
        if (mountedRef.current && startTicker === ticker) {
          setState(prev => ({ ...prev, isLoading: true }));
        }
      }
    );
  }, [ticker]);
  
  return state;
}

// ============================================================================
// Hook: useMultiStockAnalysis
// ============================================================================

interface MultiAnalysisState {
  analyses: Map<string, AnalysisState>;
  /** Whether any analysis is currently loading */
  isAnyLoading: boolean;
}

/**
 * Hook to get analysis for multiple stocks.
 * More efficient for dashboard views with many holdings.
 */
export function useMultiStockAnalysis(
  stocks: Array<{
    ticker: string;
    currentPrice?: number;
    high52w?: number | null;
    low52w?: number | null;
  }>
): MultiAnalysisState {
  const [analyses, setAnalyses] = useState<Map<string, AnalysisState>>(new Map());
  const mountedRef = useRef(true);
  
  useEffect(() => {
    mountedRef.current = true;
    
    const newAnalyses = new Map<string, AnalysisState>();
    const toQueue: AnalysisRequest[] = [];
    
    for (const stock of stocks) {
      if (!stock.ticker || !stock.currentPrice) continue;
      
      const cached = getCachedAnalysis(stock.ticker);
      
      if (cached) {
        const isValid = isCacheValid(cached);
        newAnalyses.set(stock.ticker, {
          analysis: cached.analysis,
          isLoading: !isValid && !pendingRequests.has(stock.ticker),
          isStale: !isValid,
          error: null,
          lastUpdated: cached.fetchedAt
        });
        
        // Queue refresh for stale items
        if (!isValid && !pendingRequests.has(stock.ticker)) {
          toQueue.push({
            ticker: stock.ticker,
            currentPrice: stock.currentPrice,
            high52w: stock.high52w ?? null,
            low52w: stock.low52w ?? null
          });
        }
      } else {
        newAnalyses.set(stock.ticker, {
          analysis: null,
          isLoading: true,
          isStale: false,
          error: null,
          lastUpdated: null
        });
        
        toQueue.push({
          ticker: stock.ticker,
          currentPrice: stock.currentPrice,
          high52w: stock.high52w ?? null,
          low52w: stock.low52w ?? null
        });
      }
    }
    
    setAnalyses(newAnalyses);
    
    // Queue all needed analyses
    for (const request of toQueue) {
      if (!pendingRequests.has(request.ticker) && 
          !requestQueue.some(r => r.ticker === request.ticker)) {
        requestQueue.push(request);
      }
    }
    
    // Start processing
    if (toQueue.length > 0) {
      processQueue(
        // onUpdate
        (ticker, analysis) => {
          if (mountedRef.current) {
            setAnalyses(prev => {
              const next = new Map(prev);
              next.set(ticker, {
                analysis,
                isLoading: false,
                isStale: false,
                error: null,
                lastUpdated: new Date()
              });
              return next;
            });
          }
        },
        // onError
        (ticker, error) => {
          if (mountedRef.current) {
            setAnalyses(prev => {
              const next = new Map(prev);
              const existing = prev.get(ticker);
              next.set(ticker, {
                ...existing,
                analysis: existing?.analysis ?? null,
                isLoading: false,
                error,
                isStale: existing?.isStale ?? false,
                lastUpdated: existing?.lastUpdated ?? null
              });
              return next;
            });
          }
        },
        // onStart
        (ticker) => {
          if (mountedRef.current) {
            setAnalyses(prev => {
              const next = new Map(prev);
              const existing = prev.get(ticker);
              if (existing) {
                next.set(ticker, { ...existing, isLoading: true });
              }
              return next;
            });
          }
        }
      );
    }
    
    return () => {
      mountedRef.current = false;
    };
  }, [stocks.map(s => `${s.ticker}:${s.currentPrice}`).join(',')]);
  
  const isAnyLoading = Array.from(analyses.values()).some(a => a.isLoading);
  
  return { analyses, isAnyLoading };
}

// ============================================================================
// Utility: Force Refresh
// ============================================================================

/**
 * Force refresh analysis for a specific ticker.
 * Clears cache and queues new analysis.
 */
export function refreshAnalysis(
  ticker: string,
  currentPrice: number,
  high52w: number | null,
  low52w: number | null
): void {
  // Clear cache
  analysisCache.delete(ticker);
  
  // Queue new analysis (will be processed by existing hooks)
  if (!pendingRequests.has(ticker) && 
      !requestQueue.some(r => r.ticker === ticker)) {
    requestQueue.push({ ticker, currentPrice, high52w, low52w });
  }
}

/**
 * Clear all cached analyses.
 */
export function clearAnalysisCache(): void {
  analysisCache.clear();
}

/**
 * Get cache statistics (for debugging).
 */
export function getAnalysisCacheStats(): {
  size: number;
  pending: number;
  queued: number;
} {
  return {
    size: analysisCache.size,
    pending: pendingRequests.size,
    queued: requestQueue.length
  };
}

/**
 * Get the cached analysis score for a ticker (for sorting).
 * Returns null if no cached analysis exists.
 */
export function getCachedAnalysisScore(ticker: string): number | null {
  const cached = analysisCache.get(ticker);
  if (!cached) return null;
  return cached.analysis.bestAction.totalScore;
}

