/**
 * Stock Analysis Hook
 * 
 * Manages background analysis of stocks with caching.
 * Analysis runs asynchronously to avoid blocking UI.
 * Results are cached for 15 minutes per stock per horizon.
 * Implements a Pub/Sub pattern to handle multiple components tracking different stocks.
 */

import { useState, useEffect, useRef } from 'react';
import type { StockAnalysis } from '../types';
import { getStockHistory, getStock, getStockAnalysis as fetchStockAnalysisData, getBatchHistory, getBatchAnalysis } from '../services/api';
import { getBatchWeights } from '../services/calibrationApi';
import { analyzeStock, type AdditionalAnalysisData } from '../services/stockScoring';

// ============================================================================
// Types
// ============================================================================

export interface CachedAnalysis {
  analysis: StockAnalysis | null;
  error?: string | null;
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
  horizon: number;
  currentPrice: number;
  high52w: number | null;
  low52w: number | null;
}

type AnalysisListener = (
  ticker: string, 
  horizon: number, 
  analysis: StockAnalysis | null, 
  error: string | null
) => void;

// ============================================================================
// Cache Configuration
// ============================================================================

const CACHE_DURATION_MS = 15 * 60 * 1000; // 15 minutes
const MAX_CONCURRENT_REQUESTS = 50; // Process all holdings in one batch
const REQUEST_DELAY_MS = 100; // Small delay between batches if queue is very large
const QUEUE_DEBOUNCE_MS = 50; // Wait for queue to fill before processing

// ============================================================================
// Global State (shared across all component instances)
// ============================================================================

const analysisCache = new Map<string, CachedAnalysis>();
const pendingRequests = new Set<string>(); // Key: `${ticker}:${horizon}`
const requestQueue: AnalysisRequest[] = [];
let isProcessingQueue = false;
let queueDebounceTimer: ReturnType<typeof setTimeout> | null = null;

// Listeners for updates
const listeners = new Set<AnalysisListener>();

function subscribe(listener: AnalysisListener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notifyListeners(ticker: string, horizon: number, analysis: StockAnalysis | null, error: string | null) {
  listeners.forEach(listener => listener(ticker, horizon, analysis, error));
}

function getCacheKey(ticker: string, horizon: number): string {
  return `${ticker}:${horizon}`;
}

/**
 * Get cached analysis if it exists and is not expired.
 */
function getCachedAnalysis(ticker: string, horizon: number): CachedAnalysis | null {
  const cached = analysisCache.get(getCacheKey(ticker, horizon));
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
 * Store analysis (or error) in cache.
 */
function setCachedAnalysis(ticker: string, horizon: number, analysis: StockAnalysis | null, error?: string): void {
  const now = new Date();
  analysisCache.set(getCacheKey(ticker, horizon), {
    analysis,
    error,
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
async function processQueue(): Promise<void> {
  if (isProcessingQueue) return;
  isProcessingQueue = true;
  
  while (requestQueue.length > 0) {
    // Group requests by horizon
    const firstRequest = requestQueue[0];
    const targetHorizon = firstRequest.horizon;
    
    const horizonBatch: AnalysisRequest[] = [];
    const remainingQueue: AnalysisRequest[] = [];
    
    for (const req of requestQueue) {
        if (req.horizon === targetHorizon && horizonBatch.length < MAX_CONCURRENT_REQUESTS) {
            horizonBatch.push(req);
        } else {
            remainingQueue.push(req);
        }
    }
    
    // Replace main queue with remaining items
    requestQueue.length = 0;
    requestQueue.push(...remainingQueue);
    
    // Filter out already-pending requests and mark new ones as pending
    const validRequests = horizonBatch.filter(request => {
      const key = getCacheKey(request.ticker, request.horizon);
      if (pendingRequests.has(key)) return false;
      pendingRequests.add(key);
      // No explicit "onStart" notification needed; components set loading when they queue
      return true;
    });
    
    if (validRequests.length === 0) continue;
    
    const tickers = validRequests.map(r => r.ticker);
    
    try {
      // Batch fetch data - fetch weights for all 3 strategy classes
      console.debug(`[Analysis] Batch fetching for ${tickers.length} tickers (Horizon: ${targetHorizon})`);
      const [batchHistories, batchAnalysisData, directionalWeights, premiumSellWeights, premiumBuyWeights] = await Promise.all([
        getBatchHistory(tickers, '1y'),
        getBatchAnalysis(tickers),
        getBatchWeights(tickers, targetHorizon, 'directional'),
        getBatchWeights(tickers, targetHorizon, 'premium_sell'),
        getBatchWeights(tickers, targetHorizon, 'premium_buy')
      ]);
      
      // Process each ticker
      await Promise.all(
        validRequests.map(async (request) => {
          const { ticker, horizon, currentPrice, high52w, low52w } = request;
          const requestKey = getCacheKey(ticker, horizon);
          
          try {
            const tickerUpper = ticker.toUpperCase();

            // Build strategy-specific weights map
            const strategyWeightsMap: Record<string, Record<string, number>> = {};
            let calibrationMetadata = undefined;
            let isLowConfidence = false;
            let bestSqn: number | null = null;

            // Process each strategy class's weights
            const strategyResponses = [
              { key: 'directional', response: directionalWeights },
              { key: 'premium_sell', response: premiumSellWeights },
              { key: 'premium_buy', response: premiumBuyWeights }
            ];

            for (const { key, response } of strategyResponses) {
              const calibrationData = response?.tickers?.[tickerUpper] || response?.tickers?.[ticker];
              if (calibrationData && !calibrationData.is_default) {
                strategyWeightsMap[key] = calibrationData.weights as unknown as Record<string, number>;

                // Track the best SQN across strategies for confidence assessment
                if (calibrationData.sqn !== null) {
                  if (bestSqn === null || calibrationData.sqn > bestSqn) {
                    bestSqn = calibrationData.sqn;
                  }
                }

                // Use directional calibration for metadata (primary strategy)
                if (key === 'directional') {
                  calibrationMetadata = {
                    lastCalibrated: calibrationData.updated_at || new Date().toISOString(),
                    sqn: calibrationData.sqn,
                    period: response.horizon
                  };
                }
              }
            }

            // Determine confidence based on best SQN
            if (bestSqn !== null && bestSqn < 2.0) {
              isLowConfidence = true;
            }

            // For backward compatibility, use directional weights as the primary weights
            const weights = strategyWeightsMap['directional'];

            const historyData = batchHistories[ticker];
            let history = historyData?.history || [];
            
            if (history.length === 0) {
              const historyResponse = await getStockHistory(ticker, '1y');
              history = historyResponse.history;
            }
            
            let stockData = null;
            if (high52w === null || low52w === null) {
              stockData = await getStock(ticker).catch(() => null);
            }
            
            const finalHigh52w = high52w ?? stockData?.high_52w ?? null;
            const finalLow52w = low52w ?? stockData?.low_52w ?? null;
            const finalPrice = currentPrice || stockData?.current_price || 0;
            
            if (!finalPrice) throw new Error('Unable to determine current price');
            
            let additionalData: AdditionalAnalysisData | undefined;
            const analysisData = batchAnalysisData[ticker];
            if (analysisData) {
              // Calculate days to earnings from next_earnings_date
              let daysToEarnings: number | null = null;
              const nextEarningsDate = analysisData.fundamentals?.next_earnings_date;
              if (nextEarningsDate) {
                const earningsDate = new Date(nextEarningsDate);
                const today = new Date();
                today.setHours(0, 0, 0, 0);
                earningsDate.setHours(0, 0, 0, 0);
                daysToEarnings = Math.ceil((earningsDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
              }

              // Use call/put ratio from open interest (more reliable than volume)
              const callPutRatio = analysisData.options?.call_put_ratio_oi ?? analysisData.options?.call_put_ratio_volume ?? null;

              additionalData = {
                roic: analysisData.fundamentals?.roic,
                roe: analysisData.fundamentals?.roe,
                sector: analysisData.fundamentals?.sector,
                industry: analysisData.fundamentals?.industry,
                beta: analysisData.fundamentals?.beta,
                daysToEarnings,
                callPutRatio,
                callPutRatioOI: analysisData.options?.call_put_ratio_oi,
                callPutRatioVolume: analysisData.options?.call_put_ratio_volume,
                optionsSentiment: analysisData.options?.options_sentiment,
                avgImpliedVolatility: analysisData.options?.avg_implied_volatility,
                ivPercentile: analysisData.options?.iv_percentile,
                hasOptions: analysisData.options?.has_options ?? false,
                marketRegime: (analysisData as Record<string, unknown>).market_regime as string | null | undefined
              };
            }
            
            const analysis = analyzeStock(
              ticker,
              history,
              finalPrice,
              finalHigh52w,
              finalLow52w,
              weights,
              additionalData,
              calibrationMetadata,
              horizon,
              strategyWeightsMap // Pass strategy-specific weights
            );

            if (isLowConfidence) {
                analysis.status = 'low_confidence';
            } else {
                analysis.status = 'ok';
            }
            
            setCachedAnalysis(ticker, horizon, analysis);
            notifyListeners(ticker, horizon, analysis, null);
            
          } catch (err) {
            const message = err instanceof Error ? err.message : 'Analysis failed';
            console.error(`[Analysis] Failed for ${ticker}:`, message);
            setCachedAnalysis(ticker, horizon, null, message);
            notifyListeners(ticker, horizon, null, message);
          } finally {
            pendingRequests.delete(requestKey);
          }
        })
      );
      
    } catch (err) {
      // Fallback logic for batch failure
      console.error('[Analysis] Batch fetch failed, using individual fallback:', err);
      // ... (simplified fallback for brevity, logic remains similar)
      // For each request, try individual fetch and notify listeners
      await Promise.all(validRequests.map(async (request) => {
          const { ticker, horizon } = request;
          const requestKey = getCacheKey(ticker, horizon);
          try {
              // Minimal fallback implementation
              const analysis = await analyzeStockFallback(request); // Helper needed or inline
              setCachedAnalysis(ticker, horizon, analysis);
              notifyListeners(ticker, horizon, analysis, null);
          } catch (e: any) {
              setCachedAnalysis(ticker, horizon, null, e.message);
              notifyListeners(ticker, horizon, null, e.message);
          } finally {
              pendingRequests.delete(requestKey);
          }
      }));
    }
    
    if (requestQueue.length > 0) {
      await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY_MS));
    }
  }
  
  isProcessingQueue = false;
}

// Helper for fallback (simplified from previous version)
async function analyzeStockFallback(request: AnalysisRequest): Promise<StockAnalysis> {
    const { ticker, currentPrice, high52w, low52w } = request;
    const [historyResponse, stockData] = await Promise.all([
        getStockHistory(ticker, '1y'),
        getStock(ticker).catch(() => null)
    ]);
    const finalHigh52w = high52w ?? stockData?.high_52w ?? null;
    const finalLow52w = low52w ?? stockData?.low_52w ?? null;
    const finalPrice = currentPrice || stockData?.current_price || 0;
    
    let additionalData: AdditionalAnalysisData | undefined;
    try {
        const ad = await fetchStockAnalysisData(ticker);

        // Calculate days to earnings
        let daysToEarnings: number | null = null;
        const nextEarningsDate = ad.fundamentals?.next_earnings_date;
        if (nextEarningsDate) {
            const earningsDate = new Date(nextEarningsDate);
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            earningsDate.setHours(0, 0, 0, 0);
            daysToEarnings = Math.ceil((earningsDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
        }

        const callPutRatio = ad.options?.call_put_ratio_oi ?? ad.options?.call_put_ratio_volume ?? null;

        additionalData = {
            roic: ad.fundamentals?.roic,
            roe: ad.fundamentals?.roe,
            sector: ad.fundamentals?.sector,
            industry: ad.fundamentals?.industry,
            beta: ad.fundamentals?.beta,
            daysToEarnings,
            callPutRatio,
            callPutRatioOI: ad.options?.call_put_ratio_oi,
            callPutRatioVolume: ad.options?.call_put_ratio_volume,
            optionsSentiment: ad.options?.options_sentiment,
            avgImpliedVolatility: ad.options?.avg_implied_volatility,
            ivPercentile: ad.options?.iv_percentile,
            hasOptions: ad.options?.has_options ?? false,
            marketRegime: (ad as Record<string, unknown>).market_regime as string | null | undefined
        };
    } catch {}

    const analysis = analyzeStock(
        ticker, historyResponse.history, finalPrice, finalHigh52w, finalLow52w, undefined, additionalData, undefined, request.horizon
    );
    analysis.status = 'ok';
    return analysis;
}

// ============================================================================
// Hook: useStockAnalysis
// ============================================================================

export function useStockAnalysis(
  ticker: string | null,
  currentPrice?: number,
  high52w?: number | null,
  low52w?: number | null,
  horizon: number = 3
): AnalysisState {
  const [state, setState] = useState<AnalysisState>({
    analysis: null,
    isLoading: false,
    isStale: false,
    error: null,
    lastUpdated: null
  });
  
  const mountedRef = useRef(true);
  const lastRequestTime = useRef<number>(0);
  
  useEffect(() => {
    mountedRef.current = true;
    
    if (!ticker || !currentPrice) {
      setState(prev => ({ ...prev, analysis: null, isLoading: false }));
      return;
    }
    
    const cached = getCachedAnalysis(ticker, horizon);
    const key = getCacheKey(ticker, horizon);
    
    // Initial state from cache
    if (cached) {
      const isValid = isCacheValid(cached);
      setState({
        analysis: cached.analysis,
        isLoading: !isValid && !pendingRequests.has(key),
        isStale: !isValid,
        error: cached.error ?? null,
        lastUpdated: cached.fetchedAt
      });
      
      if (!isValid && !pendingRequests.has(key)) {
        const now = Date.now();
        if (now - lastRequestTime.current > 60000) {
            lastRequestTime.current = now;
            queueAnalysis(ticker, horizon, currentPrice, high52w ?? null, low52w ?? null);
        }
      }
    } else {
      const now = Date.now();
      if (now - lastRequestTime.current > 60000) {
          lastRequestTime.current = now;
          setState(prev => ({ ...prev, isLoading: true, analysis: null }));
          queueAnalysis(ticker, horizon, currentPrice, high52w ?? null, low52w ?? null);
      }
    }

    // Subscribe to updates
    const handleUpdate: AnalysisListener = (t, h, a, e) => {
        if (!mountedRef.current) return;
        if (t === ticker && h === horizon) {
            setState({
                analysis: a,
                isLoading: false,
                isStale: false,
                error: e,
                lastUpdated: new Date()
            });
        }
    };
    
    const unsubscribe = subscribe(handleUpdate);
    return () => {
        mountedRef.current = false;
        unsubscribe();
    };
  }, [ticker, horizon, currentPrice, high52w, low52w]);
  
  return state;
}

// ============================================================================
// Hook: useMultiStockAnalysis
// ============================================================================

export function useMultiStockAnalysis(
  stocks: Array<{
    ticker: string;
    currentPrice?: number;
    high52w?: number | null;
    low52w?: number | null;
  }>,
  horizon: number = 3
): { analyses: Map<string, AnalysisState>; isAnyLoading: boolean } {
  const [analyses, setAnalyses] = useState<Map<string, AnalysisState>>(new Map());
  const mountedRef = useRef(true);
  const lastRequestTimes = useRef<Map<string, number>>(new Map());
  
  useEffect(() => {
    mountedRef.current = true;
    const newAnalyses = new Map<string, AnalysisState>();
    const toQueue: AnalysisRequest[] = [];
    const now = Date.now();
    
    // Initialize state
    for (const stock of stocks) {
      if (!stock.ticker || !stock.currentPrice) continue;
      
      const cached = getCachedAnalysis(stock.ticker, horizon);
      const key = getCacheKey(stock.ticker, horizon);
      
      if (cached) {
        const isValid = isCacheValid(cached);
        newAnalyses.set(stock.ticker, {
          analysis: cached.analysis,
          isLoading: !isValid && !pendingRequests.has(key),
          isStale: !isValid,
          error: cached.error ?? null,
          lastUpdated: cached.fetchedAt
        });
        
        if (!isValid && !pendingRequests.has(key)) {
          const lastTime = lastRequestTimes.current.get(key) || 0;
          if (now - lastTime > 60000) {
             lastRequestTimes.current.set(key, now);
             toQueue.push({ ticker: stock.ticker, horizon, currentPrice: stock.currentPrice, high52w: stock.high52w ?? null, low52w: stock.low52w ?? null });
          }
        }
      } else {
        newAnalyses.set(stock.ticker, {
          analysis: null,
          isLoading: true,
          isStale: false,
          error: null,
          lastUpdated: null
        });
        
        const lastTime = lastRequestTimes.current.get(key) || 0;
        if (now - lastTime > 60000) {
             lastRequestTimes.current.set(key, now);
             toQueue.push({ ticker: stock.ticker, horizon, currentPrice: stock.currentPrice, high52w: stock.high52w ?? null, low52w: stock.low52w ?? null });
        }
      }
    }
    setAnalyses(newAnalyses);
    
    // Queue batch
    if (toQueue.length > 0) {
        toQueue.forEach(req => {
            const key = getCacheKey(req.ticker, req.horizon);
            if (!pendingRequests.has(key) && !requestQueue.some(r => r.ticker === req.ticker && r.horizon === req.horizon)) {
                requestQueue.push(req);
            }
        });
        startQueueProcessing();
    }

    // Subscribe
    const handleUpdate: AnalysisListener = (t, h, a, e) => {
        if (!mountedRef.current) return;
        if (h === horizon && stocks.some(s => s.ticker === t)) {
            setAnalyses(prev => {
                const next = new Map(prev);
                next.set(t, {
                    analysis: a,
                    isLoading: false,
                    isStale: false,
                    error: e,
                    lastUpdated: new Date()
                });
                return next;
            });
        }
    };

    const unsubscribe = subscribe(handleUpdate);
    return () => {
        mountedRef.current = false;
        unsubscribe();
    };
  }, [stocks.map(s => `${s.ticker}:${s.currentPrice}`).join(','), horizon]);
  
  const isAnyLoading = Array.from(analyses.values()).some(a => a.isLoading);
  return { analyses, isAnyLoading };
}

// Helper to trigger queue processing
function queueAnalysis(ticker: string, horizon: number, currentPrice: number, high52w: number | null, low52w: number | null) {
    const key = getCacheKey(ticker, horizon);
    if (pendingRequests.has(key)) return;
    if (requestQueue.some(r => r.ticker === ticker && r.horizon === horizon)) return;
    
    requestQueue.push({ ticker, horizon, currentPrice, high52w, low52w });
    startQueueProcessing();
}

function startQueueProcessing() {
    if (queueDebounceTimer) clearTimeout(queueDebounceTimer);
    queueDebounceTimer = setTimeout(() => {
        queueDebounceTimer = null;
        processQueue();
    }, QUEUE_DEBOUNCE_MS);
}

// Export utilities
export function refreshAnalysis(ticker: string, currentPrice: number, high52w: number | null, low52w: number | null, horizon: number = 3): void {
  analysisCache.delete(getCacheKey(ticker, horizon));
  queueAnalysis(ticker, horizon, currentPrice, high52w, low52w);
}

export function clearAnalysisCache(): void {
  analysisCache.clear();
}

export function getAnalysisCacheStats() {
  return { size: analysisCache.size, pending: pendingRequests.size, queued: requestQueue.length };
}

export function getCachedAnalysisScore(ticker: string, horizon: number = 3): number | null {
  const cached = analysisCache.get(getCacheKey(ticker, horizon));
  return cached?.analysis?.bestAction.totalScore ?? null;
}

// ============================================================================
// Types for Dual Horizon Analysis
// ============================================================================

export interface DualAnalysisState {
  /** Analysis for swing (3-day) horizon */
  swing: AnalysisState;
  /** Analysis for trend (15-day) horizon */
  trend: AnalysisState;
}

// ============================================================================
// Hook: useDualHorizonAnalysis
// ============================================================================

/**
 * Hook that fetches analysis for two horizons.
 * By default uses swing (3d) and trend (15d), but can be configured.
 * Returns analysis state for both horizons simultaneously.
 */
export function useDualHorizonAnalysis(
  ticker: string | null,
  currentPrice?: number,
  high52w?: number | null,
  low52w?: number | null,
  horizonA: number = 3,
  horizonB: number = 15
): DualAnalysisState {
  const swing = useStockAnalysis(ticker, currentPrice, high52w, low52w, horizonA);
  const trend = useStockAnalysis(ticker, currentPrice, high52w, low52w, horizonB);

  return { swing, trend };
}
