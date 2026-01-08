/**
 * Centralized Data Cache Service
 * 
 * Provides a single source of truth for price and history data.
 * Prevents duplicate API calls by caching data with staleness tracking.
 * 
 * Design principles (per GUIDELINES.md):
 * - Batch-first: Always prefer batch endpoints
 * - SQLite First, API Second: Backend handles this, we just cache results
 * - Real-time prices: Continuous refresh loop with 3-second minimum interval
 */

import type { HistoryPoint } from '../types';
import type { ChartPoint } from './api';
import * as api from './api';

// ============================================================================
// Types
// ============================================================================

/**
 * Generic cache entry with staleness tracking.
 */
export interface CacheEntry<T> {
  data: T;
  fetchedAt: Date;
}

/**
 * History cache entry includes period information.
 */
export interface HistoryCacheEntry {
  history: HistoryPoint[];
  referenceClose: number | null;
  isComplete: boolean;
  expectedStart: string | null;
  actualStart: string | null;
  fetchedAt: Date;
  period: string;
}

/**
 * Price data with 52-week range for analysis.
 */
export interface PriceData {
  currentPrice: number;
  previousClose: number;
  change: number;
  changePct: number;
  marketState: string;
  chartPoint: ChartPoint | null;
  high52w: number | null;
  low52w: number | null;
}

/**
 * Listener callback for cache updates.
 */
export type CacheListener<T> = (ticker: string, data: T) => void;

/**
 * Configuration for the refresh loop.
 */
export interface RefreshConfig {
  minInterval: number;  // Minimum ms between price fetches
  enabled: boolean;     // Whether refresh loop is running
}

// ============================================================================
// DataCacheService Class
// ============================================================================

/**
 * Centralized cache for stock data.
 * 
 * Usage:
 * - Call setTickers() when portfolio changes
 * - Call setChartPeriod() when user changes period
 * - Subscribe to updates via onPriceUpdate/onHistoryUpdate
 * - Call startRefreshLoop() to begin background updates
 */
export class DataCacheService {
  // Cache stores
  private priceCache: Map<string, CacheEntry<PriceData>> = new Map();
  private historyCache: Map<string, HistoryCacheEntry> = new Map();
  
  // Current state
  private tickers: string[] = [];
  private chartPeriod: string = '1d';
  
  // Refresh control
  private refreshConfig: RefreshConfig = {
    minInterval: 3000,
    enabled: false
  };
  private refreshLoopRunning = false;
  private abortController: AbortController | null = null;
  
  // Event listeners
  private priceListeners: Set<CacheListener<PriceData>> = new Set();
  private historyListeners: Set<CacheListener<HistoryCacheEntry>> = new Set();
  private batchHistoryListeners: Set<() => void> = new Set();
  
  // Timestamps for staleness
  private lastPricesFetchedAt: Date | null = null;
  private lastHistoryFetchedAt: Date | null = null;
  
  // ============================================================================
  // Getters
  // ============================================================================
  
  /**
   * Get cached price data for a ticker.
   */
  getPrice(ticker: string): PriceData | null {
    return this.priceCache.get(ticker)?.data ?? null;
  }
  
  /**
   * Get all cached prices.
   */
  getAllPrices(): Map<string, PriceData> {
    const result = new Map<string, PriceData>();
    for (const [ticker, entry] of this.priceCache) {
      result.set(ticker, entry.data);
    }
    return result;
  }
  
  /**
   * Get cached history for a ticker and period.
   * Returns null if not cached or cached for different period.
   */
  getHistory(ticker: string, period?: string): HistoryCacheEntry | null {
    const entry = this.historyCache.get(ticker);
    if (!entry) return null;
    // If period specified, only return if it matches
    if (period && entry.period !== period) return null;
    return entry;
  }
  
  /**
   * Get all cached history entries.
   */
  getAllHistory(): Map<string, HistoryCacheEntry> {
    return new Map(this.historyCache);
  }
  
  /**
   * Get when prices were last fetched.
   */
  getLastPricesFetchedAt(): Date | null {
    return this.lastPricesFetchedAt;
  }
  
  /**
   * Get when history was last fetched.
   */
  getLastHistoryFetchedAt(): Date | null {
    return this.lastHistoryFetchedAt;
  }
  
  /**
   * Get current chart period.
   */
  getChartPeriod(): string {
    return this.chartPeriod;
  }
  
  /**
   * Get current tickers being tracked.
   */
  getTickers(): string[] {
    return [...this.tickers];
  }
  
  /**
   * Check if refresh loop is running.
   */
  isRefreshing(): boolean {
    return this.refreshLoopRunning;
  }
  
  // ============================================================================
  // Setters / State Management
  // ============================================================================
  
  /**
   * Set the tickers to track.
   * Triggers history fetch for any new tickers.
   */
  setTickers(tickers: string[]): void {
    const newTickers = tickers.filter(t => !this.tickers.includes(t));
    const removedTickers = this.tickers.filter(t => !tickers.includes(t));
    
    this.tickers = [...tickers];
    
    // Clean up cache for removed tickers
    for (const ticker of removedTickers) {
      this.priceCache.delete(ticker);
      this.historyCache.delete(ticker);
    }
    
    // Fetch history for new tickers if we have any
    if (newTickers.length > 0 && this.refreshConfig.enabled) {
      this.fetchBatchHistory(newTickers).catch(console.error);
    }
  }
  
  /**
   * Set the chart period.
   * Clears history cache and fetches new data for all tickers.
   */
  async setChartPeriod(period: string): Promise<void> {
    if (period === this.chartPeriod) return;
    
    const oldPeriod = this.chartPeriod;
    this.chartPeriod = period;
    
    // Clear all history cache (period changed)
    this.historyCache.clear();
    
    console.log(`[DataCache] Period changed from ${oldPeriod} to ${period}, fetching new history...`);
    
    // Fetch history for all tickers with new period
    if (this.tickers.length > 0) {
      await this.fetchBatchHistory(this.tickers);
    }
  }
  
  // ============================================================================
  // Subscription Methods
  // ============================================================================
  
  /**
   * Subscribe to price updates for individual tickers.
   */
  onPriceUpdate(listener: CacheListener<PriceData>): () => void {
    this.priceListeners.add(listener);
    return () => this.priceListeners.delete(listener);
  }
  
  /**
   * Subscribe to history updates for individual tickers.
   */
  onHistoryUpdate(listener: CacheListener<HistoryCacheEntry>): () => void {
    this.historyListeners.add(listener);
    return () => this.historyListeners.delete(listener);
  }
  
  /**
   * Subscribe to batch history fetch completions.
   */
  onBatchHistoryComplete(listener: () => void): () => void {
    this.batchHistoryListeners.add(listener);
    return () => this.batchHistoryListeners.delete(listener);
  }
  
  // ============================================================================
  // Fetch Methods
  // ============================================================================
  
  /**
   * Fetch prices for all tracked tickers.
   * Also updates 1d history with chart_point data.
   */
  async fetchPrices(): Promise<void> {
    if (this.tickers.length === 0) return;
    
    try {
      const priceData = await api.getBatchPrices(this.tickers);
      const now = new Date();
      this.lastPricesFetchedAt = now;
      
      for (const [ticker, data] of Object.entries(priceData)) {
        const priceEntry: PriceData = {
          currentPrice: data.current_price,
          previousClose: data.previous_close,
          change: data.change,
          changePct: data.change_pct,
          marketState: data.market_state,
          chartPoint: data.chart_point,
          high52w: data.high_52w,
          low52w: data.low_52w
        };
        
        this.priceCache.set(ticker, { data: priceEntry, fetchedAt: now });
        
        // Notify price listeners
        for (const listener of this.priceListeners) {
          listener(ticker, priceEntry);
        }
        
        // For 1d period, append chart_point to history
        if (this.chartPeriod === '1d' && data.chart_point) {
          this.appendChartPoint(ticker, data.chart_point);
        }
      }
    } catch (err) {
      console.error('[DataCache] Failed to fetch prices:', err);
    }
  }
  
  /**
   * Fetch history for specified tickers.
   */
  async fetchBatchHistory(tickers: string[]): Promise<void> {
    if (tickers.length === 0) return;
    
    const period = this.chartPeriod;
    console.log(`[DataCache] Batch fetching history for ${tickers.length} tickers, period: ${period}`);
    const startTime = Date.now();
    
    try {
      const historyData = await api.getBatchHistory(tickers, period);
      const now = new Date();
      this.lastHistoryFetchedAt = now;
      
      const elapsed = Date.now() - startTime;
      console.log(`[DataCache] Batch history fetch completed in ${elapsed}ms`);
      
      for (const [ticker, data] of Object.entries(historyData)) {
        if (!data?.history?.length) continue;
        
        const entry: HistoryCacheEntry = {
          history: data.history,
          referenceClose: data.reference_close,
          isComplete: data.is_complete ?? false,
          expectedStart: null,
          actualStart: data.history[0]?.date ?? null,
          fetchedAt: now,
          period
        };
        
        this.historyCache.set(ticker, entry);
        
        // Notify history listeners
        for (const listener of this.historyListeners) {
          listener(ticker, entry);
        }
      }
      
      // Notify batch completion listeners
      for (const listener of this.batchHistoryListeners) {
        listener();
      }
    } catch (err) {
      console.error('[DataCache] Failed to batch fetch history:', err);
    }
  }
  
  /**
   * Fetch single ticker history (for modals viewing different period).
   */
  async fetchSingleHistory(ticker: string, period: string): Promise<HistoryCacheEntry | null> {
    try {
      const result = await api.getStockHistory(ticker, period);
      const now = new Date();
      
      const entry: HistoryCacheEntry = {
        history: result.history,
        referenceClose: result.reference_close,
        isComplete: result.is_complete,
        expectedStart: result.expected_start,
        actualStart: result.actual_start,
        fetchedAt: now,
        period
      };
      
      // Only cache if it matches current period
      if (period === this.chartPeriod) {
        this.historyCache.set(ticker, entry);
        
        for (const listener of this.historyListeners) {
          listener(ticker, entry);
        }
      }
      
      return entry;
    } catch (err) {
      console.error(`[DataCache] Failed to fetch history for ${ticker}:`, err);
      return null;
    }
  }
  
  /**
   * Append a chart point to existing history (for real-time 1d updates).
   */
  private appendChartPoint(ticker: string, chartPoint: ChartPoint): void {
    const existing = this.historyCache.get(ticker);
    if (!existing || existing.period !== '1d') return;
    
    const existingHistory = existing.history;
    const lastPoint = existingHistory[existingHistory.length - 1];
    
    // Convert chart point to history point format
    const newPoint: HistoryPoint = {
      date: chartPoint.date,
      open: chartPoint.open,
      high: chartPoint.high,
      low: chartPoint.low,
      close: chartPoint.close,
      volume: chartPoint.volume
    };
    
    let updatedHistory: HistoryPoint[];
    
    if (lastPoint?.date === chartPoint.date) {
      // Same minute - update the last point (OHLC update)
      updatedHistory = [...existingHistory.slice(0, -1), newPoint];
    } else if (!lastPoint || chartPoint.date > lastPoint.date) {
      // New minute - append the point
      updatedHistory = [...existingHistory, newPoint];
    } else {
      // Point is older than existing, skip
      return;
    }
    
    const updatedEntry: HistoryCacheEntry = {
      ...existing,
      history: updatedHistory,
      fetchedAt: new Date()
    };
    
    this.historyCache.set(ticker, updatedEntry);
    
    // Notify listeners
    for (const listener of this.historyListeners) {
      listener(ticker, updatedEntry);
    }
  }
  
  // ============================================================================
  // Refresh Loop
  // ============================================================================
  
  /**
   * Start the background refresh loop.
   * Fetches prices continuously with minimum interval.
   */
  async startRefreshLoop(): Promise<void> {
    if (this.refreshLoopRunning) {
      console.log('[DataCache] Refresh loop already running, skipping');
      return;
    }
    
    this.refreshConfig.enabled = true;
    this.refreshLoopRunning = true;
    this.abortController = new AbortController();
    
    console.log('[DataCache] Starting refresh loop...');
    
    try {
      // Initial batch history fetch
      if (this.tickers.length > 0) {
        await this.fetchBatchHistory(this.tickers);
        console.log('[DataCache] Initial history fetch complete, starting price loop');
      }
      
      // Price refresh loop
      while (this.refreshConfig.enabled && !this.abortController.signal.aborted) {
        const cycleStartTime = Date.now();
        
        // Fetch prices
        await this.fetchPrices();
        
        if (!this.refreshConfig.enabled) break;
        
        // Check for tickers missing history
        const missingHistoryTickers = this.tickers.filter(
          t => !this.historyCache.has(t) || this.historyCache.get(t)?.history.length === 0
        );
        
        if (missingHistoryTickers.length > 0) {
          console.log(`[DataCache] Fetching history for ${missingHistoryTickers.length} tickers missing data`);
          await this.fetchBatchHistory(missingHistoryTickers);
        }
        
        if (!this.refreshConfig.enabled) break;
        
        // Wait minimum interval before next fetch
        const elapsed = Date.now() - cycleStartTime;
        const remainingWait = this.refreshConfig.minInterval - elapsed;
        if (remainingWait > 0) {
          await this.sleep(remainingWait);
        }
      }
    } finally {
      this.refreshLoopRunning = false;
      console.log('[DataCache] Refresh loop stopped');
    }
  }
  
  /**
   * Stop the background refresh loop.
   */
  stopRefreshLoop(): void {
    console.log('[DataCache] Stopping refresh loop...');
    this.refreshConfig.enabled = false;
    this.abortController?.abort();
  }
  
  /**
   * Clear all cached data.
   */
  clear(): void {
    this.priceCache.clear();
    this.historyCache.clear();
    this.lastPricesFetchedAt = null;
    this.lastHistoryFetchedAt = null;
  }
  
  /**
   * Clear history for a specific ticker (e.g., after user requests refresh).
   */
  clearTickerHistory(ticker: string): void {
    this.historyCache.delete(ticker);
  }
  
  // ============================================================================
  // Utilities
  // ============================================================================
  
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let dataCacheInstance: DataCacheService | null = null;

/**
 * Get the singleton DataCacheService instance.
 */
export function getDataCache(): DataCacheService {
  if (!dataCacheInstance) {
    dataCacheInstance = new DataCacheService();
  }
  return dataCacheInstance;
}

/**
 * Create a new DataCacheService instance (for testing or isolated use).
 */
export function createDataCache(): DataCacheService {
  return new DataCacheService();
}

