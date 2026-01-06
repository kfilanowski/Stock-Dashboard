/**
 * API service for the Stock Dashboard.
 * 
 * All API calls are centralized here for maintainability.
 * Uses versioned API endpoints (v1) with fallback support.
 */
import type { Portfolio, StockData, PortfolioHistoryPoint, StockHistoryResponse } from '../types';

// API Configuration
const API_VERSION = 'v1';
const API_BASE = `/api/${API_VERSION}`;

/**
 * Handle API response and extract JSON or throw error.
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

// ============================================================================
// Portfolio API
// ============================================================================

/**
 * Get the portfolio with holdings.
 * @param lite - If true, returns structure without live stock data (faster)
 */
export async function getPortfolio(lite = false): Promise<Portfolio> {
  const url = lite ? `${API_BASE}/portfolio?lite=true` : `${API_BASE}/portfolio`;
  const response = await fetch(url);
  return handleResponse<Portfolio>(response);
}

export interface UpdatePortfolioData {
  name?: string;
  total_value?: number;
  chart_period?: string;
  sort_field?: string;
  sort_direction?: string;
}

/**
 * Update portfolio settings.
 */
export async function updatePortfolio(data: UpdatePortfolioData): Promise<Portfolio> {
  const response = await fetch(`${API_BASE}/portfolio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Portfolio>(response);
}

// ============================================================================
// Holdings API
// ============================================================================

export interface AddHoldingData {
  ticker: string;
  allocation_pct: number;
  investment_date?: string | null;
  investment_price?: number | null;
}

/**
 * Add a new holding to the portfolio.
 */
export async function addHolding(data: AddHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

/**
 * Delete a holding from the portfolio.
 */
export async function deleteHolding(holdingId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'DELETE',
  });
  return handleResponse<void>(response);
}

export interface UpdateHoldingData {
  allocation_pct?: number;
  investment_date?: string | null;
  investment_price?: number | null;
}

/**
 * Update a holding's allocation or investment info.
 */
export async function updateHolding(holdingId: number, data: UpdateHoldingData): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<void>(response);
}

// ============================================================================
// Stock API
// ============================================================================

/**
 * Get detailed stock data including price, YTD, SMA, and history.
 */
export async function getStock(ticker: string): Promise<StockData> {
  const response = await fetch(`${API_BASE}/stock/${ticker}`);
  return handleResponse<StockData>(response);
}

export interface StockQuote {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  ytd_return: number;
  sma_200: number | null;
  price_vs_sma: number | null;
}

/**
 * Get just the current price data for a single stock (lightweight).
 */
export async function getStockQuote(ticker: string): Promise<StockQuote> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/quote`);
  return handleResponse<StockQuote>(response);
}

/**
 * Get historical data for a stock.
 * @param ticker - Stock ticker symbol
 * @param period - Time period (1d, 3d, 1w, 1mo, 3mo, 6mo, ytd, 1y, 2y)
 */
export async function getStockHistory(ticker: string, period: string = '1y'): Promise<StockHistoryResponse> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/history?period=${period}`);
  return handleResponse<StockHistoryResponse>(response);
}

/**
 * Get historical data for multiple stocks at once.
 * More efficient than calling individual endpoints.
 */
export async function getMultipleStockHistories(
  tickers: string[], 
  period: string = '1y'
): Promise<Record<string, StockHistoryResponse>> {
  const response = await fetch(`${API_BASE}/stocks/history?period=${period}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(tickers),
  });
  return handleResponse<Record<string, StockHistoryResponse>>(response);
}

// ============================================================================
// Batch & Incremental API (Optimized for fast updates)
// ============================================================================

export interface ChartPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface BatchPriceData {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  market_state: string;
  chart_point: ChartPoint | null;
}

/**
 * Get current prices for multiple tickers in one efficient batch call.
 * 
 * This is optimized for frequent polling - only fetches price data,
 * not historical data or other metadata. Much faster than individual calls.
 * 
 * @param tickers - Array of ticker symbols
 * @returns Dict of ticker -> price data
 */
export async function getBatchPrices(tickers: string[]): Promise<Record<string, BatchPriceData>> {
  if (!tickers.length) return {};
  
  const tickerStr = tickers.join(',');
  const response = await fetch(`${API_BASE}/prices?tickers=${encodeURIComponent(tickerStr)}`);
  return handleResponse<Record<string, BatchPriceData>>(response);
}

export interface IncrementalHistoryPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface IncrementalHistoryResult {
  new_points: IncrementalHistoryPoint[];
  latest_timestamp: string | null;
  count?: number;
}

export interface IncrementalHistoryRequest {
  tickers: string[];
  interval: string;
  since_timestamps?: Record<string, string | null>;
}

/**
 * Get only NEW intraday history data since the provided timestamps.
 * 
 * This is the key function for efficient chart updates:
 * - Track the latest timestamp you have for each ticker
 * - Call this with those timestamps
 * - Get back only new data points (typically 1-5 points)
 * - Append to existing chart data
 * 
 * This reduces bandwidth by ~99% compared to fetching full history.
 * 
 * @param request - Tickers, interval, and since_timestamps
 * @returns Dict of ticker -> {new_points, latest_timestamp}
 */
export async function getIncrementalHistory(
  request: IncrementalHistoryRequest
): Promise<Record<string, IncrementalHistoryResult>> {
  const response = await fetch(`${API_BASE}/stocks/history/incremental`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<Record<string, IncrementalHistoryResult>>(response);
}

// ============================================================================
// Portfolio History API
// ============================================================================

/**
 * Get portfolio value history over time.
 */
export async function getPortfolioHistory(): Promise<{ history: PortfolioHistoryPoint[] }> {
  const response = await fetch(`${API_BASE}/portfolio/history`);
  return handleResponse(response);
}

/**
 * Create a snapshot of current portfolio value.
 */
export async function createPortfolioSnapshot(): Promise<void> {
  const response = await fetch(`${API_BASE}/portfolio/snapshot`, { method: 'POST' });
  return handleResponse<void>(response);
}

// ============================================================================
// Cache Management API
// ============================================================================

/**
 * Clear the intraday cache for a specific stock.
 * Forces fresh data fetch on next request.
 */
export async function clearStockCache(ticker: string): Promise<{ message: string; deleted_records: number }> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/cache`, {
    method: 'DELETE',
  });
  return handleResponse(response);
}

// ============================================================================
// Health API
// ============================================================================

export interface HealthStatus {
  status: string;
  service: string;
  version: string;
}

/**
 * Check API health status.
 */
export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/health`);
  return handleResponse<HealthStatus>(response);
}
