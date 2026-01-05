import type { Portfolio, StockData, PortfolioHistoryPoint, StockHistoryResponse } from '../types';

const API_BASE = '/api';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || 'Request failed');
  }
  return response.json();
}

// Portfolio API
export async function getPortfolio(): Promise<Portfolio> {
  const response = await fetch(`${API_BASE}/portfolio`);
  return handleResponse<Portfolio>(response);
}

export async function updatePortfolio(data: { name?: string; total_value?: number }): Promise<Portfolio> {
  const response = await fetch(`${API_BASE}/portfolio`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return handleResponse<Portfolio>(response);
}

// Holdings API
export async function addHolding(ticker: string, allocation_pct: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ticker, allocation_pct }),
  });
  return handleResponse<void>(response);
}

export async function deleteHolding(holdingId: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'DELETE',
  });
  return handleResponse<void>(response);
}

export async function updateHolding(holdingId: number, allocation_pct: number): Promise<void> {
  const response = await fetch(`${API_BASE}/holdings/${holdingId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ allocation_pct }),
  });
  return handleResponse<void>(response);
}

// Stock API
export async function getStock(ticker: string): Promise<StockData> {
  const response = await fetch(`${API_BASE}/stock/${ticker}`);
  return handleResponse<StockData>(response);
}

export async function getStockHistory(ticker: string, period: string = '1y'): Promise<StockHistoryResponse> {
  const response = await fetch(`${API_BASE}/stock/${ticker}/history?period=${period}`);
  return handleResponse<StockHistoryResponse>(response);
}

// Portfolio History
export async function getPortfolioHistory(): Promise<{ history: PortfolioHistoryPoint[] }> {
  const response = await fetch(`${API_BASE}/portfolio/history`);
  return handleResponse(response);
}

export async function createPortfolioSnapshot(): Promise<void> {
  const response = await fetch(`${API_BASE}/portfolio/snapshot`, { method: 'POST' });
  return handleResponse<void>(response);
}

