export interface Holding {
  id: number;
  portfolio_id: number;
  ticker: string;
  shares: number;  // Number of shares owned
  avg_cost?: number | null;  // Average cost per share
  added_at: string;
  current_price?: number;
  market_value?: number | null;  // shares × current_price
  cost_basis?: number | null;  // shares × avg_cost
  allocation_pct?: number | null;  // Calculated: market_value / total_market_value × 100
  ytd_return?: number;
  sma_200?: number;
  price_vs_sma?: number;
  gain_loss?: number | null;  // market_value - cost_basis
  gain_loss_pct?: number | null;  // (current_price - avg_cost) / avg_cost × 100
}

export interface Portfolio {
  id: number;
  name: string;
  total_value: number;
  created_at: string;
  updated_at: string;
  holdings: Holding[];
  total_market_value?: number;  // Sum of all holdings' market values
  total_cost_basis?: number | null;  // Sum of all cost basis
  total_gain_loss?: number | null;
  total_gain_loss_pct?: number | null;
  // User preferences (persisted in database)
  chart_period?: string;
  sort_field?: string;
  sort_direction?: string;
}

export interface StockData {
  ticker: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_pct: number;
  ytd_return: number;
  sma_200: number | null;
  price_vs_sma: number | null;
  high_52w: number | null;
  low_52w: number | null;
  history: HistoryPoint[];
}

export interface HistoryPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistoryResponse {
  ticker: string;
  history: HistoryPoint[];
  reference_close: number | null;
  is_complete: boolean;
  expected_start: string | null;
  actual_start: string | null;
}

// ============================================================================
// Stock Scoring Types
// ============================================================================

/**
 * Trading actions that can be scored/recommended.
 * 
 * Options terminology:
 * - openCSP: Open a Cash-Secured Put position (sell a put to collect premium)
 * - openCC: Open a Covered Call position (sell a call to collect premium)
 * - buyCall/buyPut: Buy options (pay premium, long gamma/vega)
 */
export type ActionType = 
  | 'buyShares'
  | 'sellShares'
  | 'openCSP'     // Open Cash-Secured Put (sell put to collect premium)
  | 'openCC'      // Open Covered Call (sell call to collect premium)
  | 'buyCall'
  | 'buyPut';

/**
 * Display names for actions.
 */
export const ACTION_LABELS: Record<ActionType, string> = {
  buyShares: 'Buy Shares',
  sellShares: 'Sell Shares',
  openCSP: 'Open CSP',
  openCC: 'Open CC',
  buyCall: 'Buy Call',
  buyPut: 'Buy Put'
};

/**
 * Technical metrics used for scoring.
 */
export type MetricType =
  | 'rsi'
  | 'macd'
  | 'bollingerBands'
  | 'bollingerSqueeze'  // Volatility contraction/expansion
  | 'vwap'
  | 'momentum'
  | 'volume'
  | 'pricePosition'
  | 'smaAlignment'
  | 'rvol'              // Relative Volume
  | 'adx'               // Average Directional Index (trend strength)
  | 'crossPattern'      // Golden Cross / Death Cross
  | 'roic'              // Return on Invested Capital
  | 'callPutRatio'      // Options Call/Put ratio
  | 'ivPercentile'      // Implied Volatility percentile
  | 'sectorBeta';       // Beta to sector

/**
 * A signal from a single metric for a specific action.
 * Signal ranges from -1 (strongly against) to +1 (strongly for).
 */
export interface MetricSignal {
  metric: MetricType;
  metricLabel: string;
  rawValue: string;           // Human-readable value (e.g., "RSI: 25")
  signal: number;             // -1 to +1
  weight: number;             // Weight for this metric (default 1.0)
  contribution: number;       // signal * weight
  reasoning: string;          // Why this signal was given
}

/**
 * Score for a single action, with breakdown by metric.
 */
export interface ActionScore {
  action: ActionType;
  label: string;
  totalScore: number;         // Sum of all weighted signals, normalized 0-100
  rawScore: number;           // Sum before normalization
  signals: MetricSignal[];    // Individual metric contributions
  confidence: 'high' | 'medium' | 'low';  // Based on data availability
}

/**
 * Complete stock analysis with all action scores.
 */
export interface StockAnalysis {
  ticker: string;
  analyzedAt: Date;
  scores: ActionScore[];
  bestAction: ActionScore;
  hasOptions: boolean;          // Whether options are available for this stock
  dataQuality: {
    availableMetrics: number;
    totalMetrics: number;
    missingMetrics: MetricType[];
    historyDays: number;        // How many days of history were used
    hasSMA200: boolean;         // Whether SMA200 could be calculated
  };
}

