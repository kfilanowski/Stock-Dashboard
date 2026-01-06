export interface Holding {
  id: number;
  portfolio_id: number;
  ticker: string;
  allocation_pct: number;
  added_at: string;
  investment_date?: string | null;
  investment_price?: number | null;
  current_price?: number;
  current_value?: number;
  ytd_return?: number;
  sma_200?: number;
  price_vs_sma?: number;
  gain_loss?: number | null;  // Gain/loss since investment_date in dollars
  gain_loss_pct?: number | null;  // Gain/loss since investment_date in %
}

export interface Portfolio {
  id: number;
  name: string;
  total_value: number;
  created_at: string;
  updated_at: string;
  holdings: Holding[];
  current_total_value?: number;
  total_gain_loss?: number;
  total_gain_loss_pct?: number;
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
 */
export type ActionType = 
  | 'buyShares'
  | 'sellShares'
  | 'buyCSP'      // Cash Secured Put
  | 'buyCC'       // Covered Call
  | 'buyCall'
  | 'buyPut';

/**
 * Display names for actions.
 */
export const ACTION_LABELS: Record<ActionType, string> = {
  buyShares: 'Buy Shares',
  sellShares: 'Sell Shares',
  buyCSP: 'Sell CSP',
  buyCC: 'Sell CC',
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
  dataQuality: {
    availableMetrics: number;
    totalMetrics: number;
    missingMetrics: MetricType[];
    historyDays: number;        // How many days of history were used
    hasSMA200: boolean;         // Whether SMA200 could be calculated
  };
}

