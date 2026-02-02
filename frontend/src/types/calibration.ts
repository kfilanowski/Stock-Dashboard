/**
 * TypeScript types for Walk-Forward Optimization (WFO) Calibration
 * 
 * Isolated from main types/index.ts to keep calibration code separate.
 */

// ============================================================================
// Weight Types
// ============================================================================

/**
 * Indicator weight configuration.
 * Maps indicator names to their weight values (0.0 to 2.5).
 *
 * Direction parameters (_dir suffix) control signal interpretation:
 * - +1.0 = mean reversion (oversold = bullish)
 * - -1.0 = momentum (oversold = bearish continuation)
 * - 0.0 = neutral/disabled
 */
export interface WeightMatrix {
  rsi: number;
  macd: number;
  bollinger: number;
  adx: number;
  cmf: number;
  momentum: number;
  volume: number;
  rvol: number;
  sma: number;
  position: number;
  squeeze: number;
  cross?: number;
  vwap?: number;
  // Relative strength indicators (vs sector)
  rel_momentum?: number;
  rs_ratio?: number;
  // Signal direction parameters (-1 to +1)
  // Allows optimizer to flip mean-reversion signals to momentum signals
  rsi_dir?: number;        // RSI signal direction
  bollinger_dir?: number;  // Bollinger signal direction
  position_dir?: number;   // Price position signal direction
  vwap_dir?: number;       // VWAP signal direction
}

/**
 * Calibrated weight record from database.
 */
export interface CalibrationWeight {
  ticker: string;
  indicator: string;
  action: string;
  horizon: number;
  weight: number;
  sqn_score: number | null;
  stability_passed: boolean;
  // Overfit detection (added for train/test SQN comparison)
  overfit_warning?: boolean;  // True if avg_test_sqn < 0.5 * avg_train_sqn
  avg_train_sqn?: number | null;  // Average in-sample SQN across windows
  avg_test_sqn?: number | null;   // Average out-of-sample SQN across windows
  calibrated_at: string;
}

// ============================================================================
// Window Types
// ============================================================================

/**
 * Rolling window configuration for WFO.
 */
export interface CalibrationWindow {
  id: number;
  ticker: string;
  horizon: number;
  train_start: string;
  train_end: string;
  test_start: string;
  test_end: string;
  window_days: number;
  weights_json: string;
  train_sqn: number;
  test_sqn: number;
  expectancy: number;
  trades_count: number;
  created_at: string;
}

/**
 * Simulated trade from WFO backtest.
 */
export interface SimulatedTrade {
  id: number;
  window_id: number;
  ticker: string;
  entry_date: string;
  exit_date: string;
  horizon: number;
  direction: 'long' | 'short';
  action: string;
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
  transaction_cost: number;
  market_regime: MarketRegime | null;
}

// ============================================================================
// Progress Types (for SSE streaming)
// ============================================================================

/**
 * Real-time progress update during calibration.
 */
export interface CalibrationProgress {
  ticker: string;
  horizon: number;
  stage: CalibrationStage;
  progress: number;  // 0-100
  current_indicator?: string;
  message?: string;
  weights?: Partial<WeightMatrix>;
  train_sqn?: number;
  test_sqn?: number;
  gross_sqn?: number;  // Gross SQN (before costs) - shows signal quality
}

export type CalibrationStage = 
  | 'loading'     // Loading price history
  | 'optimizing'  // Running coordinate descent
  | 'testing'     // Evaluating on test window
  | 'saving'      // Persisting to database
  | 'complete'    // Successfully finished
  | 'error';      // Failed with error

// ============================================================================
// Result Types
// ============================================================================

/**
 * Result of optimizing a single indicator.
 */
export interface IndicatorOptResult {
  indicator: string;
  optimal_weight: number;
  sqn_score: number;
  stability_passed: boolean;
  coarse_results: Record<number, number>;  // weight -> SQN
  fine_results: Record<number, number>;    // weight -> SQN
}

/**
 * Full optimization result for a ticker/horizon.
 */
export interface OptimizationResult {
  ticker: string;
  horizon: number;
  weights: WeightMatrix;
  train_sqn: number;
  total_trades: number;
  per_indicator: Record<string, IndicatorOptResult>;
  optimized_at: string;
}

/**
 * Calibration API response for a single horizon.
 */
export interface HorizonResult {
  sqn: number | null;            // Net SQN (after transaction costs and slippage)
  gross_sqn?: number | null;     // Gross SQN (before costs) - shows signal quality
  trades: number;
  weights: Partial<WeightMatrix> | null;
  error: string | null;
  reduced_confidence?: boolean;  // True if trades < 30 (calibrated but with lower statistical confidence)
  overfit_warning?: boolean;     // True if test SQN < 50% of train SQN (may be overfitting)
  avg_train_sqn?: number | null; // Average in-sample SQN across rolling windows
  avg_test_sqn?: number | null;  // Average out-of-sample SQN across rolling windows
}

/**
 * Full calibration API response.
 *
 * Results are nested by strategy class, then by horizon:
 * { "directional": { "5": HorizonResult, "21": HorizonResult }, ... }
 */
export interface CalibrationResponse {
  status: 'complete' | 'error';
  ticker: string;
  horizons?: number[];
  strategy_classes?: string[];
  results?: Record<string, Record<string, HorizonResult>>;
  error?: string;
  error_code?: 'INSUFFICIENT_VOLATILITY' | 'INSUFFICIENT_DATA';
}

// ============================================================================
// Weight Drift Types
// ============================================================================

/**
 * Weight drift analysis for detecting regime changes.
 */
export interface WeightDrift {
  indicator: string;
  old_weight: number;
  new_weight: number;
  drift: number;  // Absolute change
  stability: 'stable' | 'moderate' | 'noisy';
}

/**
 * Historical weight record for drift visualization.
 */
export interface WeightHistory {
  ticker: string;
  horizon: number;
  date: string;
  weights: Partial<WeightMatrix>;
  sqn: number;
}

// ============================================================================
// Market Regime Types
// ============================================================================

/**
 * 6-state market regime classification.
 */
export type MarketRegime = 
  | 'BULL_QUIET'       // Trending up, low volatility (grind up)
  | 'BULL_VOLATILE'    // Trending up, high volatility (shakeouts)
  | 'BEAR_QUIET'       // Trending down, low volatility (slow bleed)
  | 'BEAR_VOLATILE'    // Trending down, high volatility (crash mode)
  | 'NEUTRAL_CHOP'     // Flat, low volatility (range-bound)
  | 'NEUTRAL_VOLATILE'; // Flat, high volatility (breakout imminent)

/**
 * Regime-specific strategy recommendations.
 */
export const REGIME_STRATEGIES: Record<MarketRegime, string> = {
  BULL_QUIET: 'Buy Calls, Trend Following',
  BULL_VOLATILE: 'Sell Puts (Dip Buying)',
  BEAR_QUIET: 'Slow Bleed - Avoid',
  BEAR_VOLATILE: 'Long Puts (Crash Mode) - Force Cash',
  NEUTRAL_CHOP: 'Mean Reversion, Iron Condors',
  NEUTRAL_VOLATILE: 'Breakout Imminent - Wait'
};

// ============================================================================
// API Request/Response Types
// ============================================================================

/**
 * Available optimizer types for calibration.
 */
export type OptimizerType =
  | 'coordinate_descent'      // Fast, greedy two-pass optimization (default)
  | 'differential_evolution'  // Global optimizer using scipy
  | 'hybrid';                 // DE + coordinate descent refinement

/**
 * Request to start calibration.
 */
export interface CalibrationRequest {
  ticker: string;
  horizons?: number[];  // Default: [3, 15]
  optimizer?: OptimizerType; // Default: 'coordinate_descent'
}

/**
 * Response for weight query.
 */
export interface WeightsResponse {
  ticker: string;
  horizon: number;
  weights: WeightMatrix;
  calibrated_at?: string;
  is_default: boolean;
}

/**
 * Strategy-specific weights data.
 */
export interface StrategyWeightsData {
  weights: WeightMatrix;
  is_default: boolean;
  sqn: number | null;
  updated_at: string | null;
}

/**
 * Batch weights response.
 * When multiple strategy classes are requested, each ticker includes a 'strategies' object.
 */
export interface BatchWeightsResponse {
  horizon: number;
  strategy_class?: string; // 'all', 'directional', 'premium_sell', 'premium_buy', or comma-separated list
  tickers: Record<string, {
    weights: WeightMatrix;
    is_default: boolean;
    sqn: number | null;
    updated_at: string | null;
    // Present when multiple strategy classes requested
    strategies?: Record<string, StrategyWeightsData>;
  }>;
}

// ============================================================================
// Resonance Types
// ============================================================================

export interface HorizonResonanceResult {
  horizon: number;
  ic: number;
  ic_t_stat: number;
  avg_return: number;
  hit_rate: number;
  signal_coverage: number;
  is_significant: boolean;
}

export interface ResonanceResponse {
  ticker: string;
  horizons_tested: number;
  heatmap: Record<number, HorizonResonanceResult>;
  top_horizons: number[];
  recommended: {
    short: number | null;
    medium: number | null;
    long: number | null;
  };
}

