/**
 * Walk-Forward Optimization Calibration Page
 * 
 * Isolated from the main dashboard to avoid rate limit conflicts.
 * Displays real-time progress and results of weight calibration.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowLeft, 
  Play, 
  CheckCircle, 
  XCircle,
  AlertTriangle,
  TrendingUp,
  Loader2,
  Activity,
  Download,
  Database,
  Eye,
  CheckCircle2,
  Clock,
  Zap,
  Shield
} from 'lucide-react';

// Import proper types and API
import type {
  WeightMatrix,
  HorizonResult,
  OptimizerType,
  ResonanceResponse
} from '../types/calibration';
import * as calibrationApi from '../services/calibrationApi';
import type { DataStatus, CalibrationVerification } from '../services/calibrationApi';
import { refreshAnalysis } from '../hooks/useStockAnalysis';
// Note: SSE streaming removed - was causing duplicate calibration runs
// import { streamCalibrationProgress } from '../services/calibrationApi';
import { trackError } from '../services/errorTracking';
import { ErrorPanel } from '../components/ErrorPanel';

// ============================================================================
// Local Types
// ============================================================================

interface LogEntry {
  timestamp: Date;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
}

interface TickerCalibration {
  ticker: string;
  status: 'idle' | 'running' | 'saving' | 'verifying' | 'complete' | 'error';
  progress: number;
  currentIndicator?: string;
  results: Record<number, HorizonResult>;
  error?: string;
  previousWeights?: Partial<WeightMatrix>;
  dataStatus?: DataStatus;
  fetchingData?: boolean;
  verification?: CalibrationVerification;
  resonance?: ResonanceResponse;
  logs: LogEntry[];
}

// ============================================================================
// Helper Components
// ============================================================================

function WeightBar({
  indicator,
  weight,
  defaultWeight = 1.0,
  stable
}: {
  indicator: string;
  weight: number;
  defaultWeight?: number;
  stable?: boolean;
}) {
  const maxWeight = 2.5;
  const widthPercent = (weight / maxWeight) * 100;
  const defaultWidthPercent = (defaultWeight / maxWeight) * 100;

  const isAboveDefault = weight > defaultWeight;
  const isBelowDefault = weight < defaultWeight;

  return (
    <div className="flex items-center gap-3 group">
      <span className="text-xs text-white/60 w-24 capitalize truncate" title={indicator}>
        {indicator.replace(/_/g, ' ')}
      </span>
      <div className="flex-1 h-4 bg-white/5 rounded overflow-hidden relative">
        {/* Default marker */}
        <div
          className="absolute top-0 bottom-0 w-px bg-white/20"
          style={{ left: `${defaultWidthPercent}%` }}
        />
        {/* Weight bar */}
        <div
          className={`h-full rounded transition-all ${
            isAboveDefault ? 'bg-green-500/70' :
            isBelowDefault ? 'bg-amber-500/70' :
            'bg-blue-500/70'
          }`}
          style={{ width: `${Math.min(widthPercent, 100)}%` }}
        />
      </div>
      <span className={`text-xs font-mono w-10 text-right ${
        isAboveDefault ? 'text-green-400' :
        isBelowDefault ? 'text-amber-400' :
        'text-white/60'
      }`}>
        {weight.toFixed(2)}
      </span>
      {stable !== undefined && (
        <span className={`text-xs ${stable ? 'text-green-400' : 'text-amber-400'}`}>
          {stable ? '✓' : '⚠'}
        </span>
      )}
    </div>
  );
}

function SQNGauge({ sqn, label }: { sqn: number | null; label: string }) {
  if (sqn === null) return null;
  
  const getColor = (val: number) => {
    if (val < 0) return 'text-red-500 bg-red-500/20';
    if (val < 1.6) return 'text-red-400 bg-red-500/10';
    if (val < 2.0) return 'text-amber-400 bg-amber-500/10';
    if (val < 2.5) return 'text-yellow-400 bg-yellow-500/10';
    if (val < 3.0) return 'text-green-400 bg-green-500/10';
    return 'text-cyan-400 bg-cyan-500/10';
  };
  
  const getLabel = (val: number) => {
    if (val < 0) return 'Losing';
    if (val < 1.6) return 'Poor';
    if (val < 2.0) return 'Below Avg';
    if (val < 2.5) return 'Average';
    if (val < 3.0) return 'Good';
    if (val < 5.0) return 'Excellent';
    return 'Superb';
  };
  
  return (
    <div className={`text-center px-3 py-2 rounded-lg ${getColor(sqn)}`}>
      <div className="text-xs opacity-70 mb-1">{label}</div>
      <div className="text-lg font-bold font-mono">
        {sqn.toFixed(2)}
      </div>
      <div className="text-xs opacity-70">{getLabel(sqn)}</div>
    </div>
  );
}

function LogViewer({ logs }: { logs: LogEntry[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);
  
  if (logs.length === 0) {
    return (
      <div className="text-xs text-white/30 italic p-2">
        No activity yet
      </div>
    );
  }
  
  return (
    <div 
      ref={scrollRef}
      className="h-32 overflow-y-auto font-mono text-xs space-y-1 p-2 bg-black/30 rounded"
    >
      {logs.map((log, i) => (
        <div key={i} className={`flex gap-2 ${
          log.type === 'error' ? 'text-red-400' :
          log.type === 'warning' ? 'text-amber-400' :
          log.type === 'success' ? 'text-green-400' :
          'text-white/60'
        }`}>
          <span className="text-white/30">
            {log.timestamp.toLocaleTimeString('en-US', { 
              hour12: false, 
              hour: '2-digit', 
              minute: '2-digit', 
              second: '2-digit' 
            })}
          </span>
          <span>{log.message}</span>
        </div>
      ))}
    </div>
  );
}

function TickerChip({ 
  ticker, 
  selected, 
  hasCalibration,
  onClick 
}: { 
  ticker: string; 
  selected: boolean;
  hasCalibration: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
        selected
          ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-500/50'
          : hasCalibration
          ? 'bg-green-500/10 text-green-300 border border-green-500/30 hover:bg-green-500/20'
          : 'bg-white/5 text-white/70 border border-white/10 hover:bg-white/10'
      }`}
    >
      {ticker}
      {hasCalibration && !selected && (
        <CheckCircle2 className="inline-block w-3 h-3 ml-1 text-green-400" />
      )}
    </button>
  );
}

// Strategy class display names and colors
const STRATEGY_CLASS_CONFIG: Record<string, { label: string; color: string; bgColor: string }> = {
  'directional': { label: 'Directional', color: 'text-blue-400', bgColor: 'bg-blue-500/10 border-blue-500/30' },
  'premium_sell': { label: 'Premium Sell', color: 'text-amber-400', bgColor: 'bg-amber-500/10 border-amber-500/30' },
  'premium_buy': { label: 'Premium Buy', color: 'text-purple-400', bgColor: 'bg-purple-500/10 border-purple-500/30' },
  'all': { label: 'All Actions', color: 'text-white/70', bgColor: 'bg-white/5 border-white/20' },
};

function DatabaseWeightsView({ verification }: { verification: CalibrationVerification }) {
  if (!verification.verified || verification.weights.length === 0) {
    return (
      <div className="text-white/50 text-sm p-4 text-center">
        No calibrated weights in database
      </div>
    );
  }

  // Group by horizon, then by strategy_class
  const byHorizonAndStrategy = verification.weights.reduce((acc, w) => {
    const horizon = w.horizon;
    const strategy = w.strategy_class || 'all';
    if (!acc[horizon]) acc[horizon] = {};
    if (!acc[horizon][strategy]) acc[horizon][strategy] = [];
    acc[horizon][strategy].push(w);
    return acc;
  }, {} as Record<number, Record<string, typeof verification.weights>>);

  return (
    <div className="space-y-6">
      {Object.entries(byHorizonAndStrategy)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([horizon, byStrategy]) => (
        <div key={horizon} className="border border-white/10 rounded-lg overflow-hidden">
          {/* Horizon Header */}
          <div className="bg-white/5 px-3 py-2 border-b border-white/10">
            <div className="text-sm font-medium text-white/70 flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              {horizon === '3' ? 'Swing (3-day)' : `${horizon}-day`} Horizon
            </div>
          </div>

          {/* Strategy Class Sections */}
          <div className="divide-y divide-white/5">
            {Object.entries(byStrategy)
              .sort(([a], [b]) => {
                // Sort order: directional, premium_sell, premium_buy, all
                const order = ['directional', 'premium_sell', 'premium_buy', 'all'];
                return order.indexOf(a) - order.indexOf(b);
              })
              .map(([strategy, weights]) => {
                const config = STRATEGY_CLASS_CONFIG[strategy] || STRATEGY_CLASS_CONFIG['all'];
                // Get SQN from first weight (all weights in same strategy have same SQN)
                const strategySqn = weights[0]?.sqn_score;
                const getSqnColor = (sqn: number | null | undefined) => {
                  if (sqn === null || sqn === undefined) return 'text-white/40';
                  if (sqn < 0) return 'text-red-400';
                  if (sqn < 1.6) return 'text-red-400';
                  if (sqn < 2.0) return 'text-amber-400';
                  if (sqn < 2.5) return 'text-yellow-400';
                  if (sqn < 3.0) return 'text-green-400';
                  return 'text-cyan-400';
                };
                const getSqnLabel = (sqn: number | null | undefined) => {
                  if (sqn === null || sqn === undefined) return '';
                  if (sqn < 0) return 'Losing';
                  if (sqn < 1.6) return 'Poor';
                  if (sqn < 2.0) return 'Below Avg';
                  if (sqn < 2.5) return 'Average';
                  if (sqn < 3.0) return 'Good';
                  if (sqn < 5.0) return 'Excellent';
                  return 'Superb';
                };
                return (
                  <div key={strategy} className="p-3">
                    {/* Strategy Header with SQN */}
                    <div className="flex items-center justify-between mb-2">
                      <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-xs font-medium border ${config.bgColor}`}>
                        <span className={config.color}>{config.label}</span>
                        <span className="text-white/30">({weights.length} indicators)</span>
                      </div>
                      {strategySqn !== null && strategySqn !== undefined && (
                        <div className={`text-xs font-mono ${getSqnColor(strategySqn)}`}>
                          SQN: {strategySqn.toFixed(2)} <span className="text-white/40">({getSqnLabel(strategySqn)})</span>
                        </div>
                      )}
                    </div>

                    {/* Weight Bars */}
                    <div className="space-y-1">
                      {weights.map(w => (
                        <WeightBar
                          key={`${w.indicator}-${strategy}`}
                          indicator={w.indicator}
                          weight={w.weight}
                          stable={w.stability_passed}
                        />
                      ))}
                    </div>
                  </div>
                );
              })}
          </div>

          {/* Timestamp */}
          {Object.values(byStrategy)[0]?.[0]?.updated_at && (
            <div className="px-3 py-2 bg-white/5 border-t border-white/10 text-xs text-white/30 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Updated: {new Date(Object.values(byStrategy)[0][0].updated_at).toLocaleString()}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function CalibrationCard({
  calibration,
  onStart,
  onFetchData,
  onViewWeights,
  defaultWeights
}: {
  calibration: TickerCalibration;
  onStart: (ticker: string, resonance?: ResonanceResponse) => void;
  onFetchData: (ticker: string) => Promise<{ success: boolean; resonance?: ResonanceResponse }>;
  onViewWeights: (ticker: string) => void;
  defaultWeights: Record<string, number>;
}) {
  const { 
    ticker, status, progress, results, error, 
    dataStatus, fetchingData, verification, logs, resonance 
  } = calibration;
  
  const horizonResults = Object.entries(results);
  const hasResults = horizonResults.some(([, r]) => r.weights !== null);
  const hasSufficientData = dataStatus?.has_sufficient_data ?? false;
  const isProcessing = status === 'running' || status === 'saving' || status === 'verifying';
  
  return (
    <div className="glass-card p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-bold text-white flex items-center gap-2">
            {ticker}
            {verification?.verified && (
              <span className="text-green-400" title="Calibration verified in database">
                <Shield className="w-4 h-4" />
              </span>
            )}
          </h3>
          <p className="text-sm text-white/50">
            {status === 'idle' && (hasSufficientData ? 'Ready to calibrate' : 'Need more data')}
            {status === 'running' && 'Calibrating...'}
            {status === 'saving' && 'Saving to database...'}
            {status === 'verifying' && 'Verifying saved weights...'}
            {status === 'complete' && 'Calibration complete'}
            {status === 'error' && 'Calibration failed'}
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          {status === 'running' && (
            <div className="flex items-center gap-2 px-3 py-1 bg-cyan-500/10 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
              <span className="text-sm text-cyan-400 font-mono">{Math.round(progress)}%</span>
            </div>
          )}
          
          {status === 'saving' && (
            <div className="flex items-center gap-2 px-3 py-1 bg-amber-500/10 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
              <span className="text-sm text-amber-400">Saving...</span>
            </div>
          )}

          {status === 'verifying' && (
            <div className="flex items-center gap-2 px-3 py-1 bg-green-500/10 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-green-400" />
              <span className="text-sm text-green-400">Verifying...</span>
            </div>
          )}
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => onViewWeights(ticker)}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all"
              title="View current weights"
            >
              <Eye className="w-4 h-4 text-white/60" />
            </button>
            
            <button
              onClick={() => onFetchData(ticker)}
              disabled={fetchingData || isProcessing}
              className="btn-secondary flex items-center gap-2"
              title="Fetch comprehensive history (5y daily + intraday)"
            >
              {fetchingData ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Download className="w-4 h-4" />
              )}
              Fetch
            </button>
            
            <button
              onClick={() => onStart(ticker, resonance)}
              disabled={!hasSufficientData || isProcessing}
              className={`btn-primary flex items-center gap-2 ${(!hasSufficientData || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''}`}
              title={hasSufficientData ? 'Start calibration' : 'Fetch more data first'}
            >
              <Play className="w-4 h-4" />
              {status === 'complete' || status === 'error' ? 'Re-Calibrate' : 'Calibrate'}
            </button>
          </div>
          
          {status === 'complete' && !isProcessing && (
            <div className="flex items-center gap-2">
              {verification?.verified ? (
                <div className="flex items-center gap-1 text-green-400">
                  <CheckCircle className="w-6 h-6" />
                  <span className="text-sm">Verified</span>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-amber-400">
                  <AlertTriangle className="w-6 h-6" />
                  <span className="text-sm">Not Verified</span>
                </div>
              )}
            </div>
          )}
          
          {status === 'error' && (
            <XCircle className="w-8 h-8 text-red-400" />
          )}
        </div>
      </div>
      
      {/* Data Status */}
      {dataStatus && (
        <div className={`p-3 rounded-lg mb-4 ${
          dataStatus.has_sufficient_data 
            ? 'bg-green-500/10 border border-green-500/30' 
            : 'bg-amber-500/10 border border-amber-500/30'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className={`w-4 h-4 ${dataStatus.has_sufficient_data ? 'text-green-400' : 'text-amber-400'}`} />
              <span className={`text-sm ${dataStatus.has_sufficient_data ? 'text-green-400' : 'text-amber-400'}`}>
                {dataStatus.available_days} / {dataStatus.required_days} days
              </span>
            </div>
            {dataStatus.earliest_date && dataStatus.latest_date && (
              <span className="text-xs text-white/40">
                {dataStatus.earliest_date} → {dataStatus.latest_date}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Resonance Analysis */}
      {resonance && (
        <div className="mb-4 p-3 bg-purple-500/10 border border-purple-500/30 rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-sm text-purple-300 font-medium">
              <Activity className="w-4 h-4" />
              Resonant Horizons
              <span className="text-xs font-normal text-purple-400/60 ml-1">
                (Highest Confidence Selected)
              </span>
            </div>
            {resonance.recommended && (
              <div className="text-xs text-purple-400/50" title="Best horizon within each timeframe bucket">
                Buckets: {
                  [resonance.recommended.short, resonance.recommended.medium, resonance.recommended.long]
                  .filter(Boolean)
                  .map(h => `${h}d`)
                  .join(', ')
                }
              </div>
            )}
          </div>
          
          {resonance.top_horizons && resonance.top_horizons.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {resonance.top_horizons.map(horizon => (
                <div key={horizon} className="flex flex-col items-center justify-center bg-purple-500/20 border border-purple-500/30 rounded-lg p-2 min-w-[80px]">
                  <div className="text-xl font-bold text-white font-mono">{horizon}d</div>
                  <div className="text-[10px] text-purple-300/70 uppercase tracking-wider">Top Signal</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4">
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            <span className="text-sm">{error}</span>
          </div>
        </div>
      )}
      
      {/* Real-time Log */}
      {(status === 'running' || status === 'saving' || status === 'verifying' || logs.length > 0) && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-white/60">
            <Activity className="w-4 h-4" />
            Activity Log
          </div>
          <LogViewer logs={logs} />
        </div>
      )}
      
      {/* Verification Info */}
      {status === 'complete' && verification && (
        <div className="mb-4 p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-2 mb-2 text-sm text-white/60">
            <Shield className="w-4 h-4" />
            Database Verification
          </div>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-xl font-bold text-white">{verification.weights_count}</div>
              <div className="text-xs text-white/50">Weights</div>
            </div>
            <div>
              <div className="text-xl font-bold text-white">{verification.windows_count}</div>
              <div className="text-xs text-white/50">Windows</div>
            </div>
            <div>
              <div className="text-xl font-bold text-white">{verification.trades_count}</div>
              <div className="text-xs text-white/50">Trades</div>
            </div>
          </div>
        </div>
      )}
      
      {/* Results */}
      {hasResults && (
        <div className="space-y-6">
          {horizonResults.map(([horizon, result]) => (
            result.weights && (
              <div key={horizon} className="border-t border-white/10 pt-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-white/70 flex items-center gap-2">
                    <Zap className="w-4 h-4" />
                    {horizon}d Horizon
                  </h4>
                  <div className="flex items-center gap-3">
                    <SQNGauge sqn={result.gross_sqn ?? null} label="Gross SQN" />
                    <SQNGauge sqn={result.sqn} label="Net SQN" />
                    <div className="text-center px-3 py-2 bg-white/5 rounded-lg">
                      <div className="text-xs text-white/50">Trades</div>
                      <div className="text-lg font-bold text-white">
                        {result.trades}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Strategy Status Messages */}
                {/* Good signal quality but costs eating edge */}
                {result.gross_sqn !== undefined && result.gross_sqn !== null && result.gross_sqn >= 2.0 &&
                 result.sqn !== null && result.sqn < 1.0 && (
                  <div className="mb-3 p-2 bg-amber-500/10 border border-amber-500/30 rounded text-xs text-amber-400 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                    <div>
                      <strong>Good signals (Gross SQN {result.gross_sqn.toFixed(2)}) but transaction costs reduce edge.</strong>
                      <br />
                      Consider longer holding periods or stocks with higher per-trade returns.
                    </div>
                  </div>
                )}

                {/* Negative net SQN with poor gross */}
                {result.sqn !== null && result.sqn < 0 && (result.gross_sqn === undefined || result.gross_sqn === null || result.gross_sqn < 2.0) && (
                  <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                    <div>
                      <strong>Negative SQN ({result.sqn.toFixed(2)}) - strategy loses money historically.</strong>
                      <br />
                      This stock may not follow technical patterns well, or the tested horizons don't match its price behavior.
                    </div>
                  </div>
                )}

                {/* Fully profitable */}
                {result.sqn !== null && result.sqn >= 2.0 && (
                  <div className="mb-3 p-2 bg-green-500/10 border border-green-500/30 rounded text-xs text-green-400 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                    Strategy shows positive edge after costs. Weights saved to database.
                  </div>
                )}
                
                {/* Weight Bars */}
                <div className="space-y-2">
                  {Object.entries(result.weights).map(([indicator, weight]) => (
                    <WeightBar 
                      key={indicator}
                      indicator={indicator}
                      weight={weight as number}
                      defaultWeight={defaultWeights[indicator] || 1.0}
                    />
                  ))}
                </div>
              </div>
            )
          ))}
        </div>
      )}
      
      {/* Show stored weights if viewing */}
      {status === 'idle' && verification?.verified && (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-2 text-sm text-white/60">
            <Database className="w-4 h-4" />
            Current Database Weights
          </div>
          <DatabaseWeightsView verification={verification} />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

// Optimizer options with descriptions
const OPTIMIZER_OPTIONS: { value: OptimizerType; label: string; description: string }[] = [
  { value: 'coordinate_descent', label: 'Coordinate Descent', description: 'Fast, default optimizer' },
  { value: 'differential_evolution', label: 'Differential Evolution', description: 'Global optimizer (slower, may find better weights)' },
  { value: 'hybrid', label: 'Hybrid', description: 'DE + CD refinement (slowest, most thorough)' },
];

export default function CalibrationPage() {
  const [calibrations, setCalibrations] = useState<Record<string, TickerCalibration>>({});
  const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [defaultWeights, setDefaultWeights] = useState<Record<string, number>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [calibratedTickers, setCalibratedTickers] = useState<Set<string>>(new Set());
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  const [isFetchingAll, setIsFetchingAll] = useState(false);
  const [fetchAllProgress, setFetchAllProgress] = useState<{ current: number; total: number; ticker: string } | null>(null);
  const [selectedOptimizer, setSelectedOptimizer] = useState<OptimizerType>('coordinate_descent');


  // Batch calibration progress tracking
  const [batchProgress, setBatchProgress] = useState<{
    phase: 'idle' | 'fetching' | 'calibrating' | 'complete';
    fetchProgress: { current: number; total: number; ticker: string };
    calibrateProgress: { current: number; total: number; activeTickers: string[] };
    results: { success: string[]; failed: string[]; skipped: string[] };
  } | null>(null);
  
  // Add log entry helper
  const addLog = useCallback((ticker: string, type: LogEntry['type'], message: string) => {
    setCalibrations(prev => ({
      ...prev,
      [ticker]: {
        ...prev[ticker],
        logs: [...(prev[ticker]?.logs || []), { timestamp: new Date(), type, message }]
      }
    }));
  }, []);

  // Initialize calibration state for a ticker if needed
  const ensureTickerState = useCallback((ticker: string) => {
    setCalibrations(prev => {
      if (prev[ticker]) return prev;
      return {
        ...prev,
        [ticker]: {
          ticker,
          status: 'idle',
          progress: 0,
          results: {},
          logs: []
        }
      };
    });
  }, []);
  
  // Load portfolio tickers and default weights on mount
  useEffect(() => {
    Promise.all([
      calibrationApi.getPortfolioTickers(),
      calibrationApi.getDefaultWeights()
    ])
      .then(([tickers, defaults]) => {
        setPortfolioTickers(tickers);
        setDefaultWeights((defaults.weights || {}) as unknown as Record<string, number>);

        // Check which tickers have calibration
        // Errors are automatically tracked via calibrationApi
        tickers.forEach(async (ticker) => {
          try {
            const verification = await calibrationApi.verifyCalibration(ticker);
            if (verification.verified) {
              setCalibratedTickers(prev => new Set([...prev, ticker]));
            }
          } catch {
            // Error already tracked by calibrationApi.verifyCalibration
          }
        });

        setIsLoading(false);
      })
      .catch(err => {
        // Error already tracked, but we need to stop loading
        trackError(err, {
          operation: 'initializeCalibrationPage',
          source: 'CalibrationPage',
          metadata: { phase: 'loadPortfolio' }
        });
        setIsLoading(false);
      });
  }, []);
  
  // Handle ticker selection
  const handleSelectTicker = useCallback(async (ticker: string) => {
    setSelectedTicker(ticker);
    ensureTickerState(ticker);

    // Fetch data status, verification, and resonance
    try {
        const [dataStatus, verification] = await Promise.all([
          calibrationApi.getDataStatus(ticker),
          calibrationApi.verifyCalibration(ticker)
        ]);

        // If we have data, try to fetch resonance
        let resonance: ResonanceResponse | undefined;
        if (dataStatus.has_sufficient_data) {
           try {
             resonance = await calibrationApi.getResonance(ticker);
           } catch (e) {
             // Surface resonance fetch failure to user (will use default horizons)
             const errMsg = e instanceof Error ? e.message : 'Unknown error';
             addLog(ticker, 'warning', `Could not fetch resonant horizons: ${errMsg}. Will use defaults.`);
           }
        }

        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            dataStatus,
            verification,
            resonance
          }
        }));
    } catch (err) {
        const errMsg = err instanceof Error ? err.message : 'Unknown error';
        addLog(ticker, 'error', `Failed to load ticker status: ${errMsg}`);
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            error: errMsg
          }
        }));
    }
  }, [ensureTickerState, addLog]);

  // View weights for a ticker
  const handleViewWeights = useCallback(async (ticker: string) => {
    try {
      const verification = await calibrationApi.verifyCalibration(ticker);
      setCalibrations(prev => ({
        ...prev,
        [ticker]: {
          ...prev[ticker],
          verification
        }
      }));
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error';
      addLog(ticker, 'error', `Failed to load weights: ${errMsg}`);
    }
  }, [addLog]);

  // Fetch data
  const handleFetchData = useCallback(async (ticker: string) => {
    setCalibrations(prev => ({
      ...prev,
      [ticker]: {
        ...prev[ticker],
        fetchingData: true,
        error: undefined
      }
    }));
    
    addLog(ticker, 'info', 'Fetching comprehensive price history (5y daily + intraday)...');
    
    try {
      const result = await calibrationApi.fetchCalibrationData(ticker);
      
      if (result.status === 'success') {
        addLog(ticker, 'success', `Fetched ${result.days_fetched} days of daily data + intraday history`);
        
        const status = await calibrationApi.getDataStatus(ticker);
        
        // Fetch resonance now that we have data
        let resonance: ResonanceResponse | undefined;
        try {
           resonance = await calibrationApi.getResonance(ticker);
           if (resonance?.top_horizons?.length) {
             addLog(ticker, 'info', `Discovered resonant horizons: ${resonance.top_horizons.join(', ')}d`);
           }
        } catch (e) {
           console.warn('Failed to fetch resonance:', e);
        }

        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            fetchingData: false,
            dataStatus: status,
            resonance
          }
        }));
        // Return resonance directly to avoid stale closure issue
        return { success: true, resonance };
      } else {
        addLog(ticker, 'error', result.message);
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            fetchingData: false,
            error: result.message
          }
        }));
        return { success: false, resonance: undefined };
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch data';
      addLog(ticker, 'error', msg);
      setCalibrations(prev => ({
        ...prev,
        [ticker]: {
          ...prev[ticker],
          fetchingData: false,
          error: msg
        }
      }));
      return { success: false, resonance: undefined };
    }
  }, [addLog]);

  // Fetch all data (for all holdings)
  const handleFetchAll = useCallback(async () => {
    if (portfolioTickers.length === 0) return;

    if (!window.confirm(`This will fetch price history for all ${portfolioTickers.length} holdings. This may take a while. Continue?`)) {
      return;
    }

    setIsFetchingAll(true);
    const total = portfolioTickers.length;

    for (let i = 0; i < portfolioTickers.length; i++) {
      const ticker = portfolioTickers[i];
      setFetchAllProgress({ current: i + 1, total, ticker });
      ensureTickerState(ticker);
      addLog(ticker, 'info', 'Starting data fetch...');
      await handleFetchData(ticker);
      // Small delay between fetches to avoid rate limiting
      await new Promise(r => setTimeout(r, 500));
    }

    setIsFetchingAll(false);
    setFetchAllProgress(null);
    alert('Finished fetching data for all holdings!');
  }, [portfolioTickers, ensureTickerState, handleFetchData, addLog]);

  // Start calibration
  // passedResonance: Optional resonance data passed directly (avoids stale closure)
  const handleStartCalibration = useCallback(async (ticker: string, passedResonance?: ResonanceResponse) => {
    // 1. Determine horizons (Prioritize Top Confidence/Resonance)
    // Use passed resonance first (avoids React stale closure), then try state, then fetch
    let currentResonance = passedResonance || calibrations[ticker]?.resonance;
    let horizons: number[] = [];
    let horizonSource = 'fallback';

    // Debug: Log what we have
    console.log('[Calibration] Resonance for', ticker, ':', {
      passedResonance: !!passedResonance,
      stateResonance: !!calibrations[ticker]?.resonance,
      usingResonance: !!currentResonance,
      top_horizons: currentResonance?.top_horizons,
      recommended: currentResonance?.recommended
    });

    // If no resonance available but we have data, fetch it now
    if (!currentResonance && calibrations[ticker]?.dataStatus?.has_sufficient_data) {
      console.log('[Calibration] Resonance missing, fetching fresh for', ticker);
      addLog(ticker, 'info', 'Fetching resonance data...');
      try {
        currentResonance = await calibrationApi.getResonance(ticker);
        console.log('[Calibration] Fresh resonance fetched:', currentResonance);
        // Update state with fetched resonance
        setCalibrations(prev => ({
          ...prev,
          [ticker]: { ...prev[ticker], resonance: currentResonance }
        }));
      } catch (e) {
        console.warn('[Calibration] Failed to fetch resonance:', e);
        addLog(ticker, 'warning', `Could not fetch resonance: ${e}`);
      }
    }

    console.log('[Calibration] Starting for', ticker, 'using resonance:', currentResonance);

    if (currentResonance?.top_horizons && currentResonance.top_horizons.length > 0) {
      horizons = [...currentResonance.top_horizons]; // Copy to avoid mutation
      horizonSource = 'resonance.top_horizons';
      console.log('[Calibration] Using top_horizons:', horizons);
    } else if (currentResonance?.recommended) {
      // Fallback to buckets if no top horizons found (rare)
      const { short, medium, long } = currentResonance.recommended;
      console.log('[Calibration] top_horizons empty/missing, trying recommended:', { short, medium, long });
      if (short) horizons.push(short);
      if (medium) horizons.push(medium);
      if (long) horizons.push(long);
      horizonSource = 'resonance.recommended';
    } else {
      console.log('[Calibration] No resonance data available, will use defaults');
    }

    // Deduplicate and sort
    horizons = [...new Set(horizons)].sort((a, b) => a - b);

    // Final fallback
    if (horizons.length === 0) {
      horizons = [3, 15];
      horizonSource = 'default fallback (no resonance)';
      addLog(ticker, 'warning', 'No resonant horizons found, using defaults [3, 15]');
    }

    console.log('[Calibration] Final horizons:', horizons, 'source:', horizonSource);

    setCalibrations(prev => ({
      ...prev,
      [ticker]: {
        ...prev[ticker],
        status: 'running',
        progress: 10, // Show some initial progress
        error: undefined,
        logs: [],
        results: {} // Clear previous results
      }
    }));

    addLog(ticker, 'info', `Starting WFO Calibration (Horizons: ${horizons.join(', ')}d from ${horizonSource})`);
    addLog(ticker, 'info', 'Calibration running... (this may take 2-5 minutes per ticker)');

    try {
      // Run calibration via POST - this does the full optimization and saves results
      const result = await calibrationApi.startCalibration({
        ticker,
        horizons,
        optimizer: selectedOptimizer
      });
      
      if (result.status === 'complete' && result.results) {
        // Results are now nested: { strategy_class: { horizon: HorizonResult } }
        // Flatten to get all horizon results for logging
        const flatResults: Array<{ strategy: string; horizon: string; hr: HorizonResult }> = [];

        for (const [strategy, horizonMap] of Object.entries(result.results)) {
          if (typeof horizonMap === 'object' && horizonMap !== null) {
            for (const [horizon, hr] of Object.entries(horizonMap as Record<string, HorizonResult>)) {
              flatResults.push({ strategy, horizon, hr });
            }
          }
        }

        const successfulResults = flatResults.filter(r => r.hr.sqn !== null && r.hr.sqn !== undefined);
        const failedResults = flatResults.filter(r => r.hr.sqn === null || r.hr.sqn === undefined);

        if (successfulResults.length === flatResults.length && flatResults.length > 0) {
          addLog(ticker, 'success', 'Optimization complete');
        } else if (successfulResults.length > 0) {
          addLog(ticker, 'warning', `Optimization partially complete (${successfulResults.length}/${flatResults.length} strategy+horizon combos)`);
        } else {
          addLog(ticker, 'warning', 'Optimization failed for all horizons');
        }

        // Surface save warnings from backend
        const saveWarnings = (result as { save_warnings?: string[] }).save_warnings;
        if (saveWarnings && saveWarnings.length > 0) {
          for (const warning of saveWarnings) {
            addLog(ticker, 'warning', `Database: ${warning}`);
          }
        }

        // Check for individual save errors in results
        for (const { strategy, horizon, hr } of flatResults) {
          const saveError = (hr as { save_error?: string }).save_error;
          if (saveError) {
            addLog(ticker, 'error', `Failed to save ${strategy} ${horizon}d: ${saveError}`);
          }
        }

        // Log result details grouped by strategy
        const byStrategy = new Map<string, typeof flatResults>();
        for (const r of successfulResults) {
          if (!byStrategy.has(r.strategy)) byStrategy.set(r.strategy, []);
          byStrategy.get(r.strategy)!.push(r);
        }

        for (const [strategy, results] of byStrategy) {
          for (const { horizon, hr } of results) {
            const confidenceNote = hr.reduced_confidence ? ' (reduced confidence)' : '';
            const overfitNote = hr.overfit_warning ? ' [OVERFIT WARNING]' : '';
            const sqnStr = hr.sqn != null ? hr.sqn.toFixed(2) : 'N/A';
            addLog(ticker, hr.overfit_warning ? 'warning' : 'info',
              `${strategy} ${horizon}d: SQN=${sqnStr}, Trades=${hr.trades}${confidenceNote}${overfitNote}`);
          }
        }

        for (const { strategy, horizon, hr } of failedResults) {
          addLog(ticker, 'warning', `${strategy} ${horizon}d failed: ${hr.error || 'Unknown error'}`);
        }

        // Update to verifying state
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            status: 'verifying',
            progress: 100
          }
        }));

        addLog(ticker, 'info', 'Verifying database writes...');

        // Verify the calibration was saved (with timeout for user feedback)
        let verification: calibrationApi.CalibrationVerification;
        try {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Verification timed out')), 10000)
          );
          verification = await Promise.race([
            calibrationApi.verifyCalibration(ticker),
            timeoutPromise
          ]) as calibrationApi.CalibrationVerification;
        } catch (verifyErr) {
          addLog(ticker, 'warning', `Verification slow: ${verifyErr instanceof Error ? verifyErr.message : 'unknown'}`);
          // Try once more without timeout
          verification = await calibrationApi.verifyCalibration(ticker);
        }

        if (verification.verified) {
          addLog(ticker, 'success', `Verified: ${verification.weights_count} weights saved to database`);
          setCalibratedTickers(prev => new Set([...prev, ticker]));

          // Force refresh analysis cache
          refreshAnalysis(ticker, 0, null, null);
        } else {
           addLog(ticker, 'warning', 'Calibration completed but no weights found in database');
        }
        
        // Flatten nested results {strategy: {horizon: result}} to {horizon: result}
        // Prefer 'directional' strategy for display, fallback to first available
        const flattenedResults: Record<number, HorizonResult> = {};
        const strategyPriority = ['directional', 'premium_sell', 'premium_buy'];

        for (const strategy of strategyPriority) {
          const strategyResults = result.results![strategy];
          if (strategyResults && typeof strategyResults === 'object') {
            for (const [horizon, hr] of Object.entries(strategyResults as Record<string, HorizonResult>)) {
              const horizonNum = parseInt(horizon);
              if (!flattenedResults[horizonNum]) {
                flattenedResults[horizonNum] = hr;
              }
            }
          }
        }

        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            status: 'complete',
            progress: 100,
            results: flattenedResults,
            verification
          }
        }));
        return true;
        
      } else {
        // Handle explicit error
        const errorMsg = result.error || 'Unknown error';
        addLog(ticker, 'error', errorMsg);
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            status: 'error',
            error: errorMsg
          }
        }));
        return false;
      }
    } catch (err) {
      cleanup();
      const msg = err instanceof Error ? err.message : 'Unknown error';
      addLog(ticker, 'error', msg);
      setCalibrations(prev => ({
        ...prev,
        [ticker]: {
          ...prev[ticker],
          status: 'error',
          error: msg
        }
      }));
      return false;
    }
  }, [addLog, calibrations, selectedOptimizer]);

  // Batch Calibration - Fetch sequentially (rate limiting), calibrate in parallel
  const handleBatchCalibration = useCallback(async () => {
    if (portfolioTickers.length === 0) return;

    if (!window.confirm(`This will fetch data sequentially then calibrate all ${portfolioTickers.length} holdings in parallel. Continue?`)) {
        return;
    }

    setIsBatchRunning(true);
    const results = { success: [] as string[], failed: [] as string[], skipped: [] as string[] };

    // Initialize batch progress
    setBatchProgress({
      phase: 'fetching',
      fetchProgress: { current: 0, total: portfolioTickers.length, ticker: '' },
      calibrateProgress: { current: 0, total: 0, activeTickers: [] },
      results
    });

    // Phase 1: Fetch data sequentially (Yahoo Finance rate limiting)
    const tickersWithData: Array<{ ticker: string; resonance?: ResonanceResponse }> = [];

    for (let i = 0; i < portfolioTickers.length; i++) {
        const ticker = portfolioTickers[i];

        setBatchProgress(prev => prev ? {
          ...prev,
          fetchProgress: { current: i + 1, total: portfolioTickers.length, ticker }
        } : null);

        setSelectedTicker(ticker);
        ensureTickerState(ticker);
        addLog(ticker, 'info', `Fetching data (${i + 1}/${portfolioTickers.length})...`);

        const fetchResult = await handleFetchData(ticker);

        if (fetchResult.success) {
            tickersWithData.push({ ticker, resonance: fetchResult.resonance });
        } else {
            results.skipped.push(ticker);
            addLog(ticker, 'warning', 'Skipped - insufficient data');
        }

        // Small delay between fetches for rate limiting
        await new Promise(r => setTimeout(r, 500));
    }

    if (tickersWithData.length === 0) {
        setBatchProgress(prev => prev ? { ...prev, phase: 'complete', results } : null);
        setIsBatchRunning(false);
        return;
    }

    // Phase 2: Calibrate in parallel (thread-safe with atomic upserts)
    const PARALLEL_LIMIT = 4;
    let completedCount = 0;

    setBatchProgress(prev => prev ? {
      ...prev,
      phase: 'calibrating',
      calibrateProgress: { current: 0, total: tickersWithData.length, activeTickers: [] }
    } : null);

    // Process in batches of PARALLEL_LIMIT
    for (let i = 0; i < tickersWithData.length; i += PARALLEL_LIMIT) {
        const batch = tickersWithData.slice(i, i + PARALLEL_LIMIT);
        const batchTickers = batch.map(b => b.ticker);

        // Update progress with active tickers
        setBatchProgress(prev => prev ? {
          ...prev,
          calibrateProgress: {
            current: completedCount,
            total: tickersWithData.length,
            activeTickers: batchTickers
          }
        } : null);

        // Start all calibrations in this batch simultaneously
        const batchPromises = batch.map(async ({ ticker, resonance }) => {
            addLog(ticker, 'info', `Calibrating in parallel batch ${Math.floor(i / PARALLEL_LIMIT) + 1}...`);
            const success = await handleStartCalibration(ticker, resonance);
            return { ticker, success };
        });

        // Wait for entire batch to complete
        const batchResults = await Promise.all(batchPromises);

        // Update results
        for (const { ticker, success } of batchResults) {
            if (success) {
                results.success.push(ticker);
            } else {
                results.failed.push(ticker);
            }
            completedCount++;
        }

        // Update progress after batch
        setBatchProgress(prev => prev ? {
          ...prev,
          calibrateProgress: {
            current: completedCount,
            total: tickersWithData.length,
            activeTickers: []
          },
          results: { ...results }
        } : null);
    }

    // Final state
    setBatchProgress(prev => prev ? { ...prev, phase: 'complete', results: { ...results } } : null);
    setIsBatchRunning(false);
  }, [portfolioTickers, ensureTickerState, handleFetchData, handleStartCalibration, addLog]);
  
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="flex items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin text-cyan-400" />
          <span className="text-white/70">Loading calibration system...</span>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link 
              to="/"
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all"
            >
              <ArrowLeft className="w-5 h-5 text-white/60" />
            </Link>
            <div>
              <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                <TrendingUp className="w-6 h-6 text-cyan-400" />
                Walk-Forward Optimization
              </h1>
              <p className="text-sm text-white/50">
                Calibrate indicator weights using historical backtesting
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            {/* Optimizer Selector */}
            <div className="flex flex-col">
              <label className="text-xs text-white/40 mb-1">Optimizer</label>
              <select
                value={selectedOptimizer}
                onChange={(e) => setSelectedOptimizer(e.target.value as OptimizerType)}
                disabled={isBatchRunning}
                className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white/80 focus:outline-none focus:border-cyan-500/50"
              >
                {OPTIMIZER_OPTIONS.map(opt => (
                  <option key={opt.value} value={opt.value} className="bg-slate-800">
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleFetchAll}
              disabled={isFetchingAll || isBatchRunning || portfolioTickers.length === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isFetchingAll
                      ? 'bg-purple-500/20 text-purple-400 cursor-not-allowed'
                      : 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 border border-purple-500/30'
              }`}
            >
              {isFetchingAll && fetchAllProgress ? (
                  <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="font-mono">{fetchAllProgress.ticker}</span>
                      <span className="text-purple-400/60">({fetchAllProgress.current}/{fetchAllProgress.total})</span>
                  </>
              ) : (
                  <>
                      <Download className="w-4 h-4" />
                      Fetch All Data
                  </>
              )}
            </button>

            <button
              onClick={handleBatchCalibration}
              disabled={isBatchRunning || isFetchingAll || portfolioTickers.length === 0}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                  isBatchRunning
                      ? 'bg-cyan-500/20 text-cyan-400 cursor-not-allowed'
                      : 'bg-cyan-500 hover:bg-cyan-600 text-white shadow-lg shadow-cyan-500/20'
              }`}
            >
              {isBatchRunning ? (
                  <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Auto-Calibrating...
                  </>
              ) : (
                  <>
                      <Play className="w-4 h-4" />
                      Auto-Calibrate All
                  </>
              )}
            </button>
          </div>
        </div>

        {/* Comprehensive Error Panel */}
        <ErrorPanel maxHeight="300px" />

        {/* Batch Progress Panel */}
        {batchProgress && (
          <div className="glass-card p-4 mb-6 border-l-4 border-l-cyan-500">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                {batchProgress.phase !== 'complete' && (
                  <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                )}
                {batchProgress.phase === 'complete' && (
                  <CheckCircle className="w-5 h-5 text-green-400" />
                )}
                <div>
                  <div className="font-medium text-white">
                    {batchProgress.phase === 'fetching' && 'Phase 1: Fetching Data'}
                    {batchProgress.phase === 'calibrating' && 'Phase 2: Parallel Calibration'}
                    {batchProgress.phase === 'complete' && 'Batch Calibration Complete'}
                  </div>
                  <div className="text-sm text-white/50">
                    {batchProgress.phase === 'fetching' && (
                      <>Fetching {batchProgress.fetchProgress.current}/{batchProgress.fetchProgress.total} - <span className="text-cyan-400 font-mono">{batchProgress.fetchProgress.ticker}</span></>
                    )}
                    {batchProgress.phase === 'calibrating' && (
                      <>Calibrated {batchProgress.calibrateProgress.current}/{batchProgress.calibrateProgress.total} (4 concurrent)</>
                    )}
                    {batchProgress.phase === 'complete' && (
                      <>{batchProgress.results.success.length} success, {batchProgress.results.failed.length} failed, {batchProgress.results.skipped.length} skipped</>
                    )}
                  </div>
                </div>
              </div>

              {batchProgress.phase === 'complete' && (
                <button
                  onClick={() => setBatchProgress(null)}
                  className="text-xs text-white/40 hover:text-white/60 px-2 py-1"
                >
                  Dismiss
                </button>
              )}
            </div>

            {/* Progress Bar */}
            {batchProgress.phase !== 'complete' && (
              <div className="mb-4">
                <div className="h-2 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      batchProgress.phase === 'fetching' ? 'bg-purple-500' : 'bg-cyan-500'
                    }`}
                    style={{
                      width: `${
                        batchProgress.phase === 'fetching'
                          ? (batchProgress.fetchProgress.current / batchProgress.fetchProgress.total) * 100
                          : (batchProgress.calibrateProgress.current / batchProgress.calibrateProgress.total) * 100
                      }%`
                    }}
                  />
                </div>
              </div>
            )}

            {/* Active Tickers (during calibration) */}
            {batchProgress.phase === 'calibrating' && batchProgress.calibrateProgress.activeTickers.length > 0 && (
              <div className="flex items-center gap-2 mb-4">
                <span className="text-xs text-white/40">Currently calibrating:</span>
                <div className="flex gap-1">
                  {batchProgress.calibrateProgress.activeTickers.map(ticker => (
                    <span
                      key={ticker}
                      className="px-2 py-0.5 bg-cyan-500/20 text-cyan-300 text-xs rounded font-mono animate-pulse"
                    >
                      {ticker}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Results Summary (when complete) */}
            {batchProgress.phase === 'complete' && (
              <div className="grid grid-cols-3 gap-4">
                {batchProgress.results.success.length > 0 && (
                  <div className="p-3 bg-green-500/10 rounded-lg border border-green-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-sm font-medium text-green-400">Success ({batchProgress.results.success.length})</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {batchProgress.results.success.map(t => (
                        <span key={t} className="text-xs text-green-300 font-mono">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
                {batchProgress.results.failed.length > 0 && (
                  <div className="p-3 bg-red-500/10 rounded-lg border border-red-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <XCircle className="w-4 h-4 text-red-400" />
                      <span className="text-sm font-medium text-red-400">Failed ({batchProgress.results.failed.length})</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {batchProgress.results.failed.map(t => (
                        <span key={t} className="text-xs text-red-300 font-mono">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
                {batchProgress.results.skipped.length > 0 && (
                  <div className="p-3 bg-amber-500/10 rounded-lg border border-amber-500/30">
                    <div className="flex items-center gap-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-amber-400" />
                      <span className="text-sm font-medium text-amber-400">Skipped ({batchProgress.results.skipped.length})</span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {batchProgress.results.skipped.map(t => (
                        <span key={t} className="text-xs text-amber-300 font-mono">{t}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Holdings Picker */}
        <div className="glass-card p-4 mb-6">
          <div className="flex items-center gap-2 mb-3 text-sm text-white/60">
            <Database className="w-4 h-4" />
            Select from Portfolio
            <span className="text-xs text-white/30">
              ({portfolioTickers.length} holdings • {calibratedTickers.size} calibrated)
            </span>
          </div>
          
          {portfolioTickers.length === 0 ? (
            <div className="text-white/40 text-sm">
              No holdings in portfolio. Add stocks to your portfolio first.
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {portfolioTickers.map(ticker => (
                <TickerChip
                  key={ticker}
                  ticker={ticker}
                  selected={selectedTicker === ticker}
                  hasCalibration={calibratedTickers.has(ticker)}
                  onClick={() => !isBatchRunning && handleSelectTicker(ticker)}
                />
              ))}
            </div>
          )}
        </div>
        
        {/* Selected Ticker Calibration */}
        {selectedTicker && calibrations[selectedTicker] && (
          <CalibrationCard
            calibration={calibrations[selectedTicker]}
            onStart={handleStartCalibration}
            onFetchData={handleFetchData}
            onViewWeights={handleViewWeights}
            defaultWeights={defaultWeights}
          />
        )}
        
        {/* Empty State */}
        {!selectedTicker && (
          <div className="glass-card p-12 text-center">
            <TrendingUp className="w-12 h-12 text-white/20 mx-auto mb-4" />
            <p className="text-white/50">
              Select a ticker from your portfolio above to begin calibration
            </p>
          </div>
        )}
        
        {/* Info Footer */}
        <div className="mt-8 p-4 bg-white/5 rounded-lg text-xs text-white/40">
          <div className="font-medium text-white/60 mb-2">About WFO Calibration</div>
          <ul className="list-disc list-inside space-y-1">
            <li>Uses <strong>6-month rolling windows</strong> with 1-month roll step</li>
            <li>Requires minimum <strong>30 trades</strong> for statistical significance</li>
            <li>Optimizes using <strong>{OPTIMIZER_OPTIONS.find(o => o.value === selectedOptimizer)?.label}</strong>
              {selectedOptimizer === 'differential_evolution' && ' (global search, 5-10x slower)'}
              {selectedOptimizer === 'hybrid' && ' (DE + local refinement, slowest)'}
            </li>
            <li>Validates weight <strong>stability</strong> to prevent overfitting</li>
            <li>Applies <strong>0.1% transaction cost</strong> for realistic simulation</li>
            <li>Supports <strong>parallel calibration</strong> (4 concurrent) with atomic database writes</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
