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
} from '../types/calibration';
import * as calibrationApi from '../services/calibrationApi';
import type { DataStatus, CalibrationVerification } from '../services/calibrationApi';
import { refreshAnalysis } from '../hooks/useStockAnalysis';

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
  status: 'idle' | 'running' | 'verifying' | 'complete' | 'error';
  progress: number;
  currentIndicator?: string;
  results: Record<number, HorizonResult>;
  error?: string;
  previousWeights?: Partial<WeightMatrix>;
  dataStatus?: DataStatus;
  fetchingData?: boolean;
  verification?: CalibrationVerification;
  logs: LogEntry[];
}

// ============================================================================
// Helper Components
// ============================================================================

function WeightBar({ 
  indicator, 
  weight, 
  defaultWeight = 1.0,
  sqnScore,
  stable
}: { 
  indicator: string; 
  weight: number;
  defaultWeight?: number;
  sqnScore?: number | null;
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
      {sqnScore !== undefined && sqnScore !== null && (
        <span className="text-xs text-white/40 w-16 text-right">
          SQN: {sqnScore === -Infinity ? '-∞' : sqnScore.toFixed(1)}
        </span>
      )}
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

function DatabaseWeightsView({ verification }: { verification: CalibrationVerification }) {
  if (!verification.verified || verification.weights.length === 0) {
    return (
      <div className="text-white/50 text-sm p-4 text-center">
        No calibrated weights in database
      </div>
    );
  }
  
  // Group by horizon
  const byHorizon = verification.weights.reduce((acc, w) => {
    if (!acc[w.horizon]) acc[w.horizon] = [];
    acc[w.horizon].push(w);
    return acc;
  }, {} as Record<number, typeof verification.weights>);
  
  return (
    <div className="space-y-4">
      {Object.entries(byHorizon).map(([horizon, weights]) => (
        <div key={horizon} className="border border-white/10 rounded-lg p-3">
          <div className="text-sm font-medium text-white/70 mb-2">
            {horizon === '3' ? 'Swing (3-day)' : `${horizon}-day`} Horizon
          </div>
          <div className="space-y-1">
            {weights.map(w => (
              <WeightBar
                key={w.indicator}
                indicator={w.indicator}
                weight={w.weight}
                sqnScore={w.sqn_score}
                stable={w.stability_passed}
              />
            ))}
          </div>
          {weights[0]?.updated_at && (
            <div className="text-xs text-white/30 mt-2 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              Updated: {new Date(weights[0].updated_at).toLocaleString()}
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
  onStart: (ticker: string) => void;
  onFetchData: (ticker: string) => void;
  onViewWeights: (ticker: string) => void;
  defaultWeights: Record<string, number>;
}) {
  const { 
    ticker, status, progress, results, error, 
    dataStatus, fetchingData, verification, logs 
  } = calibration;
  
  const horizonResults = Object.entries(results);
  const hasResults = horizonResults.some(([, r]) => r.weights !== null);
  const hasSufficientData = dataStatus?.has_sufficient_data ?? false;
  
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
            {status === 'verifying' && 'Verifying database...'}
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
          
          {status === 'verifying' && (
            <div className="flex items-center gap-2 px-3 py-1 bg-amber-500/10 rounded-lg">
              <Loader2 className="w-4 h-4 animate-spin text-amber-400" />
              <span className="text-sm text-amber-400">Verifying...</span>
            </div>
          )}
          
          {status === 'idle' && (
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
                disabled={fetchingData}
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
                onClick={() => onStart(ticker)}
                disabled={!hasSufficientData}
                className={`btn-primary flex items-center gap-2 ${!hasSufficientData ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={hasSufficientData ? 'Start calibration' : 'Fetch more data first'}
              >
                <Play className="w-4 h-4" />
                Calibrate
              </button>
            </div>
          )}
          
          {status === 'complete' && (
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
      {dataStatus && status === 'idle' && (
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
      {(status === 'running' || status === 'verifying' || logs.length > 0) && (
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
                    {horizon === '3' ? 'Swing (3-day)' : 'Trend (15-day)'} Horizon
                  </h4>
                  <div className="flex items-center gap-4">
                    <SQNGauge sqn={result.sqn} label="SQN" />
                    <div className="text-center px-3 py-2 bg-white/5 rounded-lg">
                      <div className="text-xs text-white/50">Trades</div>
                      <div className="text-lg font-bold text-white">
                        {result.trades}
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Strategy Status */}
                {result.sqn !== null && result.sqn < 0 && (
                  <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                    Strategy is unprofitable. Try different indicators or a longer timeframe.
                  </div>
                )}
                
                {result.sqn !== null && result.sqn >= 2.0 && (
                  <div className="mb-3 p-2 bg-green-500/10 border border-green-500/30 rounded text-xs text-green-400 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                    Strategy shows positive edge. Weights saved to database.
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

export default function CalibrationPage() {
  const [calibrations, setCalibrations] = useState<Record<string, TickerCalibration>>({});
  const [portfolioTickers, setPortfolioTickers] = useState<string[]>([]);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);
  const [defaultWeights, setDefaultWeights] = useState<Record<string, number>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [calibratedTickers, setCalibratedTickers] = useState<Set<string>>(new Set());
  const [isBatchRunning, setIsBatchRunning] = useState(false);
  
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
        tickers.forEach(async (ticker) => {
          try {
            const verification = await calibrationApi.verifyCalibration(ticker);
            if (verification.verified) {
              setCalibratedTickers(prev => new Set([...prev, ticker]));
            }
          } catch {
            // Ignore errors
          }
        });
        
        setIsLoading(false);
      })
      .catch(err => {
        console.error('Failed to load:', err);
        setIsLoading(false);
      });
  }, []);
  
  // Handle ticker selection
  const handleSelectTicker = useCallback(async (ticker: string) => {
    setSelectedTicker(ticker);
    ensureTickerState(ticker);
    
    // Fetch data status and verification if not already loaded
    // We check via ref or just blindly fetch for simplicity as it's cheap
    try {
        const [dataStatus, verification] = await Promise.all([
          calibrationApi.getDataStatus(ticker),
          calibrationApi.verifyCalibration(ticker)
        ]);
        
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            dataStatus,
            verification
          }
        }));
    } catch (err) {
        console.error('Failed to get status for', ticker, err);
    }
  }, [ensureTickerState]);
  
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
      console.error('Failed to get weights:', err);
    }
  }, []);
  
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
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            fetchingData: false,
            dataStatus: status
          }
        }));
        return true;
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
        return false;
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
      return false;
    }
  }, [addLog]);
  
  // Start calibration
  const handleStartCalibration = useCallback(async (ticker: string) => {
    setCalibrations(prev => ({
      ...prev,
      [ticker]: {
        ...prev[ticker],
        status: 'running',
        progress: 0,
        error: undefined,
        logs: []
      }
    }));
    
    addLog(ticker, 'info', 'Starting Walk-Forward Optimization...');
    addLog(ticker, 'info', 'Loading price history from database...');
    
    try {
      addLog(ticker, 'info', 'Running two-pass coordinate descent...');
      
      // Simulate progress updates (real implementation would use SSE)
      const progressInterval = setInterval(() => {
        setCalibrations(prev => {
          const current = prev[ticker];
          if (current && current.status === 'running' && current.progress < 90) {
            return {
              ...prev,
              [ticker]: {
                ...current,
                progress: current.progress + Math.random() * 10
              }
            };
          }
          return prev;
        });
      }, 500);
      
      const result = await calibrationApi.startCalibration({ 
        ticker, 
        horizons: [3, 15] 
      });
      
      clearInterval(progressInterval);
      
      if (result.status === 'complete' && result.results) {
        addLog(ticker, 'success', 'Optimization complete');
        
        // Log result details
        Object.entries(result.results).forEach(([horizon, hr]) => {
          if (hr.sqn !== null) {
            addLog(ticker, 'info', `Horizon ${horizon}: SQN=${hr.sqn.toFixed(2)}, Trades=${hr.trades}`);
          }
        });
        
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
        
        // Verify the calibration was saved
        const verification = await calibrationApi.verifyCalibration(ticker);
        
        if (verification.verified) {
          addLog(ticker, 'success', `Verified: ${verification.weights_count} weights saved to database`);
          setCalibratedTickers(prev => new Set([...prev, ticker]));
          
          // Force refresh analysis cache so dashboard picks up new weights
          refreshAnalysis(ticker, 0, null, null);
        } else {
          addLog(ticker, 'warning', 'Calibration completed but no weights found in database');
        }
        
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            status: 'complete',
            progress: 100,
            results: Object.fromEntries(
              Object.entries(result.results!).map(([h, r]) => [parseInt(h), r])
            ),
            verification
          }
        }));
        return true;
        
      } else if (result.status === 'error') {
        addLog(ticker, 'error', result.error || 'Unknown error');
        setCalibrations(prev => ({
          ...prev,
          [ticker]: {
            ...prev[ticker],
            status: 'error',
            progress: 0,
            error: result.error
          }
        }));
        return false;
      } else {
        // Check if all horizons failed due to insufficient trades
        const allFailed = result.results && Object.values(result.results).every(
          r => r.error || r.sqn === null
        );
        
        if (allFailed) {
          addLog(ticker, 'warning', 'All horizons failed - likely insufficient trades (need 30+)');
          addLog(ticker, 'info', 'Tight thresholds (0.3) may generate too few signals for this stock');
          
          setCalibrations(prev => ({
            ...prev,
            [ticker]: {
              ...prev[ticker],
              status: 'complete',
              progress: 100,
              results: {},
              error: 'Insufficient signals: Stock generated <30 trades with current thresholds. Try a more volatile stock or this stock may not be suitable for WFO calibration.'
            }
          }));
        }
        return false;
      }
    } catch (err) {
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
  }, [addLog]);

  // Batch Calibration
  const handleBatchCalibration = useCallback(async () => {
    if (portfolioTickers.length === 0) return;
    
    if (!window.confirm(`This will sequentially fetch data and calibrate all ${portfolioTickers.length} holdings. This may take a while. Continue?`)) {
        return;
    }

    setIsBatchRunning(true);
    
    for (const ticker of portfolioTickers) {
        // Select the ticker so the user sees progress
        setSelectedTicker(ticker);
        ensureTickerState(ticker);
        
        // 1. Fetch Data
        const fetchSuccess = await handleFetchData(ticker);
        
        if (fetchSuccess) {
            // 2. Calibrate
            await handleStartCalibration(ticker);
        } else {
            addLog(ticker, 'error', 'Skipping calibration due to fetch failure');
        }
        
        // Small delay between tickers
        await new Promise(r => setTimeout(r, 1000));
    }
    
    setIsBatchRunning(false);
    alert('Batch calibration complete!');
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
          
          <button
            onClick={handleBatchCalibration}
            disabled={isBatchRunning || portfolioTickers.length === 0}
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
            <li>Optimizes using <strong>Two-Pass Coordinate Descent</strong></li>
            <li>Validates weight <strong>stability</strong> to prevent overfitting</li>
            <li>Applies <strong>0.1% transaction cost</strong> for realistic simulation</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
