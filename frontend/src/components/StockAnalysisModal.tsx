/**
 * Stock Analysis Modal
 * 
 * Displays trading action scores with a radar chart visualization
 * and detailed metric breakdown.
 */

import { useEffect, useState, useMemo } from 'react';
import { X, TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp, AlertCircle } from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip
} from 'recharts';
import type { HistoryPoint, StockAnalysis, ActionScore, ActionType } from '../types';
import { analyzeStock, type AdditionalAnalysisData } from '../services/stockScoring';
import * as api from '../services/api';

interface StockAnalysisModalProps {
  ticker: string | null;
  onClose: () => void;
  // Optional: pass existing data to avoid re-fetching
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
  history?: HistoryPoint[];
}

// Background color for score badges
function getScoreBgClass(score: number): string {
  if (score >= 70) return 'bg-green-500/20 text-green-400 border-green-500/30';
  if (score >= 55) return 'bg-lime-500/20 text-lime-400 border-lime-500/30';
  if (score >= 45) return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
  if (score >= 30) return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
  return 'bg-red-500/20 text-red-400 border-red-500/30';
}

// Signal indicator component
function SignalIndicator({ signal }: { signal: number }) {
  if (signal > 0.3) {
    return <TrendingUp className="w-4 h-4 text-green-400" />;
  } else if (signal < -0.3) {
    return <TrendingDown className="w-4 h-4 text-red-400" />;
  }
  return <Minus className="w-4 h-4 text-yellow-400" />;
}

// Confidence badge component
function ConfidenceBadge({ confidence }: { confidence: 'high' | 'medium' | 'low' }) {
  const styles = {
    high: 'bg-green-500/20 text-green-400',
    medium: 'bg-yellow-500/20 text-yellow-400',
    low: 'bg-red-500/20 text-red-400'
  };
  
  return (
    <span className={`text-xs px-2 py-0.5 rounded-full ${styles[confidence]}`}>
      {confidence} confidence
    </span>
  );
}

// Action score card component
function ActionScoreCard({ 
  score, 
  isExpanded, 
  onToggle,
  isBest
}: { 
  score: ActionScore; 
  isExpanded: boolean;
  onToggle: () => void;
  isBest: boolean;
}) {
  return (
    <div 
      className={`rounded-lg border transition-all ${
        isBest 
          ? 'border-accent-cyan/50 bg-accent-cyan/5' 
          : 'border-white/10 bg-white/5'
      }`}
    >
      <button
        onClick={onToggle}
        className="w-full p-3 flex items-center justify-between hover:bg-white/5 transition-colors rounded-lg"
      >
        <div className="flex items-center gap-3">
          <div className={`w-12 h-12 rounded-lg flex items-center justify-center text-lg font-bold border ${getScoreBgClass(score.totalScore)}`}>
            {score.totalScore}
          </div>
          <div className="text-left">
            <div className="flex items-center gap-2">
              <span className="font-medium text-white">{score.label}</span>
              {isBest && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-accent-cyan/20 text-accent-cyan">
                  Recommended
                </span>
              )}
            </div>
            <ConfidenceBadge confidence={score.confidence} />
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-white/50" />
        ) : (
          <ChevronDown className="w-5 h-5 text-white/50" />
        )}
      </button>
      
      {isExpanded && (
        <div className="px-3 pb-3 border-t border-white/10">
          <div className="mt-3 space-y-2">
            {score.signals.map((signal, idx) => (
              <div 
                key={idx} 
                className="flex items-center justify-between text-sm py-1.5 px-2 rounded bg-white/5"
              >
                <div className="flex items-center gap-2">
                  <SignalIndicator signal={signal.signal} />
                  <span className="text-white/70">{signal.metricLabel}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-white/50 text-xs">{signal.rawValue}</span>
                  <span className={`font-mono text-xs ${
                    signal.contribution > 0 ? 'text-green-400' : 
                    signal.contribution < 0 ? 'text-red-400' : 'text-white/50'
                  }`}>
                    {signal.contribution > 0 ? '+' : ''}{signal.contribution.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-white/40 mt-2 italic">
            {score.signals[0]?.reasoning}
          </p>
        </div>
      )}
    </div>
  );
}

// Note: high52w, low52w, history props are accepted but not used
// We always fetch full history for consistent analysis (need 200+ days for SMA200)
export function StockAnalysisModal({
  ticker,
  onClose,
  currentPrice: propCurrentPrice
}: StockAnalysisModalProps) {
  const [analysis, setAnalysis] = useState<StockAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedAction, setExpandedAction] = useState<ActionType | null>(null);
  
  // Track if we've done the initial load
  const hasInitialData = analysis !== null;
  
  // Fetch data and analyze - only run when ticker changes (modal opens)
  // Don't re-run when props update from parent polling
  useEffect(() => {
    if (!ticker) return;
    
    // Capture ticker in a const to satisfy TypeScript in the async closure
    const currentTicker = ticker;
    
    async function fetchAndAnalyze() {
      // Show full loading spinner only on initial load
      // For refreshes, show subtle indicator and keep old data visible
      if (!hasInitialData) {
        setLoading(true);
      } else {
        setIsRefreshing(true);
      }
      setError(null);
      
      try {
        // Fetch FULL history (1 year) and stock data in parallel
        // The /stock/{ticker} endpoint only returns 30 days for mini charts
        // We need the /stock/{ticker}/history endpoint for full analysis
        // This ensures:
        // 1. SMA200 and Golden Cross can be calculated (need 200+ days)
        // 2. RSI/MACD are properly smoothed with consistent data
        // 3. Fair comparison across all stocks
        const [historyResponse, stockData] = await Promise.all([
          api.getStockHistory(currentTicker, '1y'),  // Full year of history
          api.getStock(currentTicker)                 // Current price and 52w data
        ]);
        
        const history = historyResponse.history;
        let currentPrice = stockData.current_price || propCurrentPrice;
        const high52w = stockData.high_52w;
        const low52w = stockData.low_52w;
        
        // If still missing price, try to get it from quote
        if (!currentPrice) {
          const quote = await api.getStockQuote(currentTicker);
          currentPrice = quote.current_price;
        }
        
        if (!history || history.length < 14) {
          // Only show error if we don't have existing data
          if (!hasInitialData) {
            setError('Insufficient price history for analysis (need at least 14 days)');
          }
          return;
        }
        
        if (!currentPrice) {
          if (!hasInitialData) {
            setError('Unable to get current price');
          }
          return;
        }
        
        // Fetch additional analysis data (fundamentals, options, sector) in parallel
        // These are optional - analysis works without them
        let additionalData: AdditionalAnalysisData = {};
        
        try {
          const [fundamentals, options, sectorData] = await Promise.allSettled([
            api.getStockFundamentals(currentTicker),
            api.getStockOptions(currentTicker),
            api.getSectorCorrelation(currentTicker)
          ]);
          
          // Merge fundamentals data
          if (fundamentals.status === 'fulfilled') {
            additionalData.roic = fundamentals.value.roic;
            additionalData.roe = fundamentals.value.roe;
            additionalData.sector = fundamentals.value.sector;
            additionalData.industry = fundamentals.value.industry;
            additionalData.beta = fundamentals.value.beta;
          }
          
          // Merge options data
          if (options.status === 'fulfilled') {
            additionalData.callPutRatioOI = options.value.call_put_ratio_oi;
            additionalData.callPutRatioVolume = options.value.call_put_ratio_volume;
            additionalData.optionsSentiment = options.value.options_sentiment;
            additionalData.avgImpliedVolatility = options.value.avg_implied_volatility;
            additionalData.ivPercentile = options.value.iv_percentile;
            additionalData.hasOptions = options.value.has_options;
          }
          
          // Merge sector correlation data
          if (sectorData.status === 'fulfilled') {
            additionalData.sectorCorrelation = sectorData.value.correlation;
            additionalData.betaToSector = sectorData.value.beta_to_sector;
          }
        } catch {
          // Additional data is optional, continue without it
          console.warn('Could not fetch additional analysis data');
        }
        
        // Perform analysis with all available data
        const result = analyzeStock(
          currentTicker,
          history,
          currentPrice,
          high52w ?? null,
          low52w ?? null,
          undefined, // Use default weights
          additionalData
        );
        
        setAnalysis(result);
        
        // Auto-expand best action only on initial load
        if (!hasInitialData) {
          setExpandedAction(result.bestAction.action);
        }
      } catch (err) {
        // Only show error if we don't have existing data to display
        if (!hasInitialData) {
          setError(err instanceof Error ? err.message : 'Analysis failed');
        }
      } finally {
        setLoading(false);
        setIsRefreshing(false);
      }
    }
    
    fetchAndAnalyze();
    // Only re-run when ticker changes, not when props update from polling
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);
  
  // Prepare radar chart data
  const radarData = useMemo(() => {
    if (!analysis) return [];
    
    return analysis.scores.map(score => ({
      action: score.label,
      score: score.totalScore,
      fullMark: 100
    }));
  }, [analysis]);
  
  if (!ticker) return null;
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative glass-card w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-white">
                {ticker} Analysis
              </h2>
              {isRefreshing && (
                <div className="flex items-center gap-1.5 text-xs text-accent-cyan/70">
                  <div className="animate-spin rounded-full h-3 w-3 border border-accent-cyan/50 border-t-accent-cyan" />
                  <span>Updating...</span>
                </div>
              )}
            </div>
            <p className="text-sm text-white/50">
              Trading action recommendations based on technical indicators
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-white/70" />
          </button>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Show full loading spinner only on initial load (no existing data) */}
          {loading && !analysis && (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-accent-cyan border-t-transparent" />
              <span className="ml-3 text-white/70">Analyzing...</span>
            </div>
          )}
          
          {/* Only show error if we have no data to display */}
          {error && !analysis && (
            <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <span className="text-red-400">{error}</span>
            </div>
          )}
          
          {/* Show analysis content - keep visible even during background refresh */}
          {analysis && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Radar Chart */}
              <div className="glass-card p-4">
                <h3 className="text-sm font-medium text-white/70 mb-4">Action Scores</h3>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                      <PolarGrid 
                        stroke="rgba(255,255,255,0.1)" 
                        strokeDasharray="3 3"
                      />
                      <PolarAngleAxis 
                        dataKey="action" 
                        tick={{ fill: 'rgba(255,255,255,0.7)', fontSize: 11 }}
                        tickLine={false}
                      />
                      <PolarRadiusAxis 
                        angle={30} 
                        domain={[0, 100]} 
                        tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 10 }}
                        tickCount={5}
                        axisLine={false}
                      />
                      <Radar
                        name="Score"
                        dataKey="score"
                        stroke="#06b6d4"
                        fill="#06b6d4"
                        fillOpacity={0.3}
                        strokeWidth={2}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'rgba(15, 23, 42, 0.95)',
                          border: '1px solid rgba(255,255,255,0.1)',
                          borderRadius: '8px',
                          padding: '8px 12px'
                        }}
                        labelStyle={{ color: 'white', fontWeight: 'bold' }}
                        itemStyle={{ color: '#06b6d4' }}
                        formatter={(value: number) => [`${value}/100`, 'Score']}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
                
                {/* Data quality note */}
                <div className="mt-4 text-xs text-white/40 space-y-1">
                  <div className="flex items-center gap-2">
                    <AlertCircle className="w-3 h-3" />
                    <span>
                      {analysis.dataQuality.historyDays} days of history
                      {analysis.dataQuality.hasSMA200 
                        ? ' (full analysis)' 
                        : ' (limited - need 200+ for SMA200)'
                      }
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3" />
                    <span>
                      Using {analysis.dataQuality.availableMetrics}/{analysis.dataQuality.totalMetrics} metrics
                      {analysis.dataQuality.missingMetrics.length > 0 && (
                        <span className="ml-1 text-yellow-400/70">
                          (missing: {analysis.dataQuality.missingMetrics.slice(0, 3).join(', ')}
                          {analysis.dataQuality.missingMetrics.length > 3 && '...'})
                        </span>
                      )}
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Action Scores List */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-white/70">Action Breakdown</h3>
                {analysis.scores
                  .sort((a, b) => b.totalScore - a.totalScore)
                  .map(score => (
                    <ActionScoreCard
                      key={score.action}
                      score={score}
                      isExpanded={expandedAction === score.action}
                      onToggle={() => setExpandedAction(
                        expandedAction === score.action ? null : score.action
                      )}
                      isBest={score.action === analysis.bestAction.action}
                    />
                  ))
                }
              </div>
            </div>
          )}
        </div>
        
        {/* Footer */}
        {analysis && (
          <div className="p-4 border-t border-white/10 bg-white/5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-white/50">
                Analysis performed at {analysis.analyzedAt.toLocaleTimeString()}
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-white/70">Best Action:</span>
                <span className={`font-bold px-3 py-1 rounded-lg border ${getScoreBgClass(analysis.bestAction.totalScore)}`}>
                  {analysis.bestAction.label} ({analysis.bestAction.totalScore})
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

