/**
 * Stock Analysis Modal
 * 
 * Displays trading action scores with a radar chart visualization
 * and detailed metric breakdown.
 * 
 * Uses the shared analysis cache from useStockAnalysis to ensure
 * consistency between the badge and the detailed modal view.
 */

import { useState, useMemo, useCallback } from 'react';
import { X, TrendingUp, TrendingDown, Minus, ChevronDown, ChevronUp, AlertCircle, RefreshCw, HelpCircle } from 'lucide-react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Tooltip
} from 'recharts';
import type { ActionScore, ActionType } from '../types';
import { useStockAnalysis, refreshAnalysis } from '../hooks/useStockAnalysis';

interface StockAnalysisModalProps {
  ticker: string | null;
  onClose: () => void;
  currentPrice?: number;
  high52w?: number | null;
  low52w?: number | null;
}

// Indicator descriptions for user reference
const indicatorDescriptions: Record<string, { name: string; description: string; interpretation: string }> = {
  rsi: {
    name: 'RSI (Relative Strength Index)',
    description: 'Momentum oscillator measuring speed and magnitude of price changes on a 0-100 scale.',
    interpretation: 'Below 30 = oversold (potential bounce). Above 70 = overbought (potential pullback). In strong trends, these levels can persist.'
  },
  macd: {
    name: 'MACD',
    description: 'Moving Average Convergence Divergence - shows relationship between two moving averages of price.',
    interpretation: 'Bullish when MACD crosses above signal line. Histogram shows momentum strength. Divergences can signal reversals.'
  },
  bollingerBands: {
    name: 'Bollinger Bands (%B)',
    description: 'Shows where price sits relative to volatility bands around a 20-day moving average.',
    interpretation: '%B of 0% = at lower band (oversold). 100% = at upper band (overbought). 50% = at middle band.'
  },
  bollingerSqueeze: {
    name: 'Volatility Squeeze',
    description: 'Measures band width contraction. Tight bands often precede explosive moves.',
    interpretation: 'Squeeze = low volatility, cheap options premiums. Expansion = high volatility, rich premiums for selling.'
  },
  vwap: {
    name: 'VWAP',
    description: 'Volume Weighted Average Price - the average price weighted by trading volume.',
    interpretation: 'Acts as support when above, resistance when below. Institutional traders often use VWAP as a benchmark.'
  },
  momentum: {
    name: 'Momentum',
    description: 'Rate of price change over short and medium timeframes.',
    interpretation: 'Positive momentum = bullish trend. Negative = bearish. Slowing momentum can signal trend exhaustion.'
  },
  volume: {
    name: 'Volume',
    description: 'Trading activity level compared to recent average.',
    interpretation: 'High volume confirms price moves. Low volume suggests weak conviction. Watch volume on breakouts.'
  },
  rvol: {
    name: 'Relative Volume (RVOL)',
    description: 'Today\'s volume as a multiple of average daily volume.',
    interpretation: 'RVOL > 1.5x = unusual activity. > 3x = potential catalyst. Low RVOL = lack of interest.'
  },
  pricePosition: {
    name: '52-Week Range Position',
    description: 'Where current price sits within the 52-week high/low range.',
    interpretation: 'Near lows = potential value (or falling knife). Near highs = strength (or resistance). Context matters.'
  },
  smaAlignment: {
    name: 'SMA Alignment',
    description: 'Price position relative to 20, 50, and 200-day Simple Moving Averages.',
    interpretation: 'Price above all SMAs with SMA20 > SMA50 > SMA200 = strong bullish alignment. Inverse = bearish.'
  },
  adx: {
    name: 'ADX (Trend Strength)',
    description: 'Average Directional Index measures trend strength regardless of direction.',
    interpretation: 'ADX > 25 = trending market. ADX < 20 = range-bound. Strong trends (>40) favor momentum plays.'
  },
  crossPattern: {
    name: 'Cross Patterns',
    description: 'Golden Cross (SMA50 crosses above SMA200) and Death Cross (opposite).',
    interpretation: 'Golden Cross = long-term bullish signal. Death Cross = bearish. Best with volume confirmation.'
  },
  roic: {
    name: 'ROIC',
    description: 'Return on Invested Capital - measures how efficiently a company uses capital.',
    interpretation: 'ROIC > 15% = quality business. > 20% = exceptional. Important for long-term holds.'
  },
  callPutRatio: {
    name: 'Call/Put Ratio',
    description: 'Ratio of call to put options activity in the market.',
    interpretation: 'High ratio (>1.5) = bullish sentiment. Low ratio (<0.7) = bearish. Can be contrarian indicator at extremes.'
  },
  ivPercentile: {
    name: 'IV Percentile',
    description: 'Where current implied volatility ranks vs. its historical range.',
    interpretation: 'High IV = expensive options (sell premium). Low IV = cheap options (buy calls/puts).'
  },
  sectorBeta: {
    name: 'Sector Beta',
    description: 'Stock\'s correlation and volatility relative to its sector.',
    interpretation: 'Beta > 1 = more volatile than sector. Beta < 1 = defensive. Helps gauge relative risk.'
  }
};

// Indicator Guide Component
function IndicatorGuide({ isOpen, onToggle }: { isOpen: boolean; onToggle: () => void }) {
  return (
    <div className="glass-card p-3 mb-4">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between text-left hover:bg-white/5 rounded-lg p-1 -m-1 transition-colors"
      >
        <div className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4 text-accent-cyan" />
          <span className="text-sm font-medium text-white/80">Indicator Guide</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-white/50" />
        ) : (
          <ChevronDown className="w-4 h-4 text-white/50" />
        )}
      </button>
      
      {isOpen && (
        <div className="mt-3 space-y-3 max-h-[300px] overflow-y-auto pr-2">
          {Object.entries(indicatorDescriptions).map(([key, info]) => (
            <div key={key} className="border-l-2 border-accent-cyan/30 pl-3 py-1">
              <h4 className="text-sm font-medium text-white/90">{info.name}</h4>
              <p className="text-xs text-white/50 mt-0.5">{info.description}</p>
              <p className="text-xs text-accent-cyan/70 mt-1 italic">{info.interpretation}</p>
            </div>
          ))}
          
          {/* Action types explanation */}
          <div className="border-t border-white/10 pt-3 mt-4">
            <h4 className="text-sm font-medium text-white/90 mb-2">Trading Actions</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-white/5 rounded p-2">
                <span className="text-green-400 font-medium">Buy Shares</span>
                <p className="text-white/50 mt-0.5">Long stock position for growth/value</p>
              </div>
              <div className="bg-white/5 rounded p-2">
                <span className="text-red-400 font-medium">Sell Shares</span>
                <p className="text-white/50 mt-0.5">Exit or reduce stock position</p>
              </div>
              <div className="bg-white/5 rounded p-2">
                <span className="text-blue-400 font-medium">Open CSP</span>
                <p className="text-white/50 mt-0.5">Sell cash-secured put to collect premium</p>
              </div>
              <div className="bg-white/5 rounded p-2">
                <span className="text-purple-400 font-medium">Open CC</span>
                <p className="text-white/50 mt-0.5">Sell covered call on held shares</p>
              </div>
              <div className="bg-white/5 rounded p-2">
                <span className="text-lime-400 font-medium">Buy Call</span>
                <p className="text-white/50 mt-0.5">Long call for bullish leverage</p>
              </div>
              <div className="bg-white/5 rounded p-2">
                <span className="text-orange-400 font-medium">Buy Put</span>
                <p className="text-white/50 mt-0.5">Long put for bearish bet or hedge</p>
              </div>
            </div>
          </div>
          
          {/* Signal interpretation */}
          <div className="border-t border-white/10 pt-3">
            <h4 className="text-sm font-medium text-white/90 mb-2">Reading Signals</h4>
            <div className="text-xs text-white/60 space-y-1">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span>Positive signal for this action (+contribution)</span>
              </div>
              <div className="flex items-center gap-2">
                <TrendingDown className="w-3 h-3 text-red-400" />
                <span>Negative signal for this action (-contribution)</span>
              </div>
              <div className="flex items-center gap-2">
                <Minus className="w-3 h-3 text-yellow-400" />
                <span>Neutral/weak signal (near zero)</span>
              </div>
              <p className="mt-2 text-white/40">
                Each indicator contributes to the total score. Higher scores indicate stronger alignment 
                with that action based on current market conditions.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
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

export function StockAnalysisModal({
  ticker,
  onClose,
  currentPrice,
  high52w,
  low52w
}: StockAnalysisModalProps) {
  // Use the shared analysis cache - same data as ActionScoreBadge
  const { analysis, isLoading, isStale, error, lastUpdated } = useStockAnalysis(
    ticker,
    currentPrice,
    high52w,
    low52w
  );
  
  const [expandedAction, setExpandedAction] = useState<ActionType | null>(null);
  const [showIndicatorGuide, setShowIndicatorGuide] = useState(false);
  
  // Auto-expand best action when analysis loads
  useMemo(() => {
    if (analysis && !expandedAction) {
      setExpandedAction(analysis.bestAction.action);
    }
  }, [analysis, expandedAction]);
  
  // Force refresh handler
  const handleRefresh = useCallback(() => {
    if (ticker && currentPrice) {
      refreshAnalysis(ticker, currentPrice, high52w ?? null, low52w ?? null);
    }
  }, [ticker, currentPrice, high52w, low52w]);
  
  // Options-related actions to filter when stock has no options
  const optionsActions: ActionType[] = ['openCSP', 'openCC', 'buyCall', 'buyPut'];
  
  // Filter scores to only show applicable actions
  const applicableScores = useMemo(() => {
    if (!analysis) return [];
    
    // If stock has options, show all actions; otherwise filter out options actions
    return analysis.hasOptions 
      ? analysis.scores 
      : analysis.scores.filter(s => !optionsActions.includes(s.action));
  }, [analysis]);
  
  // Prepare radar chart data (only applicable actions)
  const radarData = useMemo(() => {
    return applicableScores.map(score => ({
      action: score.label,
      score: score.totalScore,
      fullMark: 100
    }));
  }, [applicableScores]);
  
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
              {isLoading && analysis && (
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
          <div className="flex items-center gap-2">
            {/* Refresh button */}
            <button
              onClick={handleRefresh}
              disabled={isLoading}
              className={`
                p-2 rounded-lg transition-colors
                ${isLoading 
                  ? 'bg-white/5 text-white/30 cursor-not-allowed' 
                  : 'hover:bg-white/10 text-white/70 hover:text-white'
                }
              `}
              title="Refresh analysis"
            >
              <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/10 transition-colors"
            >
              <X className="w-5 h-5 text-white/70" />
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {/* Show full loading spinner only when no data */}
          {isLoading && !analysis && (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-2 border-accent-cyan border-t-transparent" />
              <span className="ml-3 text-white/70">Analyzing {ticker}...</span>
            </div>
          )}
          
          {/* Only show error if we have no data to display */}
          {error && !analysis && (
            <div className="flex flex-col items-center gap-4 py-12">
              <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <span className="text-red-400">{error}</span>
              </div>
              <button
                onClick={handleRefresh}
                className="px-4 py-2 rounded-lg bg-accent-cyan/20 text-accent-cyan hover:bg-accent-cyan/30 transition-colors"
              >
                Try Again
              </button>
            </div>
          )}
          
          {/* Show analysis content - keep visible even during background refresh */}
          {analysis && (
            <>
              {/* Collapsible Indicator Guide */}
              <IndicatorGuide 
                isOpen={showIndicatorGuide} 
                onToggle={() => setShowIndicatorGuide(!showIndicatorGuide)} 
              />
              
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
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium text-white/70">Action Breakdown</h3>
                  {!analysis.hasOptions && (
                    <span className="text-xs text-yellow-400/70 bg-yellow-500/10 px-2 py-0.5 rounded">
                      No options available
                    </span>
                  )}
                </div>
                {applicableScores
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
            </>
          )}
        </div>
        
        {/* Footer */}
        {analysis && (
          <div className="p-4 border-t border-white/10 bg-white/5">
            <div className="flex items-center justify-between">
              <div className="text-sm text-white/50">
                {lastUpdated ? (
                  <>
                    Analysis from {lastUpdated.toLocaleTimeString()}
                    {isStale && <span className="text-yellow-400/70 ml-2">(refreshing...)</span>}
                  </>
                ) : (
                  'Analysis performed just now'
                )}
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
