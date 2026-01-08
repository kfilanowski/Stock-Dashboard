/**
 * Option Analysis Modal
 * 
 * Displays detailed analysis of an option position, including:
 * - Large recommendation with confidence score and risk level
 * - Factor breakdown grouped by category
 * - Reasoning bullets explaining the recommendation
 */

import { useEffect, useMemo } from 'react';
import { 
  X, TrendingUp, TrendingDown, Minus, AlertTriangle, CheckCircle, Info,
  Clock, DollarSign, Activity, BarChart3, Layers, Shield
} from 'lucide-react';
import type { OptionHoldingWithData } from '../../types';
import { 
  analyzeOptionPosition, 
  type OptionRecommendation,
  type StockAnalysisData,
  type FactorAnalysis,
  type FactorCategory,
  ACTION_COLORS,
  ACTION_LABELS,
  CATEGORY_LABELS
} from '../../services/optionScoring';

interface OptionAnalysisModalProps {
  option: OptionHoldingWithData;
  stockAnalysis?: StockAnalysisData;
  onClose: () => void;
}

// Category icons
const CATEGORY_ICONS: Record<FactorCategory, typeof Clock> = {
  time: Clock,
  profit: DollarSign,
  greeks: Activity,
  market: BarChart3,
  structure: Layers,
};

// Risk level colors
const RISK_COLORS = {
  low: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/30' },
  medium: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', border: 'border-yellow-500/30' },
  high: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/30' },
};

export function OptionAnalysisModal({ 
  option, 
  stockAnalysis,
  onClose 
}: OptionAnalysisModalProps) {
  // Compute recommendation
  const recommendation = useMemo((): OptionRecommendation => {
    return analyzeOptionPosition(option, stockAnalysis);
  }, [option, stockAnalysis]);
  
  // Group factors by category
  const groupedFactors = useMemo(() => {
    const groups = new Map<FactorCategory, FactorAnalysis[]>();
    for (const factor of recommendation.factors) {
      const existing = groups.get(factor.category) || [];
      existing.push(factor);
      groups.set(factor.category, existing);
    }
    return groups;
  }, [recommendation.factors]);
  
  // Lock body scroll when modal is open
  useEffect(() => {
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);
  
  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [onClose]);
  
  const colors = ACTION_COLORS[recommendation.action];
  const label = ACTION_LABELS[recommendation.action];
  const riskColors = RISK_COLORS[recommendation.riskLevel];
  
  const Icon = recommendation.action === 'CLOSE' 
    ? TrendingDown 
    : recommendation.action === 'OPEN_MORE' 
      ? TrendingUp 
      : Minus;
  
  // Format option title
  const positionLabel = option.position_type === 'long' ? 'Long' : 'Short';
  const typeLabel = option.option_type === 'call' ? 'Call' : 'Put';
  const expirationDate = new Date(option.expiration_date + 'T00:00:00');
  const formattedExpiration = expirationDate.toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric', 
    year: '2-digit' 
  });
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Container */}
      <div 
        className="relative z-10 w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl border border-white/10 fade-in overflow-hidden"
        style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 pb-4 border-b border-white/10 flex-shrink-0">
          <div>
            <h2 className="text-xl font-bold text-white">
              {option.underlying_ticker} ${option.strike_price} {typeLabel}
            </h2>
            <p className="text-white/50 text-sm">
              {positionLabel} • Expires {formattedExpiration}
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
        <div className="flex-1 overflow-y-auto overscroll-contain p-6 space-y-6">
          {/* Main Recommendation */}
          <div className={`p-5 rounded-xl border-2 ${colors.border} ${colors.bg}`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`p-2.5 rounded-lg ${colors.bg}`}>
                  <Icon className={`w-6 h-6 ${colors.text}`} />
                </div>
                <div>
                  <h3 className={`text-2xl font-bold ${colors.text}`}>{label}</h3>
                  <p className="text-white/50 text-sm">Recommendation</p>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${colors.text}`}>
                  {recommendation.confidence}%
                </div>
                <p className="text-white/50 text-sm">Confidence</p>
              </div>
            </div>
            
            {/* Summary and Risk Level */}
            <div className="flex items-center justify-between gap-4">
              <p className="text-white/80 text-sm flex-1">
                {recommendation.summary}
              </p>
              <div className="flex items-center gap-2">
                {recommendation.isOverride && (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-accent-purple/20 border border-accent-purple/30">
                    <AlertTriangle className="w-4 h-4 text-accent-purple" />
                    <span className="text-xs font-semibold uppercase text-accent-purple">
                      Override
                    </span>
                  </div>
                )}
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg ${riskColors.bg} ${riskColors.border} border`}>
                  <Shield className={`w-4 h-4 ${riskColors.text}`} />
                  <span className={`text-xs font-semibold uppercase ${riskColors.text}`}>
                    {recommendation.riskLevel} Risk
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          {/* Key Reasons */}
          <div>
            <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Info className="w-4 h-4 text-accent-cyan" />
              Key Reasons ({recommendation.reasons.length})
            </h4>
            <div className="space-y-2">
              {recommendation.reasons.map((reason, idx) => (
                <div 
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-lg bg-white/5"
                >
                  {recommendation.action === 'CLOSE' ? (
                    <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  ) : recommendation.action === 'OPEN_MORE' ? (
                    <CheckCircle className="w-4 h-4 text-green-400 mt-0.5 flex-shrink-0" />
                  ) : (
                    <Info className="w-4 h-4 text-white/50 mt-0.5 flex-shrink-0" />
                  )}
                  <span className="text-white/80 text-sm">{reason}</span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Factor Breakdown by Category */}
          <div>
            <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-accent-purple" />
              Factor Analysis ({recommendation.factors.length} factors)
            </h4>
            <div className="space-y-4">
              {Array.from(groupedFactors.entries()).map(([category, factors]) => (
                <FactorCategorySection 
                  key={category} 
                  category={category} 
                  factors={factors} 
                />
              ))}
            </div>
          </div>
          
          {/* Current Position Info */}
          <div className="pt-4 border-t border-white/10">
            <h4 className="text-white/50 text-xs uppercase tracking-wider mb-3">
              Current Position
            </h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <PositionStat 
                label="Contracts" 
                value={option.contracts.toString()} 
              />
              <PositionStat 
                label="Current Value" 
                value={option.position_value 
                  ? `$${option.position_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                  : '—'
                } 
              />
              <PositionStat 
                label="P/L" 
                value={option.gain_loss 
                  ? `${option.gain_loss >= 0 ? '+' : ''}$${option.gain_loss.toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                  : '—'
                }
                className={option.gain_loss && option.gain_loss >= 0 ? 'text-green-400' : 'text-red-400'}
              />
              <PositionStat 
                label="P/L %" 
                value={option.gain_loss_pct 
                  ? `${option.gain_loss_pct >= 0 ? '+' : ''}${option.gain_loss_pct.toFixed(1)}%`
                  : '—'
                }
                className={option.gain_loss_pct && option.gain_loss_pct >= 0 ? 'text-green-400' : 'text-red-400'}
              />
            </div>
          </div>
          
          {/* Greeks Summary */}
          {option.greeks && (
            <div className="pt-4 border-t border-white/10">
              <h4 className="text-white/50 text-xs uppercase tracking-wider mb-3">
                Greeks
              </h4>
              <div className="grid grid-cols-4 gap-3">
                <PositionStat 
                  label="Delta" 
                  value={option.greeks.delta !== null ? `${(option.greeks.delta * 100).toFixed(0)}%` : '—'} 
                />
                <PositionStat 
                  label="Gamma" 
                  value={option.greeks.gamma !== null ? option.greeks.gamma.toFixed(4) : '—'} 
                />
                <PositionStat 
                  label="Theta" 
                  value={option.greeks.theta !== null ? `$${option.greeks.theta.toFixed(2)}` : '—'} 
                />
                <PositionStat 
                  label="Vega" 
                  value={option.greeks.vega !== null ? `$${option.greeks.vega.toFixed(2)}` : '—'} 
                />
              </div>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="p-6 pt-4 border-t border-white/10 flex-shrink-0">
          <button
            onClick={onClose}
            className="btn-primary w-full"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Sub-components
// ============================================================================

function FactorCategorySection({ 
  category, 
  factors 
}: { 
  category: FactorCategory; 
  factors: FactorAnalysis[];
}) {
  const CategoryIcon = CATEGORY_ICONS[category];
  const categoryLabel = CATEGORY_LABELS[category];
  
  // Determine category color based on signals
  const bearishCount = factors.filter(f => f.signal === 'bearish').length;
  const bullishCount = factors.filter(f => f.signal === 'bullish').length;
  const categorySignal = bearishCount > bullishCount ? 'bearish' : 
    bullishCount > bearishCount ? 'bullish' : 'neutral';
  
  const categoryColors = {
    bullish: 'text-green-400',
    bearish: 'text-red-400',
    neutral: 'text-white/60',
  };
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 px-1">
        <CategoryIcon className={`w-4 h-4 ${categoryColors[categorySignal]}`} />
        <span className="text-white/70 text-xs font-medium uppercase tracking-wider">
          {categoryLabel}
        </span>
        <span className="text-white/30 text-xs">
          ({factors.length})
        </span>
      </div>
      <div className="space-y-1.5">
        {factors.map((factor, idx) => (
          <FactorRow key={idx} factor={factor} compact />
        ))}
      </div>
    </div>
  );
}

function FactorRow({ factor, compact = false }: { factor: FactorAnalysis; compact?: boolean }) {
  const signalColors = {
    bullish: { bg: 'bg-green-500/20', text: 'text-green-400', border: 'border-green-500/20' },
    bearish: { bg: 'bg-red-500/20', text: 'text-red-400', border: 'border-red-500/20' },
    neutral: { bg: 'bg-white/10', text: 'text-white/60', border: 'border-white/10' },
  };
  
  const colors = signalColors[factor.signal];
  
  const SignalIcon = factor.signal === 'bullish' 
    ? TrendingUp 
    : factor.signal === 'bearish' 
      ? TrendingDown 
      : Minus;
  
  if (compact) {
    return (
      <div className={`flex items-center justify-between p-2.5 rounded-lg bg-white/5 border ${colors.border}`}>
        <div className="flex items-center gap-2.5 min-w-0 flex-1">
          <div className={`p-1 rounded ${colors.bg} flex-shrink-0`}>
            <SignalIcon className={`w-3 h-3 ${colors.text}`} />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-white text-sm font-medium truncate">{factor.name}</p>
            <p className="text-white/40 text-xs truncate">{factor.reasoning}</p>
          </div>
        </div>
        <div className="text-right flex-shrink-0 ml-3">
          <p className={`text-sm font-semibold ${colors.text}`}>
            {factor.value}
          </p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="flex items-center justify-between p-3 rounded-lg bg-white/5">
      <div className="flex items-center gap-3">
        <div className={`p-1.5 rounded ${colors.bg}`}>
          <SignalIcon className={`w-3.5 h-3.5 ${colors.text}`} />
        </div>
        <div>
          <p className="text-white text-sm font-medium">{factor.name}</p>
          <p className="text-white/50 text-xs">{factor.reasoning}</p>
        </div>
      </div>
      <div className="text-right">
        <p className={`text-sm font-semibold ${colors.text}`}>
          {factor.value}
        </p>
        <p className="text-white/40 text-xs capitalize">
          {factor.signal}
        </p>
      </div>
    </div>
  );
}

function PositionStat({ 
  label, 
  value, 
  className = '' 
}: { 
  label: string; 
  value: string; 
  className?: string;
}) {
  return (
    <div className="p-3 rounded-lg bg-white/5">
      <p className="text-white/40 text-xs mb-1">{label}</p>
      <p className={`text-white font-semibold ${className}`}>{value}</p>
    </div>
  );
}
