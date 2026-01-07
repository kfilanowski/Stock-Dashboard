import { useMemo, useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Trash2, TrendingUp, TrendingDown, Clock, Target, AlertTriangle, HelpCircle, X } from 'lucide-react';
import type { OptionHoldingWithData } from '../types';

interface OptionHoldingCardProps {
  option: OptionHoldingWithData;
  onDelete: (id: number) => void;
}

/**
 * Format a number as currency
 */
function formatCurrency(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 2 
  });
}

/**
 * Format a percentage value
 */
function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

/**
 * Get expiration status color and text
 */
function getExpirationStatus(daysToExp: number | undefined): { color: string; text: string; urgent: boolean } {
  if (daysToExp === undefined) return { color: 'text-white/50', text: '', urgent: false };
  
  if (daysToExp <= 0) {
    return { color: 'text-red-400', text: 'EXPIRED', urgent: true };
  } else if (daysToExp <= 7) {
    return { color: 'text-red-400', text: `${daysToExp}d`, urgent: true };
  } else if (daysToExp <= 30) {
    return { color: 'text-yellow-400', text: `${daysToExp}d`, urgent: false };
  } else {
    return { color: 'text-white/60', text: `${daysToExp}d`, urgent: false };
  }
}

export function OptionHoldingCard({ option, onDelete }: OptionHoldingCardProps) {
  const isCall = option.option_type === 'call';
  const isLong = option.position_type === 'long';
  const isPositive = (option.gain_loss ?? 0) >= 0;
  const [showEducation, setShowEducation] = useState(false);
  
  const expStatus = useMemo(() => 
    getExpirationStatus(option.analytics?.days_to_expiration), 
    [option.analytics?.days_to_expiration]
  );
  
  // Color themes based on option type
  const typeColors = isCall 
    ? { bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', badge: 'bg-green-500/20' }
    : { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', badge: 'bg-red-500/20' };
  
  const positionLabel = isLong ? 'LONG' : 'SHORT';
  const typeLabel = isCall ? 'CALL' : 'PUT';
  
  // Format expiration date for display
  const formattedExpiration = useMemo(() => {
    const date = new Date(option.expiration_date + 'T00:00:00');
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
  }, [option.expiration_date]);

  return (
    <div className={`glass-card p-5 hover:border-white/20 transition-all duration-300 group ${typeColors.border} border-l-2`}>
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        {/* Left: Ticker and contract info */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <h3 className="font-bold text-white text-xl">{option.underlying_ticker}</h3>
            <span className={`px-2 py-0.5 rounded text-xs font-semibold ${typeColors.badge} ${typeColors.text}`}>
              {positionLabel} {typeLabel}
            </span>
            {option.analytics?.is_itm && (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-accent-cyan/20 text-accent-cyan">
                ITM
              </span>
            )}
          </div>
          
          {/* Strike and expiration */}
          <div className="flex items-center gap-3 text-sm">
            <div className="flex items-center gap-1">
              <Target className="w-3.5 h-3.5 text-white/50" />
              <span className="text-white font-medium">${option.strike_price}</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className={`w-3.5 h-3.5 ${expStatus.color}`} />
              <span className={`font-medium ${expStatus.color}`}>
                {formattedExpiration}
              </span>
              {expStatus.text && (
                <span className={`text-xs ${expStatus.color}`}>
                  ({expStatus.text})
                </span>
              )}
              {expStatus.urgent && <AlertTriangle className="w-3 h-3 text-red-400 ml-1" />}
            </div>
          </div>
        </div>

        {/* Right: Actions and value */}
        <div className="flex flex-col items-end gap-1">
          {/* Action buttons */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => onDelete(option.id)}
              className="p-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 transition-colors"
              title="Remove option"
            >
              <Trash2 className="w-4 h-4 text-red-400" />
            </button>
          </div>
          
          {/* Position value */}
          <div className="text-right">
            <p className="text-white font-semibold">
              {formatCurrency(option.position_value)}
            </p>
            <div className="flex items-center gap-1 justify-end">
              {isPositive ? (
                <TrendingUp className="w-3 h-3 text-green-400" />
              ) : (
                <TrendingDown className="w-3 h-3 text-red-400" />
              )}
              <span className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(option.gain_loss)} ({formatPercent(option.gain_loss_pct)})
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Contract details row */}
      <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-white/5 mb-3">
        <div className="flex gap-4">
          <div>
            <span className="text-white/40 text-xs block">Contracts</span>
            <span className="text-white font-medium">{option.contracts}</span>
          </div>
          <div title="Premium you paid (long) or received (short) per contract">
            <span className="text-white/40 text-xs block">Premium</span>
            <span className="text-white font-medium">
              {option.premium_per_contract ? `$${option.premium_per_contract.toFixed(2)}` : '—'}
            </span>
          </div>
          <div title="Total amount invested: Premium × 100 shares × Contracts">
            <span className="text-white/40 text-xs block">
              {isLong ? 'You Paid' : 'You Received'}
            </span>
            <span className="text-white font-medium">{formatCurrency(option.cost_basis)}</span>
          </div>
        </div>
        <div className="text-right" title="Current option price per share">
          <span className="text-white/40 text-xs block">Option Price</span>
          <span className="text-white font-medium">
            {option.current_price ? `$${option.current_price.toFixed(2)}` : '—'}
          </span>
        </div>
      </div>

      {/* Greeks row */}
      {option.greeks && (
        <div className="flex justify-between py-2 px-3 rounded-lg bg-white/5 mb-3">
          <div className="flex items-center gap-3">
            <div className="flex gap-4">
              <GreekBadge label="Δ" value={option.greeks.delta} multiplier={100} suffix="%" />
              <GreekBadge label="Γ" value={option.greeks.gamma} multiplier={1} decimals={4} />
              <GreekBadge label="Θ" value={option.greeks.theta} multiplier={100} prefix="$" isNegativeGood={isLong} />
              <GreekBadge label="V" value={option.greeks.vega} multiplier={1} prefix="$" />
            </div>
            <button
              onClick={() => setShowEducation(true)}
              className="p-1 rounded-full hover:bg-white/10 transition-colors"
              title="Learn about Greeks & IV"
            >
              <HelpCircle className="w-4 h-4 text-white/40 hover:text-accent-cyan" />
            </button>
          </div>
          {option.implied_volatility && (
            <div className="text-right">
              <span className="text-white/40 text-xs block">IV</span>
              <span className="text-accent-purple font-medium">{option.implied_volatility.toFixed(1)}%</span>
            </div>
          )}
        </div>
      )}

      {/* Educational Modal - rendered via portal to escape card container */}
      {showEducation && createPortal(
        <OptionsEducationModal onClose={() => setShowEducation(false)} isLong={isLong} />,
        document.body
      )}

      {/* Analytics row */}
      {option.analytics && (
        <div className="flex justify-between items-center pt-2 border-t border-white/5">
          <div className="flex gap-4 text-xs">
            <div>
              <span className="text-white/40">Breakeven</span>
              <span className="text-white ml-1 font-medium">
                {option.analytics.breakeven_price ? `$${option.analytics.breakeven_price.toFixed(2)}` : '—'}
              </span>
            </div>
            <div>
              <span className="text-white/40">Underlying</span>
              <span className="text-white ml-1 font-medium">
                {option.underlying_price ? `$${option.underlying_price.toFixed(2)}` : '—'}
              </span>
            </div>
            {option.analytics.intrinsic_value !== null && (
              <div>
                <span className="text-white/40">Intrinsic</span>
                <span className="text-white ml-1 font-medium">
                  ${option.analytics.intrinsic_value.toFixed(2)}
                </span>
              </div>
            )}
          </div>
          
          {/* Profit probability */}
          {option.analytics.profit_probability !== null && (
            <div className="flex items-center gap-2">
              <span className="text-white/40 text-xs">P(Profit)</span>
              <div className="flex items-center gap-1">
                <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${
                      option.analytics.profit_probability >= 50 
                        ? 'bg-green-400' 
                        : 'bg-red-400'
                    }`}
                    style={{ width: `${option.analytics.profit_probability}%` }}
                  />
                </div>
                <span className={`text-xs font-medium ${
                  option.analytics.profit_probability >= 50 
                    ? 'text-green-400' 
                    : 'text-red-400'
                }`}>
                  {option.analytics.profit_probability.toFixed(0)}%
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Notes */}
      {option.notes && (
        <div className="mt-2 text-xs text-white/40 italic">
          {option.notes}
        </div>
      )}
    </div>
  );
}

/**
 * Small badge displaying a Greek value
 */
function GreekBadge({ 
  label, 
  value, 
  multiplier = 1, 
  decimals = 2,
  prefix = '',
  suffix = '',
  isNegativeGood = false
}: { 
  label: string; 
  value: number | null; 
  multiplier?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  isNegativeGood?: boolean;
}) {
  if (value === null) return null;
  
  const displayValue = value * multiplier;
  const isNegative = displayValue < 0;
  
  // For theta, negative is bad for long positions (time decay hurts)
  const colorClass = isNegativeGood 
    ? (isNegative ? 'text-green-400' : 'text-red-400')
    : 'text-white';
  
  return (
    <div>
      <span className="text-accent-cyan text-xs font-bold">{label}</span>
      <span className={`text-xs ml-1 font-medium ${colorClass}`}>
        {prefix}{displayValue >= 0 ? '' : ''}{displayValue.toFixed(decimals)}{suffix}
      </span>
    </div>
  );
}

/**
 * Educational modal explaining Greeks and option metrics
 */
function OptionsEducationModal({ onClose, isLong }: { onClose: () => void; isLong: boolean }) {
  // Lock body scroll when modal is open
  useEffect(() => {
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    
    return () => {
      document.body.style.overflow = originalOverflow;
    };
  }, []);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal Container */}
      <div 
        className="relative z-10 w-full max-w-2xl max-h-[85vh] flex flex-col rounded-2xl border border-white/10 fade-in"
        style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%)',
          backdropFilter: 'blur(12px)',
        }}
      >
        {/* Fixed Header */}
        <div className="flex items-center justify-between p-6 pb-4 border-b border-white/10 flex-shrink-0">
          <h2 className="text-xl font-bold gradient-text">Understanding Options Metrics</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-white/70" />
          </button>
        </div>
        
        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto overscroll-contain p-6 pt-4">

        {/* The Greeks */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-accent-cyan mb-3">The Greeks</h3>
          <div className="space-y-4">
            <EducationItem
              symbol="Δ"
              name="Delta"
              description="How much the option price moves per $1 move in the stock."
              example="Delta of 50% means if stock rises $1, option rises ~$0.50"
              goodValues="Calls: 0-100%, Puts: -100%-0%. Higher absolute value = more responsive to stock moves."
              tip={isLong 
                ? "For long calls, you want delta to increase. For long puts, you want it to decrease (become more negative)."
                : "For short positions, lower absolute delta means less risk from stock movement."
              }
            />
            <EducationItem
              symbol="Γ"
              name="Gamma"
              description="How fast delta changes as the stock moves. Measures acceleration."
              example="High gamma means delta can change quickly with small stock moves."
              goodValues="Highest when at-the-money and near expiration. Range: 0 to ~0.05 typically."
              tip={isLong
                ? "High gamma is good for long options - your delta improves as the stock moves in your favor."
                : "High gamma is risky for short options - your position can move against you quickly."
              }
            />
            <EducationItem
              symbol="Θ"
              name="Theta"
              description="How much value the option loses per day due to time decay."
              example="Theta of -$5 means the option loses $5 in value each day, all else equal."
              goodValues="Always negative for long options. Accelerates as expiration approaches."
              tip={isLong
                ? "Theta works against you. The option loses value every day you hold it."
                : "Theta works FOR you. You profit as the option decays toward zero."
              }
              isNegativeGood={!isLong}
            />
            <EducationItem
              symbol="V"
              name="Vega"
              description="How much the option price changes per 1% change in implied volatility."
              example="Vega of $0.15 means if IV rises 1%, option gains $0.15 per share."
              goodValues="Higher for longer-dated options. Typically $0.01-$0.50."
              tip={isLong
                ? "High vega means you benefit when volatility increases (like before earnings)."
                : "High vega is risky for shorts - a volatility spike can hurt your position."
              }
            />
          </div>
        </div>

        {/* Implied Volatility */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-accent-purple mb-3">Implied Volatility (IV)</h3>
          <div className="bg-white/5 rounded-lg p-4">
            <p className="text-white/80 mb-2">
              <strong>What it is:</strong> The market's expectation of how much the stock will move. Higher IV = more expensive options.
            </p>
            <p className="text-white/80 mb-3">
              <strong>How to interpret:</strong>
            </p>
            <ul className="space-y-2 text-sm text-white/70">
              <li className="flex items-start gap-2">
                <span className="text-green-400">•</span>
                <span><strong className="text-green-400">Low IV (under 20%):</strong> Market expects calm trading. Options are cheap. Good time to buy options.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-yellow-400">•</span>
                <span><strong className="text-yellow-400">Medium IV (20-40%):</strong> Normal volatility range for most stocks.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-red-400">•</span>
                <span><strong className="text-red-400">High IV (over 50%):</strong> Market expects big moves (earnings, news). Options are expensive. Good time to sell options.</span>
              </li>
            </ul>
            <p className="text-white/60 text-xs mt-3 italic">
              Tip: IV tends to drop after events like earnings ("IV crush"), hurting long option holders even if the stock moves the right way.
            </p>
          </div>
        </div>

        {/* Other Terms */}
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Other Terms</h3>
          <div className="space-y-3 text-sm">
            <div className="bg-white/5 rounded-lg p-3">
              <span className="text-accent-cyan font-medium">Breakeven:</span>
              <span className="text-white/70 ml-2">
                The stock price where you neither profit nor lose at expiration. For calls: Strike + Premium. For puts: Strike - Premium.
              </span>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <span className="text-accent-cyan font-medium">Underlying:</span>
              <span className="text-white/70 ml-2">
                The current stock price that the option is based on.
              </span>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <span className="text-accent-cyan font-medium">Intrinsic Value:</span>
              <span className="text-white/70 ml-2">
                The "real" value if exercised now. For calls: Stock Price - Strike. For puts: Strike - Stock Price. Can never be negative. If $0, you're "out of the money."
              </span>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <span className="text-accent-cyan font-medium">ITM (In The Money):</span>
              <span className="text-white/70 ml-2">
                When the option has intrinsic value. Calls: stock above strike. Puts: stock below strike.
              </span>
            </div>
            <div className="bg-white/5 rounded-lg p-3">
              <span className="text-accent-cyan font-medium">P(Profit):</span>
              <span className="text-white/70 ml-2">
                Estimated probability your position will be profitable at expiration, based on current IV and stock price.
              </span>
            </div>
          </div>
        </div>
        </div>
        {/* End scrollable content */}

        {/* Fixed Footer */}
        <div className="p-6 pt-4 border-t border-white/10 flex-shrink-0">
          <button
            onClick={onClose}
            className="btn-primary w-full"
          >
            Got it!
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Single education item for a Greek
 */
function EducationItem({ 
  symbol, 
  name, 
  description, 
  example, 
  goodValues, 
  tip,
  isNegativeGood = false
}: {
  symbol: string;
  name: string;
  description: string;
  example: string;
  goodValues: string;
  tip: string;
  isNegativeGood?: boolean;
}) {
  return (
    <div className="bg-white/5 rounded-lg p-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-accent-cyan text-lg font-bold">{symbol}</span>
        <span className="text-white font-semibold">{name}</span>
      </div>
      <p className="text-white/80 text-sm mb-2">{description}</p>
      <p className="text-white/60 text-xs mb-2 italic">Example: {example}</p>
      <p className="text-white/60 text-xs mb-2">
        <span className="text-accent-purple">Good values:</span> {goodValues}
      </p>
      <p className={`text-xs ${isNegativeGood ? 'text-green-400/80' : 'text-accent-cyan/80'}`}>
        💡 {tip}
      </p>
    </div>
  );
}

