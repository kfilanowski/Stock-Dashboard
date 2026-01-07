import { useMemo } from 'react';
import { Trash2, TrendingUp, TrendingDown, Clock, Target, AlertTriangle } from 'lucide-react';
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

      {/* Return Summary - Make this prominent */}
      <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-white/5 mb-3 border border-white/5">
        <div className="flex gap-4">
          <div title="Current market value of your position">
            <span className="text-white/40 text-xs block">Current Value</span>
            <span className="text-white font-medium">{formatCurrency(option.position_value)}</span>
          </div>
        </div>
        <div className="text-right" title={isLong ? "Current Value - What You Paid" : "What You Received - Current Liability"}>
          <span className="text-white/40 text-xs block">Your Return</span>
          <div className="flex items-center gap-2">
            <span className={`font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {formatCurrency(option.gain_loss)}
            </span>
            <span className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              ({formatPercent(option.gain_loss_pct)})
            </span>
          </div>
        </div>
      </div>

      {/* Greeks row */}
      {option.greeks && (
        <div className="flex justify-between py-2 px-3 rounded-lg bg-white/5 mb-3">
          <div className="flex gap-4">
            <GreekBadge label="Δ" value={option.greeks.delta} multiplier={100} suffix="%" />
            <GreekBadge label="Γ" value={option.greeks.gamma} multiplier={1} decimals={4} />
            <GreekBadge label="Θ" value={option.greeks.theta} multiplier={100} prefix="$" isNegativeGood={isLong} />
            <GreekBadge label="V" value={option.greeks.vega} multiplier={1} prefix="$" />
          </div>
          {option.implied_volatility && (
            <div className="text-right">
              <span className="text-white/40 text-xs block">IV</span>
              <span className="text-accent-purple font-medium">{option.implied_volatility.toFixed(1)}%</span>
            </div>
          )}
        </div>
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

