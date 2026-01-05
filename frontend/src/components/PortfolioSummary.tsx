import { useState } from 'react';
import { TrendingUp, TrendingDown, DollarSign, PieChart, Pencil, Check, X } from 'lucide-react';
import type { Portfolio } from '../types';
import { LoadingValue, LoadingSpinner } from './LoadingValue';

interface PortfolioSummaryProps {
  portfolio: Portfolio;
  isRefreshing?: boolean;
  onUpdateValue?: (value: number) => Promise<void>;
}

export function PortfolioSummary({ portfolio, isRefreshing = false, onUpdateValue }: PortfolioSummaryProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(portfolio.total_value.toString());
  const [isSaving, setIsSaving] = useState(false);

  const isPositive = (portfolio.total_gain_loss_pct ?? 0) >= 0;
  const totalAllocated = portfolio.holdings.reduce((sum, h) => sum + h.allocation_pct, 0);

  const handleStartEdit = () => {
    setEditValue(portfolio.total_value.toString());
    setIsEditing(true);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditValue(portfolio.total_value.toString());
  };

  const handleSave = async () => {
    const numValue = parseFloat(editValue);
    if (isNaN(numValue) || numValue <= 0 || !onUpdateValue) return;
    
    setIsSaving(true);
    try {
      await onUpdateValue(numValue);
      setIsEditing(false);
    } finally {
      setIsSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSave();
    } else if (e.key === 'Escape') {
      handleCancel();
    }
  };
  
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
      {/* Base Investment */}
      <div className="glass-card p-5 fade-in fade-in-delay-1 group">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent-purple/20">
            <DollarSign className="w-5 h-5 text-accent-purple" />
          </div>
          <span className="text-white/50 text-sm">Base Investment</span>
          {!isEditing && onUpdateValue && (
            <button
              onClick={handleStartEdit}
              className="ml-auto p-1.5 rounded-lg bg-white/0 hover:bg-white/10 opacity-0 group-hover:opacity-100 transition-all"
              title="Edit base investment"
            >
              <Pencil className="w-3.5 h-3.5 text-white/50 hover:text-white/80" />
            </button>
          )}
        </div>
        {isEditing ? (
          <div className="flex items-center gap-2">
            <div className="flex items-center flex-1 bg-white/5 rounded-lg px-3 py-1.5 border border-white/20 focus-within:border-accent-purple/50">
              <span className="text-white/50">$</span>
              <input
                type="number"
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onKeyDown={handleKeyDown}
                className="bg-transparent border-none w-full p-0 ml-1 text-xl font-bold text-white focus:outline-none focus:ring-0"
                min="1"
                step="100"
                autoFocus
                disabled={isSaving}
              />
            </div>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="p-1.5 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors disabled:opacity-50"
              title="Save"
            >
              <Check className="w-4 h-4" />
            </button>
            <button
              onClick={handleCancel}
              disabled={isSaving}
              className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 transition-colors disabled:opacity-50"
              title="Cancel"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <p className="text-2xl font-bold text-white">
            ${portfolio.total_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        )}
      </div>

      {/* Current Value */}
      <div className="glass-card p-5 fade-in fade-in-delay-2">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-lg ${isPositive ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
            {isPositive ? (
              <TrendingUp className="w-5 h-5 text-green-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50 text-sm">Current Value</span>
            {isRefreshing && <LoadingSpinner size="sm" />}
          </div>
        </div>
        <LoadingValue loading={isRefreshing} size="lg">
          <p className="text-2xl font-bold text-white">
            ${(portfolio.current_total_value ?? portfolio.total_value).toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </LoadingValue>
      </div>

      {/* Total Gain/Loss */}
      <div className="glass-card p-5 fade-in fade-in-delay-3">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-lg ${isPositive ? 'bg-green-500/20' : 'bg-red-500/20'}`}>
            {isPositive ? (
              <TrendingUp className="w-5 h-5 text-green-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50 text-sm">Total Gain/Loss</span>
            {isRefreshing && <LoadingSpinner size="sm" />}
          </div>
        </div>
        <LoadingValue loading={isRefreshing} size="lg">
          <div className="flex items-baseline gap-2">
            <p className={`text-2xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {isPositive ? '+' : ''}{(portfolio.total_gain_loss_pct ?? 0).toFixed(2)}%
            </p>
            <span className={`text-sm ${isPositive ? 'text-green-400/70' : 'text-red-400/70'}`}>
              ({isPositive ? '+' : ''}${(portfolio.total_gain_loss ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })})
            </span>
          </div>
        </LoadingValue>
      </div>

      {/* Allocation */}
      <div className="glass-card p-5 fade-in fade-in-delay-4">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent-cyan/20">
            <PieChart className="w-5 h-5 text-accent-cyan" />
          </div>
          <span className="text-white/50 text-sm">Allocated</span>
        </div>
        <div className="flex items-baseline gap-2">
          <p className="text-2xl font-bold text-white">{totalAllocated.toFixed(1)}%</p>
          <span className="text-white/50 text-sm">of portfolio</span>
        </div>
        <div className="mt-2 h-2 bg-white/10 rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-accent-cyan to-accent-purple rounded-full transition-all duration-500"
            style={{ width: `${Math.min(totalAllocated, 100)}%` }}
          />
        </div>
      </div>
    </div>
  );
}
