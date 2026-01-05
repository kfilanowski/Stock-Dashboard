import { useState } from 'react';
import { TrendingUp, TrendingDown, DollarSign, PieChart, Pencil, Check, X, Wallet } from 'lucide-react';
import type { Portfolio } from '../types';
import { LoadingSpinner } from './LoadingValue';

interface PortfolioSummaryProps {
  portfolio: Portfolio;
  onUpdateValue?: (value: number) => Promise<void>;
}

export function PortfolioSummary({ portfolio, onUpdateValue }: PortfolioSummaryProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(portfolio.total_value.toString());
  const [isSaving, setIsSaving] = useState(false);

  const totalAllocated = portfolio.holdings.reduce((sum, h) => sum + h.allocation_pct, 0);
  const buyingPower = portfolio.total_value * ((100 - totalAllocated) / 100);
  const currentInvested = portfolio.total_value * (totalAllocated / 100);
  
  // Check if all holdings have loaded their stock data (have current_price)
  const allDataLoaded = portfolio.holdings.length === 0 || 
    portfolio.holdings.every(h => h.current_price !== undefined && h.current_price !== null);
  
  // Only show gain/loss if we have investment prices for all holdings
  const hasGainLossData = portfolio.total_gain_loss !== undefined && portfolio.total_gain_loss !== null;
  const isPositive = (portfolio.total_gain_loss_pct ?? 0) >= 0;

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
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
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

      {/* Current Invested */}
      <div className="glass-card p-5 fade-in fade-in-delay-2">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent-cyan/20">
            <PieChart className="w-5 h-5 text-accent-cyan" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50 text-sm">Current Invested</span>
            {!allDataLoaded && <LoadingSpinner size="sm" />}
          </div>
        </div>
        {allDataLoaded ? (
          <div>
            <p className="text-2xl font-bold text-white">
              ${currentInvested.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </p>
            <span className="text-white/40 text-sm">{totalAllocated.toFixed(1)}% allocated</span>
          </div>
        ) : (
          <p className="text-2xl font-bold text-white/30">Loading...</p>
        )}
      </div>

      {/* Buying Power */}
      <div className="glass-card p-5 fade-in fade-in-delay-3">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-emerald-500/20">
            <Wallet className="w-5 h-5 text-emerald-400" />
          </div>
          <span className="text-white/50 text-sm">Buying Power</span>
        </div>
        <div>
          <p className="text-2xl font-bold text-emerald-400">
            ${buyingPower.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
          <span className="text-white/40 text-sm">{(100 - totalAllocated).toFixed(1)}% available</span>
        </div>
      </div>

      {/* Portfolio Gain/Loss */}
      <div className="glass-card p-5 fade-in fade-in-delay-4">
        <div className="flex items-center gap-3 mb-3">
          <div className={`p-2 rounded-lg ${hasGainLossData && isPositive ? 'bg-green-500/20' : hasGainLossData ? 'bg-red-500/20' : 'bg-white/10'}`}>
            {hasGainLossData && isPositive ? (
              <TrendingUp className="w-5 h-5 text-green-400" />
            ) : hasGainLossData ? (
              <TrendingDown className="w-5 h-5 text-red-400" />
            ) : (
              <TrendingUp className="w-5 h-5 text-white/30" />
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50 text-sm">Portfolio Gain/Loss</span>
            {!allDataLoaded && <LoadingSpinner size="sm" />}
          </div>
        </div>
        {hasGainLossData && allDataLoaded ? (
          <div className="flex items-baseline gap-2">
            <p className={`text-2xl font-bold ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
              {isPositive ? '+' : ''}{(portfolio.total_gain_loss_pct ?? 0).toFixed(2)}%
            </p>
            <span className={`text-sm ${isPositive ? 'text-green-400/70' : 'text-red-400/70'}`}>
              ({isPositive ? '+' : ''}${(portfolio.total_gain_loss ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })})
            </span>
          </div>
        ) : allDataLoaded ? (
          <div>
            <p className="text-lg text-white/40">Set investment dates</p>
            <span className="text-white/30 text-sm">to track gain/loss</span>
          </div>
        ) : (
          <p className="text-2xl font-bold text-white/30">Loading...</p>
        )}
      </div>
    </div>
  );
}
