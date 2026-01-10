import { TrendingUp, TrendingDown, PieChart, Wallet } from 'lucide-react';
import type { Portfolio } from '../types';
import { LoadingSpinner } from './LoadingValue';

interface PortfolioSummaryProps {
  portfolio: Portfolio;
}

export function PortfolioSummary({ portfolio }: PortfolioSummaryProps) {
  // Check if all holdings have loaded their stock data (have current_price)
  const allDataLoaded = portfolio.holdings.length === 0 || 
    portfolio.holdings.every(h => h.current_price !== undefined && h.current_price !== null);
  
  // Total market value of all holdings (sum of shares × current_price)
  const totalMarketValue = portfolio.total_market_value ?? 0;
  // Total cost basis (sum of shares × avg_cost)
  const totalCostBasis = portfolio.total_cost_basis ?? 0;
  
  // Count positions with actual shares
  const positionCount = portfolio.holdings.filter(h => h.shares > 0).length;
  
  // Only show gain/loss if we have cost basis data
  const hasGainLossData = portfolio.total_gain_loss !== undefined && portfolio.total_gain_loss !== null;
  const isPositive = (portfolio.total_gain_loss_pct ?? 0) >= 0;
  
  return (
    <div className="mb-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {/* Total Market Value */}
      <div className="glass-card p-5 fade-in fade-in-delay-1">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent-cyan/20">
            <PieChart className="w-5 h-5 text-accent-cyan" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-white/50 text-sm">Total Market Value</span>
            {!allDataLoaded && <LoadingSpinner size="sm" />}
          </div>
        </div>
        {allDataLoaded ? (
          <div>
            <p className="text-2xl font-bold text-white">
              ${totalMarketValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </p>
            <span className="text-white/40 text-sm">{positionCount} position{positionCount !== 1 ? 's' : ''}</span>
          </div>
        ) : (
          <p className="text-2xl font-bold text-white/30">Loading...</p>
        )}
      </div>

      {/* Cost Basis */}
      <div className="glass-card p-5 fade-in fade-in-delay-2">
        <div className="flex items-center gap-3 mb-3">
          <div className="p-2 rounded-lg bg-accent-purple/20">
            <Wallet className="w-5 h-5 text-accent-purple" />
          </div>
          <span className="text-white/50 text-sm">Cost Basis</span>
        </div>
        <div>
          {totalCostBasis > 0 ? (
            <>
              <p className="text-2xl font-bold text-accent-purple">
                ${totalCostBasis.toLocaleString('en-US', { minimumFractionDigits: 2 })}
              </p>
              <span className="text-white/40 text-sm">Total invested</span>
            </>
          ) : (
            <>
              <p className="text-lg text-white/40">Set avg costs</p>
              <span className="text-white/30 text-sm">to track cost basis</span>
            </>
          )}
        </div>
      </div>

      {/* Portfolio Gain/Loss */}
      <div className="glass-card p-5 fade-in fade-in-delay-3">
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
            <span className="text-white/50 text-sm">Total Gain/Loss</span>
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
            <p className="text-lg text-white/40">Set avg costs</p>
            <span className="text-white/30 text-sm">to track gain/loss</span>
          </div>
        ) : (
          <p className="text-2xl font-bold text-white/30">Loading...</p>
        )}
      </div>
    </div>
    </div>
  );
}
