import { RefreshCw, TrendingUp } from 'lucide-react';

interface HeaderProps {
  totalValue: number;
  onRefresh: () => void;
  loading: boolean;
  lastUpdated?: Date;
}

export function Header({ totalValue, onRefresh, loading, lastUpdated }: HeaderProps) {
  return (
    <header className="flex items-center justify-between mb-8 fade-in">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-gradient-to-br from-accent-cyan/20 to-accent-purple/20 border border-white/10">
          <TrendingUp className="w-8 h-8 text-accent-cyan" />
        </div>
        <div>
          <h1 className="text-2xl font-bold gradient-text">Portfolio Dashboard</h1>
          <p className="text-white/50 text-sm">
            {lastUpdated ? `Last updated: ${lastUpdated.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}` : 'Loading...'}
          </p>
        </div>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="text-right">
          <p className="text-white/50 text-sm">Total Portfolio Value</p>
          <p className="text-2xl font-bold text-white">
            ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        
        <button
          onClick={onRefresh}
          disabled={loading}
          className="p-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all duration-200 disabled:opacity-50"
        >
          <RefreshCw className={`w-5 h-5 text-white/70 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </header>
  );
}

