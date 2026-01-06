import { TrendingUp } from 'lucide-react';

interface HeaderProps {
  totalValue: number;
  lastUpdated?: Date;
  isDataReady: boolean;
}

export function Header({ totalValue, lastUpdated, isDataReady }: HeaderProps) {
  return (
    <header className="flex items-center justify-between mb-8 fade-in">
      <div className="flex items-center gap-4">
        <div className="p-3 rounded-xl bg-gradient-to-br from-accent-cyan/20 to-accent-purple/20 border border-white/10">
          <TrendingUp className="w-8 h-8 text-accent-cyan" />
        </div>
        <div>
          <h1 className="text-2xl font-bold gradient-text">Portfolio Dashboard</h1>
          <p className="text-white/50 text-sm">
            {lastUpdated 
              ? `Last updated: ${lastUpdated.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })}`
              : 'Loading...'}
          </p>
        </div>
      </div>
      
      <div className="text-right">
        <p className="text-white/50 text-sm">Total Portfolio Value</p>
        {isDataReady ? (
          <p className="text-2xl font-bold text-white">
            ${totalValue.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        ) : (
          <p className="text-2xl font-bold text-white/30">Loading...</p>
        )}
      </div>
    </header>
  );
}

