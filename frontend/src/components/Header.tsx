import { useState } from 'react';
import { TrendingUp, Copy, Check, Settings, Wrench } from 'lucide-react';
import type { Holding } from '../types';
import { SettingsModal } from './SettingsModal';

interface HeaderProps {
  totalValue: number;
  lastUpdated?: Date;
  isDataReady: boolean;
  holdings?: Holding[];
}

export function Header({ totalValue, lastUpdated, isDataReady, holdings = [] }: HeaderProps) {
  const [copied, setCopied] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const handleCopyHoldings = async () => {
    if (!holdings.length) return;
    
    // Create plaintext list of tickers (one per line)
    const tickerList = holdings.map(h => h.ticker).join('\n');
    
    try {
      await navigator.clipboard.writeText(tickerList);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy holdings:', err);
    }
  };

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
      
      <div className="flex items-center gap-4">
        {/* Settings Button */}
        <button
          onClick={() => setShowSettings(true)}
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200 group"
          title="Settings & Cache Management"
        >
          <Wrench className="w-5 h-5 text-white/50 group-hover:text-white/80" />
        </button>

        {/* Calibration Link */}
        <a
          href="/calibration"
          className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-accent-cyan/30 transition-all duration-200 group"
          title="Walk-Forward Optimization"
        >
          <Settings className="w-5 h-5 text-white/50 group-hover:text-accent-cyan" />
        </a>

        {/* Copy Holdings Button */}
        {holdings.length > 0 && (
          <button
            onClick={handleCopyHoldings}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all duration-200 group"
            title="Copy all tickers to clipboard"
          >
            {copied ? (
              <Check className="w-5 h-5 text-green-400" />
            ) : (
              <Copy className="w-5 h-5 text-white/50 group-hover:text-white/80" />
            )}
          </button>
        )}
        
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
      </div>

      {/* Settings Modal */}
      <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </header>
  );
}

