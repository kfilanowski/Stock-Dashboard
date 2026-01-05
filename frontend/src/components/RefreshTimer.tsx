import { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';

interface RefreshTimerProps {
  lastFetched: Date | null;
  refreshInterval: number; // in milliseconds
  isRefreshing: boolean;
}

export function RefreshTimer({ lastFetched, refreshInterval, isRefreshing }: RefreshTimerProps) {
  const [secondsUntilRefresh, setSecondsUntilRefresh] = useState(refreshInterval / 1000);
  
  useEffect(() => {
    if (!lastFetched) return;
    
    const updateTimer = () => {
      const elapsed = Date.now() - lastFetched.getTime();
      const remaining = Math.max(0, Math.ceil((refreshInterval - elapsed) / 1000));
      setSecondsUntilRefresh(remaining);
    };
    
    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    
    return () => clearInterval(interval);
  }, [lastFetched, refreshInterval]);
  
  if (isRefreshing) {
    return (
      <div className="flex items-center gap-2 text-white/40 text-sm">
        <div className="w-3 h-3 border border-white/20 border-t-accent-cyan rounded-full animate-spin" />
        <span>Refreshing...</span>
      </div>
    );
  }
  
  return (
    <div className="flex items-center gap-2 text-white/40 text-sm">
      <Clock className="w-3 h-3" />
      <span>
        Next refresh in {secondsUntilRefresh}s
      </span>
    </div>
  );
}

