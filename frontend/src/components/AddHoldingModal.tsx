import { useState, useEffect } from 'react';
import { X, Plus, AlertCircle, Layers, DollarSign } from 'lucide-react';

interface AddHoldingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (ticker: string, shares: number, avgCost?: number) => Promise<void>;
}

export function AddHoldingModal({ isOpen, onClose, onAdd }: AddHoldingModalProps) {
  const [ticker, setTicker] = useState('');
  const [shares, setShares] = useState('');
  const [avgCost, setAvgCost] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }
    
    const sharesNum = parseFloat(shares) || 0;
    if (sharesNum < 0) {
      setError('Shares cannot be negative');
      return;
    }
    
    const avgCostNum = avgCost ? parseFloat(avgCost) : undefined;
    if (avgCostNum !== undefined && avgCostNum <= 0) {
      setError('Average cost must be greater than 0');
      return;
    }

    setLoading(true);
    try {
      await onAdd(ticker.toUpperCase(), sharesNum, avgCostNum);
      setTicker('');
      setShares('');
      setAvgCost('');
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add holding');
    } finally {
      setLoading(false);
    }
  };

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      setTicker('');
      setShares('');
      setAvgCost('');
      setError('');
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="glass-card p-6 w-full max-w-md relative z-10 fade-in">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold gradient-text">Add Stock</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-white/70" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label className="block text-white/70 text-sm mb-2">Ticker Symbol</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL, MSFT"
                className="w-full"
                disabled={loading}
                autoFocus
              />
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-2">
                Number of Shares
                <span className="text-white/40 ml-2">(optional)</span>
              </label>
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-white/40 flex-shrink-0" />
                <input
                  type="number"
                  value={shares}
                  onChange={(e) => setShares(e.target.value)}
                  placeholder="0"
                  min="0"
                  step="0.0001"
                  className="flex-1"
                  disabled={loading}
                />
              </div>
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-2">
                Average Cost per Share
                <span className="text-white/40 ml-2">(optional)</span>
              </label>
              <div className="flex items-center gap-2">
                <DollarSign className="w-4 h-4 text-white/40 flex-shrink-0" />
                <input
                  type="number"
                  value={avgCost}
                  onChange={(e) => setAvgCost(e.target.value)}
                  placeholder="0.00"
                  min="0.01"
                  step="0.01"
                  className="flex-1"
                  disabled={loading}
                />
              </div>
              <p className="text-white/30 text-xs mt-2">
                Used to calculate your gain/loss on this position
              </p>
            </div>

            {error && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            <div className="flex gap-3 pt-2">
              <button
                type="button"
                onClick={onClose}
                className="btn-secondary flex-1"
                disabled={loading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn-primary flex-1 flex items-center justify-center gap-2"
                disabled={loading}
              >
                {loading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  <>
                    <Plus className="w-5 h-5" />
                    Add Stock
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
