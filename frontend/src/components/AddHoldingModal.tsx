import { useState, useEffect } from 'react';
import { X, Plus, AlertCircle, Percent, DollarSign } from 'lucide-react';

interface AddHoldingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (ticker: string, allocation: number) => Promise<void>;
  currentAllocation: number;
  portfolioTotalValue: number;
}

export function AddHoldingModal({ isOpen, onClose, onAdd, currentAllocation, portfolioTotalValue }: AddHoldingModalProps) {
  const [ticker, setTicker] = useState('');
  const [allocation, setAllocation] = useState('');
  const [amount, setAmount] = useState('');
  const [activeInput, setActiveInput] = useState<'percent' | 'amount' | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const remainingAllocation = 100 - currentAllocation;
  const remainingAmount = portfolioTotalValue * (remainingAllocation / 100);

  // Sync allocation and amount based on which one was edited
  useEffect(() => {
    if (activeInput === 'percent' && allocation !== '') {
      const pct = parseFloat(allocation);
      if (!isNaN(pct) && portfolioTotalValue > 0) {
        const calculatedAmount = (pct / 100) * portfolioTotalValue;
        setAmount(calculatedAmount.toFixed(2));
      }
    } else if (activeInput === 'amount' && amount !== '') {
      const amt = parseFloat(amount);
      if (!isNaN(amt) && portfolioTotalValue > 0) {
        const calculatedPct = (amt / portfolioTotalValue) * 100;
        setAllocation(calculatedPct.toFixed(2));
      }
    }
  }, [allocation, amount, activeInput, portfolioTotalValue]);

  const handleAllocationChange = (value: string) => {
    setActiveInput('percent');
    setAllocation(value);
    if (value === '') {
      setAmount('');
    }
  };

  const handleAmountChange = (value: string) => {
    setActiveInput('amount');
    setAmount(value);
    if (value === '') {
      setAllocation('');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    const allocationNum = parseFloat(allocation) || 0;
    
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }
    
    // Allow 0% allocation
    if (allocationNum < 0) {
      setError('Allocation cannot be negative');
      return;
    }
    
    if (allocationNum > remainingAllocation) {
      setError(`Allocation cannot exceed ${remainingAllocation.toFixed(1)}%`);
      return;
    }

    setLoading(true);
    try {
      await onAdd(ticker.toUpperCase(), allocationNum);
      setTicker('');
      setAllocation('');
      setAmount('');
      setActiveInput(null);
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
      setAllocation('');
      setAmount('');
      setActiveInput(null);
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
                placeholder="e.g., AAPL, ONDS"
                className="w-full"
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-white/70 text-sm mb-2">
                Allocation
                <span className="text-white/40 ml-2">
                  (optional)
                </span>
              </label>
              <div className="space-y-3">
                {/* Percentage Input */}
                <div>
                  <div className="flex items-center gap-2">
                    <Percent className="w-4 h-4 text-white/40 flex-shrink-0" />
                    <input
                      type="number"
                      value={allocation}
                      onChange={(e) => handleAllocationChange(e.target.value)}
                      placeholder="0"
                      min="0"
                      max={remainingAllocation}
                      step="0.1"
                      className="flex-1"
                      disabled={loading}
                    />
                    <span className="text-white/30 text-xs whitespace-nowrap">
                      max {remainingAllocation.toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                {/* Amount Input */}
                <div>
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-white/40 flex-shrink-0" />
                    <input
                      type="number"
                      value={amount}
                      onChange={(e) => handleAmountChange(e.target.value)}
                      placeholder="0.00"
                      min="0"
                      max={remainingAmount}
                      step="0.01"
                      className="flex-1"
                      disabled={loading}
                    />
                    <span className="text-white/30 text-xs whitespace-nowrap">
                      max ${remainingAmount.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                    </span>
                  </div>
                </div>
              </div>
              <p className="text-white/30 text-xs mt-2">
                Enter either — the other is calculated automatically
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

