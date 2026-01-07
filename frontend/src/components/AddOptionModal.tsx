import { useState, useEffect, useCallback } from 'react';
import { X, Plus, AlertCircle, Target, Calendar, Hash, DollarSign, Loader2 } from 'lucide-react';
import type { OptionHoldingCreate, OptionType, PositionType } from '../types';
import { getOptionExpirations, getOptionStrikes } from '../services/api';

interface AddOptionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (option: OptionHoldingCreate) => Promise<void>;
}

export function AddOptionModal({ isOpen, onClose, onAdd }: AddOptionModalProps) {
  // Form state
  const [ticker, setTicker] = useState('');
  const [optionType, setOptionType] = useState<OptionType>('call');
  const [positionType, setPositionType] = useState<PositionType>('long');
  const [expiration, setExpiration] = useState('');
  const [strike, setStrike] = useState('');
  const [contracts, setContracts] = useState('1');
  const [premium, setPremium] = useState('');
  const [notes, setNotes] = useState('');
  
  // UI state
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Chain data
  const [expirations, setExpirations] = useState<string[]>([]);
  const [strikes, setStrikes] = useState<number[]>([]);
  const [loadingExpirations, setLoadingExpirations] = useState(false);
  const [loadingStrikes, setLoadingStrikes] = useState(false);

  // Fetch expirations when ticker changes
  const fetchExpirations = useCallback(async (tickerSymbol: string) => {
    if (!tickerSymbol || tickerSymbol.length < 1) {
      setExpirations([]);
      return;
    }
    
    setLoadingExpirations(true);
    try {
      const response = await getOptionExpirations(tickerSymbol);
      setExpirations(response.expirations);
      setExpiration(''); // Reset selection
      setStrikes([]);
      setStrike('');
    } catch (err) {
      console.error('Failed to fetch expirations:', err);
      setExpirations([]);
    } finally {
      setLoadingExpirations(false);
    }
  }, []);

  // Fetch strikes when expiration changes
  const fetchStrikes = useCallback(async (tickerSymbol: string, exp: string) => {
    if (!tickerSymbol || !exp) {
      setStrikes([]);
      return;
    }
    
    setLoadingStrikes(true);
    try {
      const response = await getOptionStrikes(tickerSymbol, exp, optionType);
      setStrikes(response.strikes);
      setStrike(''); // Reset selection
    } catch (err) {
      console.error('Failed to fetch strikes:', err);
      setStrikes([]);
    } finally {
      setLoadingStrikes(false);
    }
  }, [optionType]);

  // Debounce ticker input to avoid too many API calls
  useEffect(() => {
    const timer = setTimeout(() => {
      if (ticker.length >= 1) {
        fetchExpirations(ticker);
      }
    }, 500);
    
    return () => clearTimeout(timer);
  }, [ticker, fetchExpirations]);

  // Fetch strikes when expiration is selected
  useEffect(() => {
    if (expiration) {
      fetchStrikes(ticker, expiration);
    }
  }, [expiration, ticker, fetchStrikes]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    // Validation
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol');
      return;
    }
    
    if (!expiration) {
      setError('Please select an expiration date');
      return;
    }
    
    const strikeNum = parseFloat(strike);
    if (!strike || strikeNum <= 0) {
      setError('Please enter a valid strike price');
      return;
    }
    
    const contractsNum = parseInt(contracts);
    if (contractsNum < 1) {
      setError('Number of contracts must be at least 1');
      return;
    }
    
    const premiumNum = premium ? parseFloat(premium) : undefined;
    if (premiumNum !== undefined && premiumNum < 0) {
      setError('Premium cannot be negative');
      return;
    }

    setLoading(true);
    try {
      await onAdd({
        underlying_ticker: ticker.toUpperCase(),
        option_type: optionType,
        position_type: positionType,
        strike_price: strikeNum,
        expiration_date: expiration,
        contracts: contractsNum,
        premium_per_contract: premiumNum ?? null,
        notes: notes.trim() || null,
      });
      resetForm();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add option');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setTicker('');
    setOptionType('call');
    setPositionType('long');
    setExpiration('');
    setStrike('');
    setContracts('1');
    setPremium('');
    setNotes('');
    setError('');
    setExpirations([]);
    setStrikes([]);
  };

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      resetForm();
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
      <div className="glass-card p-6 w-full max-w-lg relative z-10 fade-in max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-bold gradient-text">Add Option Position</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-white/70" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="space-y-4">
            {/* Ticker */}
            <div>
              <label className="block text-white/70 text-sm mb-2">Underlying Ticker</label>
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL, SPY"
                className="w-full"
                disabled={loading}
                autoFocus
              />
            </div>

            {/* Option Type & Position Type */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-white/70 text-sm mb-2">Option Type</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setOptionType('call')}
                    className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                      optionType === 'call'
                        ? 'bg-green-500/20 text-green-400 border border-green-500/40'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:border-white/20'
                    }`}
                    disabled={loading}
                  >
                    Call
                  </button>
                  <button
                    type="button"
                    onClick={() => setOptionType('put')}
                    className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                      optionType === 'put'
                        ? 'bg-red-500/20 text-red-400 border border-red-500/40'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:border-white/20'
                    }`}
                    disabled={loading}
                  >
                    Put
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-white/70 text-sm mb-2">Position</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setPositionType('long')}
                    className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                      positionType === 'long'
                        ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/40'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:border-white/20'
                    }`}
                    disabled={loading}
                  >
                    Long
                  </button>
                  <button
                    type="button"
                    onClick={() => setPositionType('short')}
                    className={`flex-1 py-2 px-3 rounded-lg font-medium transition-all ${
                      positionType === 'short'
                        ? 'bg-accent-purple/20 text-accent-purple border border-accent-purple/40'
                        : 'bg-white/5 text-white/60 border border-white/10 hover:border-white/20'
                    }`}
                    disabled={loading}
                  >
                    Short
                  </button>
                </div>
              </div>
            </div>

            {/* Expiration */}
            <div>
              <label className="block text-white/70 text-sm mb-2">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  Expiration Date
                  {loadingExpirations && <Loader2 className="w-3 h-3 animate-spin" />}
                </div>
              </label>
              {expirations.length > 0 ? (
                <select
                  value={expiration}
                  onChange={(e) => setExpiration(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-accent-cyan/50"
                  disabled={loading}
                >
                  <option value="">Select expiration...</option>
                  {expirations.map((exp) => (
                    <option key={exp} value={exp}>
                      {new Date(exp + 'T00:00:00').toLocaleDateString('en-US', { 
                        weekday: 'short', 
                        month: 'short', 
                        day: 'numeric', 
                        year: 'numeric' 
                      })}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="date"
                  value={expiration}
                  onChange={(e) => setExpiration(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-accent-cyan/50"
                  disabled={loading}
                />
              )}
              <p className="text-white/30 text-xs mt-1">
                {ticker && expirations.length === 0 && !loadingExpirations 
                  ? 'Enter ticker above to load available expirations, or enter date manually'
                  : ''}
              </p>
            </div>

            {/* Strike Price */}
            <div>
              <label className="block text-white/70 text-sm mb-2">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4" />
                  Strike Price
                  {loadingStrikes && <Loader2 className="w-3 h-3 animate-spin" />}
                </div>
              </label>
              {strikes.length > 0 ? (
                <select
                  value={strike}
                  onChange={(e) => setStrike(e.target.value)}
                  className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-accent-cyan/50"
                  disabled={loading}
                >
                  <option value="">Select strike...</option>
                  {strikes.map((s) => (
                    <option key={s} value={s}>
                      ${s.toFixed(2)}
                    </option>
                  ))}
                </select>
              ) : (
                <div className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4 text-white/40 flex-shrink-0" />
                  <input
                    type="number"
                    value={strike}
                    onChange={(e) => setStrike(e.target.value)}
                    placeholder="0.00"
                    min="0.01"
                    step="0.50"
                    className="flex-1"
                    disabled={loading}
                  />
                </div>
              )}
            </div>

            {/* Contracts and Premium */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-white/70 text-sm mb-2">
                  <div className="flex items-center gap-2">
                    <Hash className="w-4 h-4" />
                    Contracts
                  </div>
                </label>
                <input
                  type="number"
                  value={contracts}
                  onChange={(e) => setContracts(e.target.value)}
                  placeholder="1"
                  min="1"
                  step="1"
                  className="w-full"
                  disabled={loading}
                />
              </div>

              <div>
                <label className="block text-white/70 text-sm mb-2">
                  <div className="flex items-center gap-2">
                    <DollarSign className="w-4 h-4" />
                    Premium/Contract
                    <span className="text-white/40">(opt)</span>
                  </div>
                </label>
                <input
                  type="number"
                  value={premium}
                  onChange={(e) => setPremium(e.target.value)}
                  placeholder="0.00"
                  min="0"
                  step="0.01"
                  className="w-full"
                  disabled={loading}
                />
              </div>
            </div>

            {/* Position type hint */}
            <div className="p-3 rounded-lg bg-white/5 border border-white/10">
              <p className="text-white/50 text-xs">
                {positionType === 'long' ? (
                  <>
                    <span className="text-accent-cyan font-medium">Long {optionType}</span>: You paid premium to buy this option.
                    {optionType === 'call' 
                      ? ' Profits when the stock rises above strike + premium.'
                      : ' Profits when the stock falls below strike - premium.'}
                  </>
                ) : (
                  <>
                    <span className="text-accent-purple font-medium">Short {optionType}</span>: You received premium to sell this option.
                    {optionType === 'call' 
                      ? ' Profits if stock stays below strike at expiration (covered call).'
                      : ' Profits if stock stays above strike at expiration (cash-secured put).'}
                  </>
                )}
              </p>
            </div>

            {/* Notes */}
            <div>
              <label className="block text-white/70 text-sm mb-2">
                Notes <span className="text-white/40">(optional)</span>
              </label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="e.g., Earnings play, hedge for AAPL position"
                className="w-full"
                disabled={loading}
              />
            </div>

            {/* Error display */}
            {error && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Buttons */}
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
                    Add Option
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

