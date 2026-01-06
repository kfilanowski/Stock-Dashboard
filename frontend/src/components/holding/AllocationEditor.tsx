import { useState } from 'react';
import { Edit2, Check, X } from 'lucide-react';

interface AllocationEditorProps {
  holdingId: number;
  currentAllocation: number;
  portfolioTotalValue: number;
  currentTotalAllocation: number;
  onSave: (id: number, allocation: number) => Promise<void>;
}

export function AllocationEditor({
  holdingId,
  currentAllocation,
  portfolioTotalValue,
  currentTotalAllocation,
  onSave
}: AllocationEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [allocation, setAllocation] = useState(currentAllocation.toString());
  const [amount, setAmount] = useState('');
  const [activeInput, setActiveInput] = useState<'percent' | 'amount' | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  const maxAllocation = 100 - currentTotalAllocation + currentAllocation;
  const maxAmount = portfolioTotalValue * (maxAllocation / 100);

  const handleAllocationChange = (value: string) => {
    setActiveInput('percent');
    setAllocation(value);
    if (value !== '' && portfolioTotalValue > 0) {
      const pct = parseFloat(value);
      if (!isNaN(pct)) {
        const calculatedAmount = (pct / 100) * portfolioTotalValue;
        setAmount(calculatedAmount.toFixed(2));
      }
    } else {
      setAmount('');
    }
  };

  const handleAmountChange = (value: string) => {
    setActiveInput('amount');
    setAmount(value);
    if (value !== '' && portfolioTotalValue > 0) {
      const amt = parseFloat(value);
      if (!isNaN(amt)) {
        const calculatedPct = (amt / portfolioTotalValue) * 100;
        setAllocation(calculatedPct.toFixed(2));
      }
    } else {
      setAllocation('');
    }
  };

  const handleStartEditing = () => {
    setAllocation(currentAllocation.toString());
    setAmount(((currentAllocation / 100) * portfolioTotalValue).toFixed(2));
    setActiveInput(null);
    setIsEditing(true);
  };

  const handleSave = async () => {
    setError('');
    const value = parseFloat(allocation) || 0;
    
    if (value < 0) {
      setError('Cannot be negative');
      return;
    }
    
    if (value > maxAllocation) {
      setError(`Max: ${maxAllocation.toFixed(1)}%`);
      return;
    }

    setSaving(true);
    try {
      await onSave(holdingId, value);
      setIsEditing(false);
      setActiveInput(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update');
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setAllocation(currentAllocation.toString());
    setAmount(((currentAllocation / 100) * portfolioTotalValue).toFixed(2));
    setActiveInput(null);
    setError('');
    setIsEditing(false);
  };

  if (!isEditing) {
    return (
      <div className="flex items-center gap-2">
        <p className="text-white/50 text-sm">{currentAllocation}% allocation</p>
        <button
          onClick={handleStartEditing}
          className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
          title="Edit allocation"
        >
          <Edit2 className="w-3 h-3" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 mt-1">
      <div className="flex items-center gap-1.5">
        <input
          type="number"
          value={allocation}
          onChange={(e) => handleAllocationChange(e.target.value)}
          className="w-32 px-2 py-1 text-sm bg-white/10 border border-white/20 rounded focus:border-accent-cyan/50 focus:outline-none"
          min="0"
          max={maxAllocation}
          step="0.1"
          placeholder="0"
          autoFocus={activeInput !== 'amount'}
        />
        <span className="text-white/50 text-xs">%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-white/50 text-xs">$</span>
        <input
          type="number"
          value={amount}
          onChange={(e) => handleAmountChange(e.target.value)}
          className="w-36 px-2 py-1 text-sm bg-white/10 border border-white/20 rounded focus:border-accent-cyan/50 focus:outline-none"
          min="0"
          max={maxAmount}
          step="0.01"
          placeholder="0.00"
          autoFocus={activeInput === 'amount'}
        />
      </div>
      <div className="flex items-center gap-1">
        <button
          onClick={handleSave}
          disabled={saving}
          className="px-2 py-1 rounded bg-green-500/20 hover:bg-green-500/30 text-green-400 text-xs flex items-center gap-1"
        >
          <Check className="w-3 h-3" />
          Save
        </button>
        <button
          onClick={handleCancel}
          className="px-2 py-1 rounded bg-white/10 hover:bg-white/20 text-white/70 text-xs flex items-center gap-1"
        >
          <X className="w-3 h-3" />
          Cancel
        </button>
      </div>
      {error && <p className="text-red-400 text-xs">{error}</p>}
    </div>
  );
}

