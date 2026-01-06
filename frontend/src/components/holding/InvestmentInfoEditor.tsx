import { useState } from 'react';
import { Calendar, DollarSign, Edit2, Check, X } from 'lucide-react';

interface InvestmentInfoEditorProps {
  holdingId: number;
  investmentDate?: string | null;
  investmentPrice?: number | null;
  currentPrice?: number;
  gainLossPct?: number | null;
  onSave: (id: number, data: { investment_date?: string; investment_price?: number }) => Promise<void>;
}

export function InvestmentInfoEditor({
  holdingId,
  investmentDate,
  investmentPrice,
  currentPrice,
  gainLossPct,
  onSave
}: InvestmentInfoEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [date, setDate] = useState(
    investmentDate ? investmentDate.split('T')[0] : ''
  );
  const [price, setPrice] = useState(
    investmentPrice?.toString() ?? ''
  );
  const [saving, setSaving] = useState(false);

  const formatInvestmentDate = (dateStr: string | null | undefined): string => {
    if (!dateStr) return '';
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    } catch {
      return '';
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const data: { investment_date?: string; investment_price?: number } = {};
      
      if (date) {
        data.investment_date = new Date(date).toISOString();
      }
      if (price) {
        const priceNum = parseFloat(price);
        if (!isNaN(priceNum) && priceNum > 0) {
          data.investment_price = priceNum;
        }
      }
      
      await onSave(holdingId, data);
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to save investment info:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setDate(investmentDate ? investmentDate.split('T')[0] : '');
    setPrice(investmentPrice?.toString() ?? '');
    setIsEditing(false);
  };

  if (isEditing) {
    return (
      <div className="mb-3 p-2.5 rounded-lg bg-white/[0.03] border border-white/5">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5 text-white/40" />
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
              max={new Date().toISOString().split('T')[0]}
            />
          </div>
          <div className="flex items-center gap-2">
            <DollarSign className="w-3.5 h-3.5 text-white/40" />
            <input
              type="number"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              placeholder="Price at investment"
              className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
              min="0.01"
              step="0.01"
            />
            {currentPrice && (
              <button
                type="button"
                onClick={() => setPrice(currentPrice.toFixed(2))}
                className="px-1.5 py-1 text-[10px] rounded bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/70 whitespace-nowrap"
                title="Use current price"
              >
                Use ${currentPrice.toFixed(2)}
              </button>
            )}
          </div>
          <div className="flex items-center gap-2 justify-end">
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-2 py-1 text-xs rounded bg-accent-cyan/20 hover:bg-accent-cyan/30 text-accent-cyan flex items-center gap-1"
            >
              <Check className="w-3 h-3" />
              Save
            </button>
            <button
              onClick={handleCancel}
              className="px-2 py-1 text-xs rounded bg-white/5 hover:bg-white/10 text-white/60"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-3 p-2.5 rounded-lg bg-white/[0.03] border border-white/5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs">
          {investmentDate ? (
            <>
              <span className="text-white/40 flex items-center gap-1">
                <Calendar className="w-3 h-3" />
                {formatInvestmentDate(investmentDate)}
              </span>
              {investmentPrice && (
                <span className="text-white/40 flex items-center gap-1">
                  <DollarSign className="w-3 h-3" />
                  ${investmentPrice.toFixed(2)}
                </span>
              )}
              {gainLossPct !== null && gainLossPct !== undefined && (
                <span className={`font-medium ${gainLossPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {gainLossPct >= 0 ? '+' : ''}{gainLossPct.toFixed(1)}%
                </span>
              )}
            </>
          ) : (
            <span className="text-white/30 italic">No investment date set</span>
          )}
        </div>
        <button
          onClick={() => setIsEditing(true)}
          className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
          title="Edit investment info"
        >
          <Edit2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

