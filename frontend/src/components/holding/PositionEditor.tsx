import { useState } from 'react';
import { Layers, DollarSign, Edit2, Check, X } from 'lucide-react';

interface PositionEditorProps {
  holdingId: number;
  shares: number;
  avgCost?: number | null;
  currentPrice?: number;
  gainLoss?: number | null;
  gainLossPct?: number | null;
  onSave: (id: number, data: { shares?: number; avg_cost?: number }) => Promise<void>;
}

export function PositionEditor({
  holdingId,
  shares,
  avgCost,
  currentPrice,
  gainLoss,
  gainLossPct,
  onSave
}: PositionEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [sharesInput, setSharesInput] = useState(shares > 0 ? shares.toString() : '');
  const [costInput, setCostInput] = useState(avgCost?.toString() ?? '');
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const data: { shares?: number; avg_cost?: number } = {};
      
      const sharesNum = sharesInput === '' ? 0 : parseFloat(sharesInput);
      if (!isNaN(sharesNum) && sharesNum >= 0) {
        data.shares = sharesNum;
      }
      
      if (costInput) {
        const costNum = parseFloat(costInput);
        if (!isNaN(costNum) && costNum > 0) {
          data.avg_cost = costNum;
        }
      }
      
      await onSave(holdingId, data);
      setIsEditing(false);
    } catch (err) {
      console.error('Failed to save position:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setSharesInput(shares > 0 ? shares.toString() : '');
    setCostInput(avgCost?.toString() ?? '');
    setIsEditing(false);
  };

  // Format number compactly (1000 -> 1K, 1500 -> 1.5K)
  const formatCompact = (value: number): string => {
    const absValue = Math.abs(value);
    if (absValue >= 1000) {
      const k = value / 1000;
      return k % 1 === 0 ? `${k}K` : `${k.toFixed(1)}K`;
    }
    return value.toFixed(2);
  };

  // Format currency, using compact notation for large values
  const formatCurrency = (value: number | null | undefined): string => {
    if (value === null || value === undefined) return '—';
    const absValue = Math.abs(value);
    if (absValue >= 1000) {
      return formatCompact(value);
    }
    return value.toFixed(2);
  };

  // Format shares: round to 2 decimals, use K for thousands
  const formatShares = (value: number): string => {
    if (value >= 1000) {
      return formatCompact(value);
    }
    return value.toFixed(2).replace(/\.?0+$/, ''); // Remove trailing zeros
  };

  if (isEditing) {
    return (
      <div className="mb-3 p-2.5 rounded-lg bg-white/[0.03] border border-white/5">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-white/40" />
            <input
              type="number"
              value={sharesInput}
              onChange={(e) => setSharesInput(e.target.value)}
              placeholder="Number of shares"
              className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
              min="0"
              step="0.0001"
            />
          </div>
          <div className="flex items-center gap-2">
            <DollarSign className="w-3.5 h-3.5 text-white/40" />
            <input
              type="number"
              value={costInput}
              onChange={(e) => setCostInput(e.target.value)}
              placeholder="Avg cost per share"
              className="flex-1 px-2 py-1 text-xs bg-white/5 border border-white/20 rounded text-white focus:border-accent-cyan/50 focus:outline-none"
              min="0.01"
              step="0.01"
            />
            {currentPrice && (
              <button
                type="button"
                onClick={() => setCostInput(currentPrice.toFixed(2))}
                className="px-1.5 py-1 text-[10px] rounded bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/70 whitespace-nowrap"
                title="Use current price as avg cost"
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

  const hasPosition = shares > 0;
  const hasCost = avgCost !== null && avgCost !== undefined && avgCost > 0;

  return (
    <div className="mb-3 p-2.5 rounded-lg bg-white/[0.03] border border-white/5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5 text-xs">
          {hasPosition ? (
            <>
              {/* Shares count */}
              <span className="text-white/60 flex items-center gap-1">
                <Layers className="w-3 h-3" />
                {formatShares(shares)} shares
              </span>
              
              {/* Average cost */}
              {hasCost && (
                <span className="text-white/40">
                  @${avgCost!.toFixed(2)}
                </span>
              )}
              
              {/* Gain/Loss */}
              {gainLossPct !== null && gainLossPct !== undefined && (
                <span className={`font-medium ${gainLossPct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {gainLossPct >= 0 ? '+' : ''}{gainLossPct.toFixed(1)}%
                  {gainLoss !== null && gainLoss !== undefined && (
                    <span className="ml-1 font-normal">
                      ({gainLoss >= 0 ? '+' : '-'}${formatCurrency(Math.abs(gainLoss))})
                    </span>
                  )}
                </span>
              )}
            </>
          ) : (
            <span className="text-white/30 italic">No position set</span>
          )}
        </div>
        <button
          onClick={() => setIsEditing(true)}
          className="p-1 rounded hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
          title="Edit position"
        >
          <Edit2 className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

