import { useState } from 'react';
import { Settings, Save, X } from 'lucide-react';

interface PortfolioSettingsProps {
  currentValue: number;
  onUpdate: (value: number) => Promise<void>;
}

export function PortfolioSettings({ currentValue, onUpdate }: PortfolioSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [value, setValue] = useState(currentValue.toString());
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const numValue = parseFloat(value);
    if (isNaN(numValue) || numValue <= 0) return;
    
    setLoading(true);
    try {
      await onUpdate(numValue);
      setIsOpen(false);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-colors"
        title="Portfolio Settings"
      >
        <Settings className="w-5 h-5 text-white/70" />
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="flex items-center gap-2">
      <div className="flex items-center gap-2 bg-white/5 rounded-lg px-3 py-1.5 border border-white/10">
        <span className="text-white/50">$</span>
        <input
          type="number"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          className="bg-transparent border-none w-32 p-0 focus:ring-0"
          min="1"
          step="100"
          autoFocus
        />
      </div>
      <button
        type="submit"
        disabled={loading}
        className="p-2 rounded-lg bg-green-500/20 hover:bg-green-500/30 text-green-400 transition-colors"
      >
        <Save className="w-4 h-4" />
      </button>
      <button
        type="button"
        onClick={() => setIsOpen(false)}
        className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </form>
  );
}

