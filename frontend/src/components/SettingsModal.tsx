/**
 * Settings Modal
 *
 * Provides cache management and admin functions.
 */

import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { X, Trash2, RefreshCw, Database, AlertTriangle, Check, Download, Upload } from 'lucide-react';
import {
  getCacheStats,
  clearAnalysisCache,
  clearCalibrationWeights,
  clearPriceHistory,
  exportPortfolio,
  importPortfolio,
  type CacheStats
} from '../services/api';
import { clearAnalysisCache as clearFrontendAnalysisCache } from '../hooks/useStockAnalysis';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type ClearAction = 'analysis' | 'calibration' | 'price-daily' | 'price-intraday' | 'price-all';

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState<ClearAction | null>(null);
  const [ticker, setTicker] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load cache stats when modal opens
  useEffect(() => {
    if (isOpen) {
      loadStats();
    }
  }, [isOpen]);

  const loadStats = async () => {
    setLoading(true);
    try {
      const data = await getCacheStats();
      setStats(data);
    } catch (err) {
      console.error('Failed to load cache stats:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async (action: ClearAction) => {
    setClearing(action);
    setMessage(null);

    try {
      const tickerParam = ticker.trim() || undefined;
      let result;

      switch (action) {
        case 'analysis':
          result = await clearAnalysisCache(tickerParam);
          // Also clear frontend cache
          clearFrontendAnalysisCache();
          break;
        case 'calibration':
          result = await clearCalibrationWeights(tickerParam);
          break;
        case 'price-daily':
          result = await clearPriceHistory(tickerParam, 'daily');
          break;
        case 'price-intraday':
          result = await clearPriceHistory(tickerParam, 'intraday');
          break;
        case 'price-all':
          result = await clearPriceHistory(tickerParam, 'all');
          break;
      }

      setMessage({ type: 'success', text: result.message });
      // Reload stats
      loadStats();
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to clear cache' });
    } finally {
      setClearing(null);
    }
  };

  const handleExport = async () => {
    setExporting(true);
    setMessage(null);
    try {
      const data = await exportPortfolio();
      // Create and download file
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `portfolio_backup_${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setMessage({
        type: 'success',
        text: `Exported ${data.holdings.length} holdings and ${data.option_holdings.length} options`
      });
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Export failed' });
    } finally {
      setExporting(false);
    }
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImporting(true);
    setMessage(null);

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const result = await importPortfolio(data, true);
      setMessage({
        type: 'success',
        text: `Imported ${result.holdings_imported} holdings and ${result.options_imported} options`
      });
      // Reload the page to reflect changes
      setTimeout(() => window.location.reload(), 1500);
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Import failed' });
    } finally {
      setImporting(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div className="relative glass-card w-full max-w-lg max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-accent-cyan" />
            <h2 className="text-lg font-bold text-white">Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5 text-white/70" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Cache Stats */}
          <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-white/80">Cache Statistics</h3>
              <button
                onClick={loadStats}
                disabled={loading}
                className="p-1.5 rounded hover:bg-white/10 transition-colors"
                title="Refresh stats"
              >
                <RefreshCw className={`w-4 h-4 text-white/50 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>
            {stats ? (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-white/50">Analysis Cache:</span>
                  <span className="text-white font-mono">{stats.analysis_cache}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Calibrated Tickers:</span>
                  <span className="text-white font-mono">{stats.calibrated_tickers}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Calibration Weights:</span>
                  <span className="text-white font-mono">{stats.calibration_weights}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Calibration Windows:</span>
                  <span className="text-white font-mono">{stats.calibration_windows}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Daily Price History:</span>
                  <span className="text-white font-mono">{stats.daily_price_history.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-white/50">Intraday History:</span>
                  <span className="text-white font-mono">{stats.intraday_price_history.toLocaleString()}</span>
                </div>
              </div>
            ) : (
              <p className="text-white/30 text-sm">Loading...</p>
            )}
          </div>

          {/* Portfolio Backup */}
          <div className="glass-card p-4">
            <h3 className="text-sm font-medium text-white/80 mb-3">Portfolio Backup</h3>
            <p className="text-xs text-white/50 mb-3">
              Export your holdings before database changes. Import to restore.
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleExport}
                disabled={exporting}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-accent-cyan/20 hover:bg-accent-cyan/30 border border-accent-cyan/30 text-accent-cyan transition-colors disabled:opacity-50"
              >
                {exporting ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                <span className="text-sm font-medium">Export</span>
              </button>
              <button
                onClick={handleImportClick}
                disabled={importing}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-green-500/20 hover:bg-green-500/30 border border-green-500/30 text-green-400 transition-colors disabled:opacity-50"
              >
                {importing ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4" />
                )}
                <span className="text-sm font-medium">Import</span>
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleFileSelect}
                className="hidden"
              />
            </div>
          </div>

          {/* Ticker Filter */}
          <div>
            <label className="block text-sm text-white/60 mb-2">
              Ticker Filter (leave empty for all)
            </label>
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g., AAPL"
              className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white placeholder-white/30 focus:border-accent-cyan/50 focus:outline-none"
            />
          </div>

          {/* Clear Actions */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-white/80">Clear Cache</h3>

            {/* Analysis Cache */}
            <ClearButton
              label="Analysis Cache"
              description="Fundamentals, options data, has_options flag"
              isClearing={clearing === 'analysis'}
              onClick={() => handleClear('analysis')}
              variant="warning"
            />

            {/* Calibration Weights */}
            <ClearButton
              label="WFO Calibration"
              description="Weights, windows, and simulated trades"
              isClearing={clearing === 'calibration'}
              onClick={() => handleClear('calibration')}
              variant="danger"
            />

            {/* Price History */}
            <div className="space-y-2">
              <ClearButton
                label="Daily Price History"
                description="Historical daily OHLCV data"
                isClearing={clearing === 'price-daily'}
                onClick={() => handleClear('price-daily')}
                variant="warning"
              />
              <ClearButton
                label="Intraday Price History"
                description="5min, 15min, hourly candles"
                isClearing={clearing === 'price-intraday'}
                onClick={() => handleClear('price-intraday')}
                variant="warning"
              />
              <ClearButton
                label="All Price History"
                description="Both daily and intraday"
                isClearing={clearing === 'price-all'}
                onClick={() => handleClear('price-all')}
                variant="danger"
              />
            </div>
          </div>

          {/* Message */}
          {message && (
            <div
              className={`flex items-center gap-2 p-3 rounded-lg ${
                message.type === 'success'
                  ? 'bg-green-500/10 border border-green-500/30 text-green-400'
                  : 'bg-red-500/10 border border-red-500/30 text-red-400'
              }`}
            >
              {message.type === 'success' ? (
                <Check className="w-4 h-4" />
              ) : (
                <AlertTriangle className="w-4 h-4" />
              )}
              <span className="text-sm">{message.text}</span>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

// Clear button component
function ClearButton({
  label,
  description,
  isClearing,
  onClick,
  variant = 'warning'
}: {
  label: string;
  description: string;
  isClearing: boolean;
  onClick: () => void;
  variant?: 'warning' | 'danger';
}) {
  const colors =
    variant === 'danger'
      ? 'bg-red-500/10 hover:bg-red-500/20 border-red-500/30 text-red-400'
      : 'bg-amber-500/10 hover:bg-amber-500/20 border-amber-500/30 text-amber-400';

  return (
    <button
      onClick={onClick}
      disabled={isClearing}
      className={`w-full flex items-center justify-between p-3 rounded-lg border transition-colors ${colors} disabled:opacity-50`}
    >
      <div className="text-left">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs opacity-70">{description}</p>
      </div>
      {isClearing ? (
        <RefreshCw className="w-4 h-4 animate-spin" />
      ) : (
        <Trash2 className="w-4 h-4" />
      )}
    </button>
  );
}
