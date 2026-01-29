/**
 * Error Panel Component
 *
 * Displays tracked errors with full context and debugging information.
 * Supports expanding errors for details, copying error info, and exporting reports.
 */

import { useState } from 'react';
import {
  AlertTriangle,
  XCircle,
  ChevronDown,
  ChevronRight,
  Copy,
  Download,
  Trash2,
  Info,
  AlertOctagon,
  X
} from 'lucide-react';
import { useTrackedErrors, type TrackedError } from '../services/errorTracking';

// ============================================================================
// Error Item Component
// ============================================================================

function ErrorItem({
  error,
  onDismiss
}: {
  error: TrackedError;
  onDismiss: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  // Extract params check for cleaner JSX (explicit boolean to satisfy strict TS)
  const hasParams = Boolean(error.context.params && Object.keys(error.context.params).length > 0);
  const hasMetadata = Boolean(error.context.metadata && Object.keys(error.context.metadata).length > 0);

  const severityConfig = {
    error: {
      icon: XCircle,
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/30',
      textColor: 'text-red-400',
      headerBg: 'bg-red-500/20'
    },
    warning: {
      icon: AlertTriangle,
      bgColor: 'bg-amber-500/10',
      borderColor: 'border-amber-500/30',
      textColor: 'text-amber-400',
      headerBg: 'bg-amber-500/20'
    },
    info: {
      icon: Info,
      bgColor: 'bg-blue-500/10',
      borderColor: 'border-blue-500/30',
      textColor: 'text-blue-400',
      headerBg: 'bg-blue-500/20'
    }
  };

  const config = severityConfig[error.severity];
  const Icon = config.icon;

  const copyToClipboard = () => {
    const text = JSON.stringify({
      id: error.id,
      timestamp: error.timestamp,
      message: error.message,
      context: error.context,
      httpStatus: error.httpStatus,
      requestUrl: error.requestUrl,
      responseBody: error.responseBody,
      stack: error.stack
    }, null, 2);

    navigator.clipboard.writeText(text);
  };

  return (
    <div className={`rounded-lg border ${config.borderColor} ${config.bgColor} overflow-hidden`}>
      {/* Header */}
      <div
        className={`flex items-start gap-3 p-3 cursor-pointer ${config.headerBg}`}
        onClick={() => setExpanded(!expanded)}
      >
        <Icon className={`w-5 h-5 ${config.textColor} flex-shrink-0 mt-0.5`} />

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-xs font-mono ${config.textColor}`}>
              {error.context.source}::{error.context.operation}
            </span>
            {error.context.ticker && (
              <span className="px-1.5 py-0.5 bg-white/10 rounded text-xs text-white/70 font-mono">
                {error.context.ticker}
              </span>
            )}
            {error.httpStatus && (
              <span className={`px-1.5 py-0.5 rounded text-xs font-mono ${
                error.httpStatus >= 500 ? 'bg-red-500/20 text-red-400' :
                error.httpStatus >= 400 ? 'bg-amber-500/20 text-amber-400' :
                'bg-blue-500/20 text-blue-400'
              }`}>
                HTTP {error.httpStatus}
              </span>
            )}
          </div>

          <p className={`text-sm ${config.textColor} break-words`}>
            {error.message}
          </p>

          <div className="flex items-center gap-2 mt-1 text-xs text-white/40">
            <span>{error.timestamp.toLocaleTimeString()}</span>
            <span className="text-white/20">|</span>
            <span>{error.id}</span>
          </div>
        </div>

        <div className="flex items-center gap-1">
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-white/40" />
          ) : (
            <ChevronRight className="w-4 h-4 text-white/40" />
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDismiss();
            }}
            className="p-1 hover:bg-white/10 rounded"
          >
            <X className="w-4 h-4 text-white/40 hover:text-white/60" />
          </button>
        </div>
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div className="p-3 border-t border-white/10 space-y-3">
          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={copyToClipboard}
              className="flex items-center gap-1 px-2 py-1 bg-white/5 hover:bg-white/10 rounded text-xs text-white/60"
            >
              <Copy className="w-3 h-3" />
              Copy Details
            </button>
          </div>

          {/* Context */}
          {hasParams && (
            <div>
              <h4 className="text-xs font-medium text-white/50 mb-1">Parameters</h4>
              <pre className="text-xs text-white/70 bg-black/30 p-2 rounded overflow-x-auto">
                {String(JSON.stringify(error.context.params, null, 2))}
              </pre>
            </div>
          )}

          {/* Request URL */}
          {error.requestUrl && (
            <div>
              <h4 className="text-xs font-medium text-white/50 mb-1">Request URL</h4>
              <code className="text-xs text-cyan-400 bg-black/30 p-2 rounded block break-all">
                {error.requestMethod && `${error.requestMethod} `}{error.requestUrl}
              </code>
            </div>
          )}

          {/* Response Body */}
          {error.responseBody != null && (
            <div>
              <h4 className="text-xs font-medium text-white/50 mb-1">Response Body</h4>
              <pre className="text-xs text-white/70 bg-black/30 p-2 rounded overflow-x-auto max-h-40">
                {typeof error.responseBody === 'string'
                  ? error.responseBody
                  : String(JSON.stringify(error.responseBody, null, 2))}
              </pre>
            </div>
          )}

          {/* Stack Trace */}
          {error.stack && (
            <div>
              <h4 className="text-xs font-medium text-white/50 mb-1">Stack Trace</h4>
              <pre className="text-xs text-white/50 bg-black/30 p-2 rounded overflow-x-auto max-h-40 whitespace-pre-wrap">
                {error.stack}
              </pre>
            </div>
          )}

          {/* Metadata */}
          {hasMetadata && (
            <div>
              <h4 className="text-xs font-medium text-white/50 mb-1">Additional Metadata</h4>
              <pre className="text-xs text-white/70 bg-black/30 p-2 rounded overflow-x-auto">
                {String(JSON.stringify(error.context.metadata, null, 2))}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Error Panel
// ============================================================================

export function ErrorPanel({
  maxHeight = '400px',
  showOnlyActive = true
}: {
  maxHeight?: string;
  showOnlyActive?: boolean;
}) {
  const { errors, activeErrors, dismissError, dismissAllErrors, exportReport } = useTrackedErrors();
  const [minimized, setMinimized] = useState(false);

  const displayErrors = showOnlyActive ? activeErrors : errors;

  if (displayErrors.length === 0) {
    return null;
  }

  const errorCount = displayErrors.filter(e => e.severity === 'error').length;
  const warningCount = displayErrors.filter(e => e.severity === 'warning').length;

  const handleExport = () => {
    const report = exportReport();
    const blob = new Blob([report], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `calibration-errors-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glass-card border-l-4 border-l-red-500 overflow-hidden mb-6">
      {/* Header */}
      <div
        className="flex items-center justify-between p-3 bg-red-500/10 cursor-pointer"
        onClick={() => setMinimized(!minimized)}
      >
        <div className="flex items-center gap-3">
          <AlertOctagon className="w-5 h-5 text-red-400" />
          <div>
            <span className="font-medium text-white">
              {displayErrors.length} Issue{displayErrors.length !== 1 ? 's' : ''} Detected
            </span>
            <span className="text-sm text-white/50 ml-2">
              {errorCount > 0 && <span className="text-red-400">{errorCount} error{errorCount !== 1 ? 's' : ''}</span>}
              {errorCount > 0 && warningCount > 0 && ', '}
              {warningCount > 0 && <span className="text-amber-400">{warningCount} warning{warningCount !== 1 ? 's' : ''}</span>}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!minimized && (
            <>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleExport();
                }}
                className="flex items-center gap-1 px-2 py-1 bg-white/5 hover:bg-white/10 rounded text-xs text-white/60"
                title="Export error report"
              >
                <Download className="w-3 h-3" />
                Export
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  dismissAllErrors();
                }}
                className="flex items-center gap-1 px-2 py-1 bg-white/5 hover:bg-white/10 rounded text-xs text-white/60"
                title="Dismiss all"
              >
                <Trash2 className="w-3 h-3" />
                Dismiss All
              </button>
            </>
          )}
          {minimized ? (
            <ChevronRight className="w-4 h-4 text-white/40" />
          ) : (
            <ChevronDown className="w-4 h-4 text-white/40" />
          )}
        </div>
      </div>

      {/* Error List */}
      {!minimized && (
        <div
          className="p-3 space-y-2 overflow-y-auto"
          style={{ maxHeight }}
        >
          {displayErrors.map(error => (
            <ErrorItem
              key={error.id}
              error={error}
              onDismiss={() => dismissError(error.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default ErrorPanel;
