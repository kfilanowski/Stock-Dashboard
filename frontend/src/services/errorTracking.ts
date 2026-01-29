/**
 * Error Tracking Service
 *
 * Centralized error handling with full context for debugging.
 * Captures stack traces, request details, and state information.
 */

// ============================================================================
// Types
// ============================================================================

export interface ErrorContext {
  // What operation was being performed
  operation: string;
  // Component or service that threw the error
  source: string;
  // Relevant parameters/data at time of error
  params?: Record<string, unknown>;
  // Ticker being processed (if applicable)
  ticker?: string;
  // Additional metadata
  metadata?: Record<string, unknown>;
}

export interface TrackedError {
  id: string;
  timestamp: Date;
  message: string;
  originalError?: Error;
  stack?: string;
  context: ErrorContext;
  // For API errors
  httpStatus?: number;
  httpStatusText?: string;
  requestUrl?: string;
  requestMethod?: string;
  responseBody?: unknown;
  // Severity
  severity: 'error' | 'warning' | 'info';
  // Has user dismissed this?
  dismissed: boolean;
}

export interface ErrorReport {
  errors: TrackedError[];
  sessionStart: Date;
  userAgent: string;
  url: string;
}

// ============================================================================
// Error Store (in-memory, persists for session)
// ============================================================================

const MAX_ERRORS = 100;
const errorStore: TrackedError[] = [];
const sessionStart = new Date();
const listeners: Set<(errors: TrackedError[]) => void> = new Set();

function notifyListeners() {
  const errors = [...errorStore];
  listeners.forEach(fn => fn(errors));
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Generate a unique error ID
 */
function generateErrorId(): string {
  return `err_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * Extract stack trace from error
 */
function extractStack(error: unknown): string | undefined {
  if (error instanceof Error) {
    return error.stack;
  }
  // Try to get current stack
  try {
    throw new Error('Stack trace');
  } catch (e) {
    if (e instanceof Error) {
      // Remove first two lines (this function and trackError)
      const lines = e.stack?.split('\n') || [];
      return lines.slice(3).join('\n');
    }
  }
  return undefined;
}

/**
 * Track an error with full context
 */
export function trackError(
  error: unknown,
  context: ErrorContext,
  severity: 'error' | 'warning' | 'info' = 'error'
): TrackedError {
  const message = error instanceof Error
    ? error.message
    : typeof error === 'string'
      ? error
      : JSON.stringify(error);

  const tracked: TrackedError = {
    id: generateErrorId(),
    timestamp: new Date(),
    message,
    originalError: error instanceof Error ? error : undefined,
    stack: extractStack(error),
    context,
    severity,
    dismissed: false
  };

  // Add to store (FIFO if at capacity)
  if (errorStore.length >= MAX_ERRORS) {
    errorStore.shift();
  }
  errorStore.push(tracked);

  // Log to console with full context
  const logMethod = severity === 'error' ? console.error : severity === 'warning' ? console.warn : console.info;
  logMethod(
    `[${severity.toUpperCase()}] ${context.source}::${context.operation}`,
    '\nMessage:', message,
    '\nContext:', context,
    '\nStack:', tracked.stack
  );

  notifyListeners();
  return tracked;
}

/**
 * Track an API error with request/response details
 */
export async function trackApiError(
  response: Response,
  context: ErrorContext
): Promise<TrackedError> {
  let responseBody: unknown;
  try {
    const text = await response.text();
    try {
      responseBody = JSON.parse(text);
    } catch {
      responseBody = text;
    }
  } catch {
    responseBody = '<unable to read response body>';
  }

  // Extract error message from response
  let message = `HTTP ${response.status}: ${response.statusText}`;
  if (responseBody && typeof responseBody === 'object') {
    const body = responseBody as Record<string, unknown>;
    if (body.detail) {
      message = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } else if (body.error) {
      message = String(body.error);
    } else if (body.message) {
      message = String(body.message);
    }
  }

  const tracked: TrackedError = {
    id: generateErrorId(),
    timestamp: new Date(),
    message,
    stack: extractStack(new Error(message)),
    context,
    httpStatus: response.status,
    httpStatusText: response.statusText,
    requestUrl: response.url,
    requestMethod: context.metadata?.method as string || 'GET',
    responseBody,
    severity: response.status >= 500 ? 'error' : 'warning',
    dismissed: false
  };

  if (errorStore.length >= MAX_ERRORS) {
    errorStore.shift();
  }
  errorStore.push(tracked);

  console.error(
    `[API ERROR] ${context.source}::${context.operation}`,
    '\nURL:', response.url,
    '\nStatus:', response.status, response.statusText,
    '\nResponse:', responseBody,
    '\nContext:', context
  );

  notifyListeners();
  return tracked;
}

/**
 * Dismiss an error (user acknowledged it)
 */
export function dismissError(errorId: string): void {
  const error = errorStore.find(e => e.id === errorId);
  if (error) {
    error.dismissed = true;
    notifyListeners();
  }
}

/**
 * Dismiss all errors
 */
export function dismissAllErrors(): void {
  errorStore.forEach(e => e.dismissed = true);
  notifyListeners();
}

/**
 * Get all tracked errors
 */
export function getErrors(): TrackedError[] {
  return [...errorStore];
}

/**
 * Get undismissed errors
 */
export function getActiveErrors(): TrackedError[] {
  return errorStore.filter(e => !e.dismissed);
}

/**
 * Subscribe to error updates
 */
export function subscribeToErrors(callback: (errors: TrackedError[]) => void): () => void {
  listeners.add(callback);
  return () => listeners.delete(callback);
}

/**
 * Generate a full error report for debugging
 */
export function generateErrorReport(): ErrorReport {
  return {
    errors: [...errorStore],
    sessionStart,
    userAgent: navigator.userAgent,
    url: window.location.href
  };
}

/**
 * Export error report as JSON string (for copying/downloading)
 */
export function exportErrorReport(): string {
  const report = generateErrorReport();
  return JSON.stringify(report, (_key, value) => {
    // Handle Error objects
    if (value instanceof Error) {
      return {
        name: value.name,
        message: value.message,
        stack: value.stack
      };
    }
    // Handle Date objects
    if (value instanceof Date) {
      return value.toISOString();
    }
    return value;
  }, 2);
}

/**
 * Clear all errors (for testing or reset)
 */
export function clearErrors(): void {
  errorStore.length = 0;
  notifyListeners();
}

// ============================================================================
// React Hook
// ============================================================================

import { useState, useEffect } from 'react';

export function useTrackedErrors() {
  const [errors, setErrors] = useState<TrackedError[]>(getErrors());

  useEffect(() => {
    return subscribeToErrors(setErrors);
  }, []);

  return {
    errors,
    activeErrors: errors.filter(e => !e.dismissed),
    dismissError,
    dismissAllErrors,
    clearErrors,
    exportReport: exportErrorReport
  };
}
