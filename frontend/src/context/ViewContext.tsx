/**
 * View Context
 * 
 * Manages global view preferences like prediction horizon.
 * Allows switching between Swing Trading (3d) and Trend Trading (15d) views.
 */

import React, { createContext, useContext, useState, ReactNode } from 'react';

// ============================================================================
// Types
// ============================================================================

export type PredictionHorizon = 3 | 15;

interface ViewContextValue {
  /**
   * Current prediction horizon in days.
   * 3 = Swing Trading
   * 15 = Trend Trading
   */
  predictionHorizon: PredictionHorizon;
  
  /** Update the prediction horizon */
  setPredictionHorizon: (horizon: PredictionHorizon) => void;
  
  /** Human readable label for the current view */
  viewLabel: string;
}

const ViewContext = createContext<ViewContextValue | null>(null);

// ============================================================================
// Provider
// ============================================================================

interface ViewProviderProps {
  children: ReactNode;
  initialHorizon?: PredictionHorizon;
}

export function ViewProvider({ 
  children,
  initialHorizon = 3 
}: ViewProviderProps) {
  const [predictionHorizon, setPredictionHorizon] = useState<PredictionHorizon>(initialHorizon);
  
  const viewLabel = predictionHorizon === 3 ? 'Swing (3d)' : 'Trend (15d)';
  
  return (
    <ViewContext.Provider value={{ 
      predictionHorizon, 
      setPredictionHorizon,
      viewLabel 
    }}>
      {children}
    </ViewContext.Provider>
  );
}

// ============================================================================
// Hook
// ============================================================================

export function useViewContext() {
  const context = useContext(ViewContext);
  if (!context) {
    throw new Error('useViewContext must be used within a ViewProvider');
  }
  return context;
}
