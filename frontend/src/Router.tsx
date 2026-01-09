/**
 * Application Router
 * 
 * Wraps the app with React Router for navigation between:
 * - Main Portfolio Dashboard (/)
 * - WFO Calibration Page (/calibration)
 * 
 * This keeps routing isolated from the existing App component.
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import CalibrationPage from './pages/CalibrationPage';

export default function Router() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main Portfolio Dashboard */}
        <Route path="/" element={<App />} />
        
        {/* WFO Calibration Page (isolated from main app) */}
        <Route path="/calibration" element={<CalibrationPage />} />
      </Routes>
    </BrowserRouter>
  );
}

