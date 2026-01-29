# Project Vision: Quantitative Trading & Analysis Platform

## Core Objective
To build a professional-grade **Quantitative Analysis Platform** that transcends simple portfolio tracking. The application serves as an algorithmic advisor, providing mathematically rigorous, backtested, and regime-aware trading recommendations for both stocks and options.

## Key Differentiators

### 1. Predictive, Not Just Descriptive
Unlike standard dashboards that show *what happened*, this platform predicts *what is likely to happen*.
*   **Walk-Forward Optimization (WFO):** The system continuously "learns" from recent price history to determine which technical indicators (RSI, MACD, etc.) are currently predictive for each specific asset.
*   **Dynamic Calibration:** It refuses to use a "one-size-fits-all" strategy, instead mathematically optimizing weightings for every individual stock.

### 2. Regime-Adaptive Intelligence
The system recognizes that "Trend Following" works in Bull markets but fails in Chop, and "Mean Reversion" works in Chop but is dangerous in Crashes.
*   **Market Regime Detection:** Automatically classifies the market into 6 distinct states (e.g., `BULL_QUIET`, `BEAR_VOLATILE`).
*   **Safety Protocols:** Automatically disables dangerous strategies (like buying dips in a crash) regardless of raw indicator signals.

### 3. Multi-Dimensional Signal Generation
Recommendations are derived from a triangulation of data sources:
*   **Technical:** Optimized indicator signals.
*   **Fundamental:** ROIC, Profit Margins, and Earnings Proximity.
*   **Options Flow:** Implied Volatility (IV) Rank and Put/Call Ratios.

## User Experience Goals
*   **"Cockpit" Interface:** A clean, modern web interface (React/Tailwind) that abstracts complex mathematics into clear, actionable "Buy/Sell/Hold" signals with confidence scores.
*   **Transparency:** Every recommendation includes the "Why" — visualizing the specific indicators and regime factors driving the decision.
*   **Automation:** Background calibration jobs ensure recommendations are always based on the latest data without manual intervention.

## Technical Ambition
*   **Institutional Grade Engine:** Python/Pandas backend capable of vectorized simulations (1000x faster than loops) to validate strategies in real-time.
*   **Full-Stack Cohesion:** Seamless integration between the heavy compute backend (FastAPI) and the interactive frontend.
