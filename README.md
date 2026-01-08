# 🦋 ButterQuant

**ARIMA-GARCH Driven Butterfly Option Quantitative Analysis Platform**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](#)

> Leverage **Fourier Cycle Analysis** + **ARIMA Price Forecasting** + **GARCH Volatility Modeling** to intelligently design Call/Put/Iron Butterfly option strategies and maximize risk-adjusted returns.

---

## 📝 Important Notes for v4.1 Update
The v4.1 update introduces the **Daily Scanner** feature, which automatically scans Nasdaq 100 and S&P 500 components. Results are stored in an SQLite database. 
- **Time Required**: The scanning process may take 30+ minutes.
- **Visual Display**: A summary of scan results is displayed on the frontend Dashboard. You can click any ticker to jump to the analysis page for deep analysis (Click-to-Analyze).
- **Data Storage**: Detailed analysis results (including extensive intermediate calculation data) are written to a separate SQLite database. 
- **Data Accumulation**: Running the scan daily will build up historical data. Estimated storage is ~23MB per day, totaling about 7GB per year. 
- **Warning**: Before using this feature, please verify your storage path and disk capacity. If requirements are not met, you can skip this feature; the dashboard will not display scan results, but other functionalities remain unaffected.
- **Future Use**: The collected data will be used for a future Deep Learning module. Data writing standards are strictly defined for this purpose. Ensure daily data integrity if you plan to use the deep learning features later.

---

## 🎯 Project Highlights

### **Why Butterfly Options?**
- ✅ **Limited Risk**: Max Loss = Net Cost (usually < $5)
- ✅ **High Profit/Loss Ratio**: Max Profit can reach 2-5x the cost
- ✅ **Direction Neutral**: Delta ≈ 0, doesn't rely on market direction
- ✅ **Ideal for Consolidation**: A profit harvester for sideways markets

### **Core Advantages of ButterQuant**
- 🔬 **Scientific Pricing**: Black-Scholes + IV Skew adjustment to avoid gaps between theory and market reality.
- 📊 **Multidimensional Analysis**: Time domain (ARIMA) + Frequency domain (Fourier) + Volatility domain (GARCH).
- 🎲 **Greeks Management**: Automated monitoring of Delta/Gamma/Vega/Theta to ensure risk-neutral strategies.
- 🧠 **Smart Recommendations**: Multi-factor scoring system (0-100) recommending only high-probability opportunities.
- 📈 **Real-time Visualization**: Clear insights into price forecasts, volatility curves, and cycle decompositions.

---

## 🚀 Quick Start

### **1. Clone the Project**
```bash
git clone https://github.com/okuninushi-yamaguchi/ButterQuant.git
cd ButterQuant
```

### **2. Start Backend (Python)**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
> Backend runs at `http://localhost:5000`

### **3. Start Frontend (React)**
```bash
cd ..  # Back to project root
npm install --legacy-peer-deps
npm run dev
```
> Frontend runs at `http://localhost:3000`

### **4. Start Analysis**
Visit `http://localhost:3000`:
1. **Dashboard**: View real-time butterfly strategy overviews for popular stocks. Click any **Ticker** card to jump to the analysis page.
2. **Analyzer**: Enter any ticker (e.g., `TSLA`) to get a full report including price forecasts, volatility analysis, and Greeks details.

### **5. Start Daily Scan (Optional)**
To populate the strategy leaderboard, run the batch scanning script:
```bash
# In the backend directory
python daily_scanner.py
```
> This will scan Nasdaq 100 and S&P 500 components and store results in an SQLite database. The process takes 30+ minutes.
> You can also trigger a background scan via the API: `POST /api/scan`.

---

## 📚 Technical Methodology

### **1️⃣ Fourier Cycle Analysis**
Detect hidden rhythms in the market:
- Institutional algorithm execution patterns (VWAP cadence).
- Dominant cycle identification (7-180 days).
- Trend vs. Consolidation classification.
- → **Automatically selects optimal Days to Expiration (DTE).**

### **2️⃣ ARIMA Price Forecasting**
Intelligently predict future prices for 7-30 days:
- Automated parameter selection (AIC optimal).
- 95% Confidence Intervals.
- Price Stability evaluation.
- → **Determines the Butterfly center strike price (K2).**

### **3️⃣ GARCH Volatility Modeling**
Predict future volatility + detect IV mispricing:
- Real option chain IV (IV Skew).
- GARCH predicted volatility.
- IV Percentile (historical distribution).
- → **Discovers IV overestimation opportunities (Seller's Advantage).**

### **4️⃣ Black-Scholes Precise Pricing**
Avoid theory-to-market disconnects:
- Adjust Volatility based on moneyness (IV Skew).
- Include Bid-Ask spreads (3%-10%).
- Liquidity constraints (Volume/OI filtering).
- → **Calculates actual execution costs.**

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────┐
│           Frontend (React + Vite)        │
│  - Data Visualization (Recharts)         │
│  - Real-time Charts (Price/Vol/Cycle)    │
│  - Responsive UI (Tailwind CSS)          │
└──────────────┬──────────────────────────┘
               │ HTTP API
┌──────────────▼──────────────────────────┐
│           Backend (Flask API)            │
│  - yfinance (Data Acquisition)           │
│  - statsmodels (ARIMA Modeling)          │
│  - arch (GARCH Modeling)                 │
│  - numpy/scipy (Fourier Analysis)        │
│  - Black-Scholes Pricing Engine          │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Yahoo Finance API                 │
│  - Historical Price (2y Data)            │
│  - Option Chain (Strike/IV/Vol/OI)       │
│  - Risk-free Rate (^IRX)                 │
└─────────────────────────────────────────┘
```

---

## 📈 Strategy Types

### **Call Butterfly (Bullish Consolidation)**
Ideal when:
- ✅ Fourier: Bullish trend + Cycle bottom.
- ✅ ARIMA: Predicted price near center.
- ✅ GARCH: High IV (>60th percentile).

### **Put Butterfly (Bearish Consolidation)**
Ideal when:
- ✅ Fourier: Bearish trend + Cycle top.
- ✅ ARIMA: Predicted price near center.
- ✅ GARCH: High IV.

### **Iron Butterfly (Neutral Consolidation)**
Ideal when:
- ✅ Fourier: Sideways trend.
- ✅ ARIMA: High price stability (CI < 8%).
- ✅ GARCH: Very high IV (>75th percentile).

---

## 🏁 Roadmap

### **Completed ✅**
- [x] Fourier Cycle Analysis (VWAP detrended)
- [x] ARIMA Price Prediction (Auto-tuning)
- [x] GARCH Volatility Modeling
- [x] Black-Scholes Pricing with IV Skew
- [x] Full Greeks Calculation
- [x] Multi-factor Scoring System
- [x] Call/Put/Iron Butterfly Strategies
- [x] Modern UI Redesign (Dashboard/Logo/Navigation)
- [x] Click-to-Analyze Integration

### **Ongoing 🚧**
- [ ] Backtesting Framework (Synthetic Market Data)
- [ ] Liquidity Filtering (Volume/OI)
- [ ] Slippage Modeling (Three-factor decomposition)

### **Planned 🔮**
- [ ] Machine Learning Enhancement (XGBoost)
- [ ] Multi-strategy Portfolio Optimization
- [ ] Real-time Monitoring & Alerts
- [ ] Mobile Web Optimization

---

## ⚠️ Disclaimer

**IMPORTANT: PLEASE READ CAREFULLY**

This project is for **Educational, Research, and Demonstration** purposes only. It does not constitute investment advice, recommendations, or guidance. Option trading involves **extremely high risk** and can result in the **total loss of principal** or even losses exceeding the initial investment.

1. ⚠️ **Accuracy Not Guaranteed**: All predictions are based on historical data and statistical models. Past performance does not guarantee future results.
2. ⚠️ **High Risk**: You may lose all your investment in a short time. Certain strategies (like selling options) may involve unlimited loss potential.
3. ⚠️ **Requirements**: Before use, you must understand option risks, have appropriate risk tolerance, and consult a licensed financial advisor.
4. ⚠️ **Limitation of Liability**: The author is not responsible for any profits or losses resulting from the use of this tool. Use at your own risk.

---

## 📝 License

**© 2025 ButterQuant. All Rights Reserved.**

This project does **not** currently have an open-source license.
- ❌ **Prohibited**: Copying, modifying, or distributing this code; commercial use; creating derivative works.
- ✅ **Permitted**: Viewing for study/research; Forking for personal study (no public distribution).

📧 **Business or Licensing Inquiries**: [mingsely@gmail.com]

---

<p align="center">
  <strong>© 2025 ButterQuant. All Rights Reserved.</strong>
</p>
