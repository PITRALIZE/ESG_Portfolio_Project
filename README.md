# Advanced ESG Portfolio Management System 🇮🇳

## 📌 Description
This project is an intelligent portfolio management system for the Indian market (NIFTY 50). It uses hybrid Deep Learning architectures (CNN-LSTM & BiGRU) to predict returns, optimized via a Mean-CVaR algorithm with ESG constraints, and deployed through a real-time event-driven architecture (Redis + Streamlit).

## 🏗️ Architecture
- **Backend (Worker)**: AI predictions and in-memory Feature Store (Redis).
- **Frontend (Dashboard)**: Streamlit interface automatically refreshed every 5 seconds.
- **Optimization**: Mean-CVaR (Return / Risk / ESG).

## 🚀 How to Run the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Run the backend: `python worker.py`
3. Run the dashboard: `streamlit run app.py`

## 📊 Results
- Prediction/Reality correlation: 0.239
- Backtest portfolio return: 2.64%
- ESG constraint respected (> 0.50)
