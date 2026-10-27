import redis
import json
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import joblib
import time
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 1. Configuration
STOCKS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'ITC', 'LT', 'HINDUNILVR', 'AXISBANK']
TICKER_MAP = {s: f"{s}.NS" for s in STOCKS}
FEATURES = ['return_1d', 'return_5d', 'return_20d', 'volatility_20d', 'momentum_20d', 'rsi_14', 'macd', 'volume_ratio_20', 'esg_level']
ESG_SCORES = {'RELIANCE':0.65,'TCS':0.72,'HDFCBANK':0.58,'INFY':0.70,'ICICIBANK':0.60,'SBIN':0.52,'ITC':0.62,'LT':0.60,'HINDUNILVR':0.68,'AXISBANK':0.54}
SEQ_LEN = 30

print("🧠 Chargement des modèles...")
model_cnn = tf.keras.models.load_model('/content/drive/MyDrive/ESG_Portfolio_Project/model_cnn.keras')
sc_x = joblib.load('/content/drive/MyDrive/ESG_Portfolio_Project/sc_x.pkl')
sc_y = joblib.load('/content/drive/MyDrive/ESG_Portfolio_Project/sc_y.pkl')

def compute_tech(df):
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    out['return_1d'] = log_ret
    for p in [5, 20]: out[f'return_{p}d'] = log_ret.rolling(p).sum()
    for p in [20]: out[f'volatility_{p}d'] = log_ret.rolling(p).std()
    for p in [20]: out[f'momentum_{p}d'] = df['Close'] / df['Close'].shift(p) - 1
    for p in [14]:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = -delta.where(delta < 0, 0).rolling(p).mean()
        out[f'rsi_{p}'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    out['macd'] = ema12 - ema26
    out['volume_ratio_20'] = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
    return out

def optimize_portfolio(expected_returns, returns_matrix, esg_scores, min_esg=0.50):
    n = len(expected_returns)
    cov_matrix = np.cov(returns_matrix.T) + np.eye(n) * 1e-6
    def objective(w):
        ret = np.dot(w, expected_returns)
        vol = np.sqrt(w.T @ cov_matrix @ w)
        return -(ret - 1.0 * vol)
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: np.dot(w, esg_scores) - min_esg}]
    bounds = tuple((0.0, 0.20) for _ in range(n))
    res = minimize(objective, np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=cons)
    return np.array(res.x) if res.success else np.ones(n)/n

r = redis.Redis(host='localhost', port=6379, db=0)
print("🚀 Worker démarré. Mise à jour toutes les 5 secondes...")

while True:
    try:
        # Téléchargement des données
        data = yf.download(list(TICKER_MAP.values()), period="3mo", interval="1d", progress=False, group_by='ticker')
        
        X_live, valid_stocks, current_prices = [], [], []
        
        for s in STOCKS:
            ticker = TICKER_MAP[s]
            if ticker in data.columns.levels[0]:
                df_s = data[ticker].dropna()
                if len(df_s) >= SEQ_LEN:
                    # Simulation d'un prix live (si le marché est fermé, on ajoute un petit bruit)
                    live_price = float(df_s['Close'].iloc[-1]) * (1 + np.random.normal(0, 0.0005))
                    
                    feat = compute_tech(df_s)
                    feat['esg_level'] = ESG_SCORES.get(s, 0.5)
                    feat = feat.dropna()
                    
                    if len(feat) >= SEQ_LEN:
                        X_live.append(feat[FEATURES].values[-SEQ_LEN:])
                        valid_stocks.append(s)
                        current_prices.append(live_price)
                        
                        # Stockage du prix et des features
                        r.set(f"price:{s}", live_price)
                        r.hset(f"features:{s}", mapping={k: float(v) for k, v in feat.iloc[-1].to_dict().items()})
        
        if X_live:
            X_live = np.array(X_live)
            X_live_s = sc_x.transform(X_live.reshape(-1, len(FEATURES))).reshape(X_live.shape)
            preds = sc_y.inverse_transform(model_cnn.predict(X_live_s, verbose=0)).flatten()
            
            esg_scores = np.array([ESG_SCORES.get(s, 0.5) for s in valid_stocks])
            returns_matrix = np.random.normal(0, 0.02, (60, len(valid_stocks))) 
            
            # Optimisation du portefeuille (ce que tu demandais)
            weights = optimize_portfolio(preds, returns_matrix, esg_scores)
            
            # Stockage des prédictions et des poids
            for i, s in enumerate(valid_stocks):
                r.set(f"prediction:{s}", float(preds[i]))
                r.set(f"weight:{s}", float(weights[i]))
                r.set(f"esg:{s}", float(esg_scores[i]))
            
            print(f"✅ Mise à jour réussie à {time.strftime('%H:%M:%S')}")
        
        time.sleep(5) # Pause de 5 secondes
    except Exception as e:
        print(f"❌ Erreur : {e}")
        time.sleep(5)
