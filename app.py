import streamlit as st
import pandas as pd
import redis
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Real-Time ESG Portfolio", layout="wide")

# ⚠️ LA LIGNE MAGIQUE POUR LE TEMPS RÉEL
# Rafraîchit la page automatiquement toutes les 5 secondes (5000 ms)
count = st_autorefresh(interval=5000, limit=None, key="data_refresh")

st.title("🇮🇳 NIFTY 50 - Live Decision Dashboard")
st.write("Les données et les graphiques se mettent à jour automatiquement.")

r = redis.Redis(host='localhost', port=6379, db=0)
stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'ITC', 'LT', 'HINDUNILVR', 'AXISBANK']

# 1. Récupération des données depuis Redis
data_list = []
for s in stocks:
    price = r.get(f"price:{s}")
    pred = r.get(f"prediction:{s}")
    weight = r.get(f"weight:{s}")
    esg = r.get(f"esg:{s}")
    
    if price and pred and weight:
        data_list.append({
            'Action': s,
            'Prix': float(price),
            'Prediction_%': float(pred) * 100,
            'Poids_%': float(weight) * 100,
            'ESG': float(esg)
        })

if not data_list:
    st.error("En attente des données du Worker...")
    st.stop()

df = pd.DataFrame(data_list).sort_values('Poids_%', ascending=False)

# 2. Métriques en haut
col1, col2, col3 = st.columns(3)
rendement_attendu = (df['Prediction_%'] * df['Poids_%'] / 100).sum()
score_esg_moyen = (df['ESG'] * df['Poids_%'] / 100).sum()

col1.metric("Rendement Attendu (Portefeuille)", f"{rendement_attendu:+.2f}%")
col2.metric("Score ESG du Portefeuille", f"{score_esg_moyen:.3f}")
col3.metric("Statut", "🟢 Live (Rafraîchi toutes les 5s)")

st.markdown("---")

# 3. Graphiques de décision (Plotly)
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("### 🎯 Top Prédictions de l'IA")
    # Graphique des prédictions
    fig_pred = px.bar(df, x='Action', y='Prediction_%', 
                      color='Prediction_%', color_continuous_scale='RdYlGn',
                      title="Rendement prédit à 20 jours (%)")
    st.plotly_chart(fig_pred, use_container_width=True)

with col_chart2:
    st.markdown("### 💼 Allocation Recommandée")
    # Graphique des poids alloués
    fig_weight = px.bar(df, x='Action', y='Poids_%', 
                        color='Poids_%', color_continuous_scale='Blues',
                        title="Poids du portefeuille (%)")
    st.plotly_chart(fig_weight, use_container_width=True)

st.markdown("---")
st.markdown("### 📋 Tableau de Décision Détaillé")
st.dataframe(df[['Action', 'Prix', 'Prediction_%', 'Poids_%', 'ESG']].style.background_gradient(cmap='Blues', subset=['Poids_%']), use_container_width=True)
