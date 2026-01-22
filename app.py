import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

st.set_page_config(page_title="NIFTY50 Capstone Dashboard", layout="wide")

st.title("📊 NIFTY50 Portfolio Strategy & ML Forecasting Dashboard")

# ----------------------------------
# Load Dataset (Safe)
# ----------------------------------

file_path = "NIFTY50_all.xlsb"

if not os.path.exists(file_path):
    st.error("❌ Dataset file missing: NIFTY50_all.xlsb")
    st.stop()

@st.cache_data
def load_data(path):
    return pd.read_excel(path, engine="pyxlsb")

df = load_data(file_path)

st.success("✅ Dataset Loaded Successfully")

# ----------------------------------
# Data Cleaning
# ----------------------------------

df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values("Date", inplace=True)

# ----------------------------------
# Sidebar Controls
# ----------------------------------

st.sidebar.header("⚙ Dashboard Controls")

stocks = sorted(df['Symbol'].unique())
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

# ----------------------------------
# Stock Filter
# ----------------------------------

stock_df = df[df['Symbol'] == selected_stock]

# ----------------------------------
# Price Trend Chart
# ----------------------------------

st.subheader(f"📈 {selected_stock} Price Trend")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(stock_df['Date'], stock_df['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# ----------------------------------
# Returns & Volatility
# ----------------------------------

stock_df['Return'] = stock_df['Close'].pct_change()

col1, col2, col3 = st.columns(3)
col1.metric("Avg Daily Return", f"{stock_df['Return'].mean()*100:.2f}%")
col2.metric("Volatility", f"{stock_df['Return'].std()*100:.2f}%")
col3.metric("Total Return", f"{(stock_df['Close'].iloc[-1]/stock_df['Close'].iloc[0]-1)*100:.2f}%")

# ----------------------------------
# ML Prediction Model
# ----------------------------------

st.subheader("🤖 ML-Based Price Forecast")

ml_df = stock_df[['Open','High','Low','Volume','Close']].dropna()

X = ml_df[['Open','High','Low','Volume']]
y = ml_df['Close']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

st.write(f"**Model RMSE:** {rmse:.2f}")
st.write(f"**Model R² Score:** {r2:.4f}")

# ----------------------------------
# Prediction Plot
# ----------------------------------

fig2, ax2 = plt.subplots(figsize=(12,5))
ax2.plot(y_test.values[:100], label="Actual")
ax2.plot(pred[:100], label="Predicted")
ax2.legend()
ax2.set_title("Actual vs Predicted Prices")
st.pyplot(fig2)

# ----------------------------------
# Portfolio Strategy Section
# ----------------------------------

st.subheader("🎯 Portfolio Strategy Insights")

portfolio_df = (
    df.groupby('Symbol')['Close']
    .agg(['mean','std','count'])
    .reset_index()
)

portfolio_df['Risk'] = portfolio_df['std']
portfolio_df['Return'] = portfolio_df['mean']

top_stocks = portfolio_df.sort_values(by='Return', ascending=False).head(10)

st.dataframe(top_stocks)

st.markdown("""
### 📌 Business Insights
- High return stocks: Strong long-term growth candidates
- High volatility stocks: Suitable for aggressive investors
- Balanced portfolio: Mix of stable + high growth stocks
""")

st.success("🚀 Dashboard Loaded Successfully")
