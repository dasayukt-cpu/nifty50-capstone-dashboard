import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import glob

st.set_page_config(page_title="NIFTY 50 Dashboard", layout="wide")

st.title("📊 NIFTY 50 Stock Analytics Dashboard")
st.markdown("### Capstone Project — Portfolio Strategy & ML Forecasting")

# -------------------- LOAD DATA --------------------

files = glob.glob("*.csv")
stock_files = [f for f in files if "NIFTY50" not in f and "metadata" not in f]

@st.cache_data
def load_data():
    df_list = []
    for file in stock_files:
        temp = pd.read_csv(file)
        temp['Stock'] = file.replace(".csv","")
        df_list.append(temp)
    return pd.concat(df_list, ignore_index=True)

stocks_df = load_data()

nifty = pd.read_csv("NIFTY50_all.csv")

stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
nifty['Date'] = pd.to_datetime(nifty['Date'])

# -------------------- SIDEBAR --------------------

st.sidebar.header("🔧 Controls")
stock_selected = st.sidebar.selectbox("Select Stock", sorted(stocks_df['Stock'].unique()))

# -------------------- MARKET TREND --------------------

st.subheader("📈 NIFTY 50 Market Trend")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(nifty['Date'], nifty['Close'])
ax.set_title("NIFTY 50 Index Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
st.pyplot(fig)

# -------------------- STOCK TREND --------------------

st.subheader(f"📊 {stock_selected} Price Trend")

df_stock = stocks_df[stocks_df['Stock'] == stock_selected]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_stock['Date'], df_stock['Close'])
ax.set_title(f"{stock_selected} Closing Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# -------------------- PORTFOLIO STRATEGY --------------------

st.subheader("🏆 Portfolio Strategy — Top 5 Stocks")

stocks_df['Daily_Return'] = stocks_df.groupby('Stock')['Close'].pct_change()

summary = stocks_df.groupby('Stock')['Daily_Return'].agg(['mean','std'])
summary['Sharpe'] = summary['mean'] / summary['std']
top_5 = summary.sort_values(by='Sharpe', ascending=False).head(5)

st.dataframe(top_5)

# -------------------- PORTFOLIO PERFORMANCE --------------------

st.subheader("📈 Portfolio Performance Trend")

portfolio_df = stocks_df[stocks_df['Stock'].isin(top_5.index)]
portfolio_trend = portfolio_df.groupby('Date')['Close'].mean()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(portfolio_trend.index, portfolio_trend.values)
ax.set_title("Portfolio Return Trend")
ax.set_xlabel("Date")
ax.set_ylabel("Average Close Price")
st.pyplot(fig)

# -------------------- FOOTER --------------------

st.success("Dashboard Loaded Successfully 🚀")
st.caption("Capstone Project | Data Science & AI | NIFTY 50 Stock Analytics")
