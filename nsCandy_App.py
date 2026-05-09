#===Import Libraries ===
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

#=== Page Config ===
st.set_page_config(
    page_title="Nassau Candy Profitability Cashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
#=UI Style Designing
#=== Load the dataset ===
@st.cache_data
def load_data():
    return pd.read_csv("Nassau Candy Distributor.csv")

try:
    df = load_data()
except:
    st.error("❌ Dataset not found! Keep CSV in same folder")
    st.stop()

    
#=== Data Parsing ===
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(
        df['Order Date'].astype(str).str.strip(),
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )
    df['Ship Date'] = pd.to_datetime(
        df['Ship Date'].astype(str).str.strip(),
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

df = df.dropna(subset=['Order Date'])


#=== Data Clinning ===
df = df[df['Sales'] > 0]
df = df.dropna(subset=['Units', 'Cost'])
df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')
df = df.dropna(subset=['Gross Profit'])

#=== Calculate KPIs ===
df['Margin %'] = np.where(df['Sales'] > 0, (df['Gross Profit'] / df['Sales']) * 100, 0)
df['Profit per Unit'] = np.where(df['Units'] > 0, df['Gross Profit'] / df['Units'], 0)
df['Cost per Unit'] = np.where(df['Units'] > 0, df['Cost'] / df['Units'], 0)

#=== SIDEBAR FILTERS ===
st.sidebar.title("🔍 Advanced Filters")

if 'Order Date' in df.columns:
    min_date = df['Order Date'].min()
    max_date = df['Order Date'].max()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        df = df[(df['Order Date'] >= pd.to_datetime(date_range[0])) &
                (df['Order Date'] <= pd.to_datetime(date_range[1]))]
 
# Division filter
selected_div = st.sidebar.multiselect(
    "🏢 Division",
    df['Division'].unique(),
    default=df['Division'].unique()
)
df = df[df['Division'].isin(selected_div)]

# Margin threshold
margin_threshold = st.sidebar.slider(
    "📊 Margin % Threshold",
    min_value=df['Margin %'].min(),
    max_value=df['Margin %'].max(),
    value=(df['Margin %'].min(), df['Margin %'].max()),
    step=1.0
)
df_filtered = df[(df['Margin %'] >= margin_threshold[0]) & (df['Margin %'] <= margin_threshold[1])]

# Product search
product_search = st.sidebar.text_input("🔎 Search Product Name")
if product_search:
    df_filtered = df_filtered[df_filtered['Product Name'].str.contains(product_search, case=False, na=False)]

# HEADER & TITLE
st.markdown("# 🍬 Nassau Candy Profitability & Margin Analysis")
st.markdown("**Comprehensive Product Line Performance Dashboard**")
st.markdown("---")
