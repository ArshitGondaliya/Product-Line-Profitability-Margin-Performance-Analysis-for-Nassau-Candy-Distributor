# =============================
# 🚀 NASSAU CANDY - PROFITABILITY & MARGIN ANALYSIS DASHBOARD
# Product Line Profitability & Margin Performance Analysis
# =============================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Nassau Candy Profitability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# UI STYLE
# =============================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: #ffffff;
}
h1, h2, h3 { color: #ffffff; font-weight: bold; }
.metric-card { 
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #ffc107;
}
.profit-high { color: #28a745; font-weight: bold; }
.margin-low { color: #dc3545; font-weight: bold; }
.metric-value { font-size: 24px; color: #ffc107; }
.insight-box {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 8px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("Nassau Candy Distributor.csv")

try:
    df = load_data()
except:
    st.error("❌ Dataset not found! Keep CSV in same folder")
    st.stop()

# =============================
# DATE PARSING FIX
# =============================
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

# =============================
# DATA CLEANING & FEATURE ENGINEERING
# =============================
df = df[df['Sales'] > 0]
df = df.dropna(subset=['Units', 'Cost'])
df['Gross Profit'] = pd.to_numeric(df['Gross Profit'], errors='coerce')
df = df.dropna(subset=['Gross Profit'])

# Calculate KPIs
df['Margin %'] = np.where(df['Sales'] > 0, (df['Gross Profit'] / df['Sales']) * 100, 0)
df['Profit per Unit'] = np.where(df['Units'] > 0, df['Gross Profit'] / df['Units'], 0)
df['Cost per Unit'] = np.where(df['Units'] > 0, df['Cost'] / df['Units'], 0)

# =============================
# SIDEBAR FILTERS
# =============================
st.sidebar.title("🔍 Advanced Filters")

# Date filter
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

# =============================
# HEADER & TITLE
# =============================
st.markdown("# 🍬 Nassau Candy Profitability & Margin Analysis")
st.markdown("**Comprehensive Product Line Performance Dashboard**")
st.markdown("---")

# =============================
# SECTION 1: KEY METRICS (KPIs)
# =============================
st.subheader("📊 Executive Summary - Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_sales = df_filtered['Sales'].sum()
    st.metric("💰 Total Sales", f"${total_sales:,.0f}")

with col2:
    total_profit = df_filtered['Gross Profit'].sum()
    st.metric("📈 Total Profit", f"${total_profit:,.0f}")

with col3:
    avg_margin = df_filtered['Margin %'].mean()
    st.metric("📊 Avg Margin %", f"{avg_margin:.2f}%")

with col4:
    total_units = df_filtered['Units'].sum()
    st.metric("📦 Units Sold", f"{total_units:,.0f}")

with col5:
    profit_per_unit_avg = df_filtered['Profit per Unit'].mean()
    st.metric("💵 Profit/Unit", f"${profit_per_unit_avg:.2f}")

st.markdown("---")

# =============================
# SECTION 2: PRODUCT PROFITABILITY ANALYSIS
# =============================
st.subheader("🎯 Product-Level Profitability Leaderboard")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Top 10 Products by Gross Profit**")
    top_profit = df_filtered.groupby('Product Name').agg({
        'Gross Profit': 'sum',
        'Sales': 'sum',
        'Units': 'sum',
        'Margin %': 'mean'
    }).reset_index().sort_values(by='Gross Profit', ascending=False).head(10)
    
    fig_profit = px.bar(
        top_profit,
        x='Gross Profit',
        y='Product Name',
        orientation='h',
        title='Top 10 Products by Profit',
        color='Gross Profit',
        color_continuous_scale='Greens',
        labels={'Gross Profit': 'Profit ($)', 'Product Name': 'Product'}
    )
    fig_profit.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_profit, use_container_width=True)

with col_right:
    st.markdown("**Top 10 Products by Gross Margin %**")
    top_margin = df_filtered.groupby('Product Name').agg({
        'Margin %': 'mean',
        'Gross Profit': 'sum',
        'Sales': 'sum'
    }).reset_index().sort_values(by='Margin %', ascending=False).head(10)
    
    fig_margin = px.bar(
        top_margin,
        x='Margin %',
        y='Product Name',
        orientation='h',
        title='Top 10 Products by Margin %',
        color='Margin %',
        color_continuous_scale='Blues',
        labels={'Margin %': 'Margin (%)', 'Product Name': 'Product'}
    )
    fig_margin.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_margin, use_container_width=True)

st.markdown("---")

# =============================
# SECTION 3: RISK ANALYSIS - HIGH SALES / LOW MARGIN
# =============================
st.subheader("⚠️ Margin Risk Alert - High Sales with Low Margins")

risk_products = df_filtered.groupby('Product Name').agg({
    'Sales': 'sum',
    'Margin %': 'mean',
    'Gross Profit': 'sum',
    'Units': 'sum'
}).reset_index()

# Flag products with high sales but low margin
risk_threshold_margin = df_filtered['Margin %'].quantile(0.25)
risk_threshold_sales = df_filtered['Sales'].quantile(0.75)

at_risk = risk_products[
    (risk_products['Sales'] > risk_threshold_sales) & 
    (risk_products['Margin %'] < risk_threshold_margin)
].sort_values('Sales', ascending=False)

if len(at_risk) > 0:
    st.warning(f"⚠️ **{len(at_risk)} products** have high sales but low margins - review pricing strategy!")
    
    col_risk_left, col_risk_right = st.columns(2)
    with col_risk_left:
        fig_risk = px.scatter(
            risk_products,
            x='Sales',
            y='Margin %',
            size='Gross Profit',
            color='Margin %',
            hover_name='Product Name',
            color_continuous_scale='RdYlGn',
            title='Sales vs Margin % - Risk Matrix'
        )
        fig_risk.add_hline(y=risk_threshold_margin, line_dash="dash", line_color="red", annotation_text="Low Margin Threshold")
        fig_risk.add_vline(x=risk_threshold_sales, line_dash="dash", line_color="red", annotation_text="High Sales Threshold")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col_risk_right:
        st.markdown("**At-Risk Products (High Sales, Low Margin)**")
        st.dataframe(
            at_risk[['Product Name', 'Sales', 'Margin %', 'Gross Profit']].style.format({
                'Sales': '${:,.2f}',
                'Margin %': '{:.2f}%',
                'Gross Profit': '${:,.2f}'
            }),
            use_container_width=True
        )
else:
    st.success("✅ No high-risk products detected!")

st.markdown("---")

# =============================
# SECTION 4: DIVISION-LEVEL ANALYSIS
# =============================
st.subheader("🏢 Division Performance Analysis")

division_perf = df_filtered.groupby('Division').agg({
    'Sales': 'sum',
    'Gross Profit': 'sum',
    'Margin %': 'mean',
    'Profit per Unit': 'mean',
    'Units': 'sum'
}).reset_index().sort_values('Gross Profit', ascending=False)

col_div1, col_div2, col_div3 = st.columns(3)

with col_div1:
    fig_div_sales = px.bar(
        division_perf,
        x='Division',
        y='Sales',
        title='Sales by Division',
        color='Sales',
        color_continuous_scale='Blues'
    )
    fig_div_sales.update_layout(height=350)
    st.plotly_chart(fig_div_sales, use_container_width=True)

with col_div2:
    fig_div_profit = px.bar(
        division_perf,
        x='Division',
        y='Gross Profit',
        title='Profit by Division',
        color='Gross Profit',
        color_continuous_scale='Greens'
    )
    fig_div_profit.update_layout(height=350)
    st.plotly_chart(fig_div_profit, use_container_width=True)

with col_div3:
    fig_div_margin = px.bar(
        division_perf,
        x='Division',
        y='Margin %',
        title='Avg Margin % by Division',
        color='Margin %',
        color_continuous_scale='Oranges'
    )
    fig_div_margin.update_layout(height=350)
    st.plotly_chart(fig_div_margin, use_container_width=True)

# Division comparison table
st.markdown("**Division Performance Metrics Table**")
st.dataframe(
    division_perf.style.format({
        'Sales': '${:,.2f}',
        'Gross Profit': '${:,.2f}',
        'Margin %': '{:.2f}%',
        'Profit per Unit': '${:.2f}'
    }).highlight_max(subset=['Gross Profit'], color='lightgreen').highlight_min(subset=['Margin %'], color='lightcoral'),
    use_container_width=True
)

st.markdown("---")

# =============================
# SECTION 5: PARETO ANALYSIS (80/20)
# =============================
st.subheader("📈 Profit Concentration Analysis (Pareto - 80/20 Rule)")

product_contrib = df_filtered.groupby('Product Name').agg({
    'Gross Profit': 'sum',
    'Sales': 'sum'
}).reset_index().sort_values('Gross Profit', ascending=False)

product_contrib['Profit %'] = (product_contrib['Gross Profit'] / product_contrib['Gross Profit'].sum()) * 100
product_contrib['Cumulative %'] = product_contrib['Profit %'].cumsum()

# Find 80% threshold
products_for_80 = len(product_contrib[product_contrib['Cumulative %'] <= 80])
total_products = len(product_contrib)
pareto_pct = (products_for_80 / total_products) * 100

col_pareto1, col_pareto2 = st.columns(2)

with col_pareto1:
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=product_contrib.head(15)['Product Name'],
        y=product_contrib.head(15)['Profit %'],
        name='Profit %',
        marker_color='steelblue'
    ))
    fig_pareto.add_trace(go.Scatter(
        x=product_contrib.head(15)['Product Name'],
        y=product_contrib.head(15)['Cumulative %'],
        name='Cumulative %',
        yaxis='y2',
        line=dict(color='red', width=3),
        mode='lines+markers'
    ))
    fig_pareto.update_layout(
        title='Pareto Analysis - Top 15 Products',
        xaxis_title='Product Name',
        yaxis_title='Individual Profit %',
        yaxis2=dict(overlaying='y', side='right', title='Cumulative %'),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

with col_pareto2:
    st.markdown(f"""
    ### 🎯 Pareto Insights
    
    - **{products_for_80} products** ({pareto_pct:.1f}% of product line) generate **80% of profit**
    - Total unique products: **{total_products}**
    - Profit concentration: **High dependency on limited SKUs**
    
    #### Recommendation:
    - Focus marketing & operational efforts on top {products_for_80} products
    - Review ROI on remaining {total_products - products_for_80} products
    - Consider discontinuing low-contribution SKUs
    """)

st.markdown("---")

# =============================
# SECTION 6: COST STRUCTURE & MARGIN DIAGNOSTICS
# =============================
st.subheader("🔍 Cost Structure & Margin Diagnostics")

col_cost1, col_cost2 = st.columns(2)

with col_cost1:
    fig_cost_scatter = px.scatter(
        df_filtered,
        x='Cost',
        y='Margin %',
        color='Division',
        size='Profit per Unit',
        hover_name='Product Name',
        title='Cost vs Margin % - Diagnostic View',
        labels={'Cost': 'Cost ($)', 'Margin %': 'Margin (%)'}
    )
    fig_cost_scatter.update_layout(height=400)
    st.plotly_chart(fig_cost_scatter, use_container_width=True)

with col_cost2:
    fig_profit_unit = px.scatter(
        df_filtered,
        x='Cost per Unit',
        y='Profit per Unit',
        color='Margin %',
        size='Sales',
        hover_name='Product Name',
        color_continuous_scale='RdYlGn',
        title='Cost/Unit vs Profit/Unit'
    )
    fig_profit_unit.update_layout(height=400)
    st.plotly_chart(fig_profit_unit, use_container_width=True)

# Cost efficiency table
st.markdown("**Cost Efficiency Analysis**")
cost_analysis = df_filtered.groupby('Product Name').agg({
    'Cost': 'mean',
    'Sales': 'mean',
    'Margin %': 'mean',
    'Profit per Unit': 'mean'
}).reset_index().sort_values('Margin %', ascending=True).head(10)

st.warning("**Products with Lowest Margins (Potential repricing/renegotiation needed):**")
st.dataframe(
    cost_analysis.style.format({
        'Cost': '${:,.2f}',
        'Sales': '${:,.2f}',
        'Margin %': '{:.2f}%',
        'Profit per Unit': '${:.2f}'
    }).highlight_min(subset=['Margin %'], color='salmon'),
    use_container_width=True
)

st.markdown("---")

# =============================
# SECTION 7: TIME SERIES ANALYSIS
# =============================
st.subheader("📉 Profitability Trends Over Time")

# Monthly aggregation
df_time = df_filtered.copy()
df_time['YearMonth'] = df_time['Order Date'].dt.to_period('M')
monthly_trend = df_time.groupby('YearMonth').agg({
    'Sales': 'sum',
    'Gross Profit': 'sum',
    'Margin %': 'mean'
}).reset_index()
monthly_trend['YearMonth'] = monthly_trend['YearMonth'].astype(str)

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    fig_trend_sales = px.line(
        monthly_trend,
        x='YearMonth',
        y='Sales',
        title='Sales Trend',
        markers=True
    )
    fig_trend_sales.update_layout(height=350)
    st.plotly_chart(fig_trend_sales, use_container_width=True)

with col_trend2:
    fig_trend_profit = px.line(
        monthly_trend,
        x='YearMonth',
        y='Gross Profit',
        title='Profit Trend',
        markers=True
    )
    fig_trend_profit.update_traces(line=dict(color='green'))
    fig_trend_profit.update_layout(height=350)
    st.plotly_chart(fig_trend_profit, use_container_width=True)

st.markdown("---")

# =============================
# SECTION 8: PREDICTIVE ANALYTICS
# =============================
st.subheader("🤖 Profit Prediction Model")

model_df = df_filtered[['Cost', 'Units', 'Sales', 'Gross Profit']].dropna()

if len(model_df) > 10:
    X = model_df[['Cost', 'Units', 'Sales']]
    y = model_df['Gross Profit']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        st.markdown(f"**Model Performance (R² Score): {r2:.4f}**")
        
        col_input1, col_input2, col_input3 = st.columns(3)
        with col_input1:
            cost_input = st.number_input("Cost ($)", value=100.0, min_value=0.0)
        with col_input2:
            units_input = st.number_input("Units", value=10, min_value=1)
        with col_input3:
            sales_input = st.number_input("Sales ($)", value=200.0, min_value=0.0)
        
        if cost_input > 0 and sales_input > 0:
            prediction = model.predict([[cost_input, units_input, sales_input]])
            st.success(f"**Predicted Gross Profit: ${prediction[0]:.2f}**")
    
    with col_pred2:
        fig_actual_pred = px.scatter(
            x=y,
            y=y_pred,
            labels={'x': 'Actual Profit', 'y': 'Predicted Profit'},
            title='Actual vs Predicted Profit',
            trendline='ols'
        )
        fig_actual_pred.update_layout(height=300)
        st.plotly_chart(fig_actual_pred, use_container_width=True)
else:
    st.info("Not enough data for predictive modeling")

st.markdown("---")

# =============================
# SECTION 9: DATA EXPORT
# =============================
st.subheader("📥 Data Export & Download")

# Prepare summary reports
summary_report = df_filtered.groupby('Product Name').agg({
    'Sales': 'sum',
    'Gross Profit': 'sum',
    'Margin %': 'mean',
    'Profit per Unit': 'mean',
    'Units': 'sum',
    'Division': 'first'
}).reset_index().sort_values('Gross Profit', ascending=False)

csv_report = summary_report.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📊 Download Product Summary Report (CSV)",
    data=csv_report,
    file_name=f"Nassau_Candy_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

st.markdown("---")

# =============================
# FOOTER
# =============================
st.markdown("""
---
### 📝 Project Information
**Nassau Candy Distributor - Product Line Profitability & Margin Performance Analysis**

This dashboard provides comprehensive insights into:
- Product-level profitability metrics
- Division performance analysis  
- Margin risk identification
- Cost structure diagnostics
- Profit concentration analysis (Pareto)
- Trend analysis and predictive modeling

**Data Last Updated**: Processing your uploaded dataset
**Dashboard Version**: 2.0 - Full Analysis Suite
""")

st.markdown("✅ **Dashboard Ready for Submission**")
