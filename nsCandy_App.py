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
    df = pd.read_csv("Nassau Candy Distributor.csv")
    return df
try:
    df = load_data()
except:
    st.error("Dataset Not Found!")
    st.stop()
    