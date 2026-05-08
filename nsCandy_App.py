import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Nassau Candy Distributor.csv")
    return df

df = load_data()
print(df.head())