import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.dataframe_explorer import dataframe_explorer

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)

st.markdown("""<h4 style='margin-bottom:0;'>Background</h4>
                <p style='margin-top:0;'>The fast-moving consumer goods (FMCG) industry is highly competitive, and efficient supply chain management is crucial for maintaining profitability and customer satisfaction. Two years ago, a FMCG company ventured into the instant noodles market. Despite the initial success, the higher management has identified a critical issue: a mismatch between supply and demand across different regions. Where demand is high, the supply is insufficient, and where demand is low, there is an oversupply. This discrepancy leads to increased inventory costs and potential loss of sales.</p>"""
                , unsafe_allow_html=True)
st.markdown("""
<h4 style='margin-bottom:0;'>Problem Statement</h4>
<p style='margin-top:0;'>The primary challenge is to optimize the supply quantities to each warehouse across the country. The goal is to ensure that the supply matches the demand as closely as possible, minimizing inventory costs and maximizing customer satisfaction. Additionally, understanding the demand patterns in various regions will help drive targeted advertising campaigns, further boosting sales in high-demand areas.</p>
""", unsafe_allow_html=True)
st.markdown("""
<h4 style='margin-bottom:0;'>Objective</h4>
<p style='margin-top:0;'>The objective of this project is twofold:
<ol>
    <li>Optimize Supply Quantities: Build a predictive model using historical data to determine the optimal weight of instant noodles to be shipped to each warehouse. This model will help in aligning the supply with the actual demand, reducing excess inventory costs, and avoiding stockouts. Specifically, we aim to predict the number of SKUs (Stock Keeping Units), which are unique identifiers for each distinct product and service that can be purchased.</li>
    <li>Analyze Demand Patterns: Conduct an in-depth analysis of demand patterns across different regions. This analysis will help the management identify high-demand pockets and design targeted advertising campaigns to boost sales in these areas.</li>
</ol>
</p>
""", unsafe_allow_html=True)
st.markdown('<h4>Dataset</h4>' , unsafe_allow_html=True)
st.info("You can filter the data according to your preferences below!")
filtered_df = dataframe_explorer(df)
st.dataframe(filtered_df)
st.markdown("<h5 style = 'margin-bottom : 0;'>Descriptive Statistics:</h5>",unsafe_allow_html=True)
st.write(filtered_df.describe())