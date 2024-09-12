import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv('FMCG_data.csv')
df['workers_num'] = df['workers_num'].fillna(df['workers_num'].median())
df.drop(columns=['wh_est_year'] , inplace = True)

st.title('Inventory Management')
st.image('Homepage.jpg')
st.write("""
#### Hello and welcome to my project! This application is designed to provide insightful analysis and predictive modeling for optimizing supply chain operations in the FMCG industry.

**Overview:**

- **Introduction**: Understand the background and objectives of our project.
- **Exploratory Data Analysis (EDA)**: Dive into the data to uncover hidden patterns and trends.
- **Machine Learning Models**: Explore various predictive models we've developed to enhance supply chain efficiency.

Thank you for visiting, and we hope you find this project informative and engaging!
""")



