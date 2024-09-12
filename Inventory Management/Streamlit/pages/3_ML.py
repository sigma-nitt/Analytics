import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.markdown("<h4>Feature Engineering</h4>",unsafe_allow_html=True)
st.markdown("<p>to accurately predict the number of SKUs (Stock Keeping Units) for each warehouse, it was necessary to create a new feature called num_SKUs in our dataset. This feature engineering step involves several transformations and calculations to ensure the data is suitable for model training and prediction. Below is the process used to generate the num_SKUs feature:</P>",unsafe_allow_html=True)
code = """
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}
df['WH_capacity_size'] = df['WH_capacity_size'].map(size_mapping)
df['WH_capacity_size'] = pd.to_numeric(df['WH_capacity_size'], errors='coerce')
df['WH_capacity_size'].fillna(0, inplace=True)
df['num_SKUs'] = df.apply(lambda row: row['product_wg_ton'] / 100, axis=1)
"""
st.code(code , language='python')

st.markdown("<h4 style = 'margin-bottom:0'>Data Preparation</h4>",unsafe_allow_html=True)
st.markdown("<p>To build our model, we first define 'num_SKUs' as the target feature (y). The independent features (X) include various predictors such as 'num_refill_req_l3m', 'zone', 'Location_type', and more. We use one-hot encoding to convert categorical variables ('zone', 'Location_type', 'WH_regional_zone') into a numerical format. Finally, we split the data into training and testing sets to train the model and evaluate its performance, ensuring reliable predictions on unseen data. The implementation steps are shown in the code below.</p>",unsafe_allow_html=True)
code = """from sklearn.model_selection import train_test_split

# Features and target
X = df[['num_refill_req_l3m',  'govt_check_l3m', 'zone' ,'temp_reg_mach', 'Location_type' , 'Competitor_in_mkt' ,  'distributor_num',  'storage_issue_reported_l3m' ,'wh_breakdown_l3m' , 'product_wg_ton' , 'WH_regional_zone' , 'transport_issue_l1y' ]]
y = df['num_SKUs']

# One-hot encode categorical features
X = pd.get_dummies(X, columns=['zone', 'Location_type', 'WH_regional_zone' ], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""
st.code(code,language='python')

st.markdown("<h4>Machine Learning Models</h4>",unsafe_allow_html=True)
model_choice = st.selectbox("choose a machine learning model:" , ('Linear Regression','Ridge and Lasso with cross validation' , 'Random Forest' , 'KNN Regressor' , 'XGBoost'))

if model_choice == 'Linear Regression':
    st.markdown("<h5>Linear Regression</h5>",unsafe_allow_html=True)
    code = """
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred_reg = reg_model.predict(X_test)
mse_reg = mean_squared_error(y_test, y_pred_reg)
rmse_reg = np.sqrt(mse_reg)
r2_reg = r2_score(y_test, y_pred_reg)
print('Regression Model MSE:', mse_reg)
print('Regression Model RMSE:' ,rmse_reg)
print('R2 Score:', r2_reg)
"""
    st.code(code,language='python')
    st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
    st.markdown("""
<p style='margin-bottom:0'>Regression Model MSE: 9.34015855620492e-27</P>
<p style ='margin-bottom:0' >Regression Model RMSE: 9.664449573672016e-14</p>
<p>R2 Score: 1.0</p>""",unsafe_allow_html=True)
 
elif model_choice == 'Ridge and Lasso with cross validation':
    st.markdown("<h5>Ridge and Lasso with cross validation</h5>",unsafe_allow_html=True)
    code = """
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
lasso_cv_regressor = LassoCV(alphas=[0.1, 0.01, 0.001])
ridge_cv_regressor = RidgeCV(alphas=[0.1, 1.0])

lasso_cv_regressor.fit(X_train, y_train)
ridge_cv_regressor.fit(X_train, y_train)

best_alpha = lasso_cv_regressor.alpha_
y_pred_lasso = lasso_cv_regressor.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print('Best alpha:', best_alpha)
print('Lasso Regression MSE:', mse_lasso)
print('Lasso Regression RMSE:' ,rmse_lasso)
print('R2 Score:', r2_lasso)

best_alphaa = ridge_cv_regressor.alpha_
y_pred_ridge = ridge_cv_regressor.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print('Best alpha:', best_alphaa)
print('Ridge Regression MSE:', mse_ridge)
print('Ridge Regression RMSE:' ,rmse_ridge)
print('R2 Score:', r2_ridge)
"""
    st.code(code,language='python')
    st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
    st.markdown("""<p style = 'margin-bottom:0'>Best alpha: 0.001</p>
<p style='margin-bottom:0'>Lasso Regression MSE: 7.34508963244308e-15</p>
<p style = 'margin-bottom:0'>Lasso Regression RMSE: 8.570349836758754e-08</p>
<p style = 'margin-bottom:0'>R2 Score: 1.0</p>
<p style = 'margin-bottom:0'>Best alpha: 0.1</P>
<p style = 'margin-bottom:0'>Ridge Regression MSE: 5.585840500566797</P>
<p style = 'margin-bottom:0'>Ridge Regression RMSE: 2.363438279407101</P
<p style = 'margin-bottom:0'>R2 Score: 0.9995825226888752</P>""" , unsafe_allow_html=True)

elif model_choice == 'Random Forest':
    st.markdown("<h5>Random Forest</h5>",unsafe_allow_html=True)
    code = """
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)
y_pred_rf = model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print('Random Forest MSE:', mse_rf)
print('Random Forest RMSE:' ,rmse_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print('R2 Score:', r2_rf)
"""
    st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
    st.code(code,language='python')
    st.markdown("""
<p style='margin-bottom:0'>Random Forest MSE: 4.240045037938562e-05</p>
<p style='margin-bottom:0'>Random Forest RMSE: 0.0065115628215802095</p>
<p style='margin-bottom:0'>R2 Score: 0.9999999968310541</p>
""", unsafe_allow_html=True)
    
elif model_choice == 'KNN Regressor':
    st.markdown("<h5>KNN Regressor<h5>",unsafe_allow_html=True)
    code = """from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors=3)
knn_regressor.fit(X_train, y_train)

y_pred_knn = knn_regressor.predict(X_test)

mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)
print('KNN MSE:', mse_knn)
print('KNN Model RMSE:' ,rmse_knn)
print('R2 Score:', r2_knn)"""
    st.code(code,language='python')
    st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
    st.markdown("""
<p style='margin-bottom:0'>KNN MSE: 0.0005081066666666839</p>
<p style='margin-bottom:0'>KNN Model RMSE: 0.022541221498993437</p>
<p style='margin-bottom:0'>R2 Score: 0.9999999620248725</p>""",unsafe_allow_html=True)

if model_choice == 'XGBoost':
    st.markdown("<h5>XGBoost</h5>" , unsafe_allow_html=True)
    code = """import xgboost as xgb

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print('XGBoost MSE:', mse_xgb)
print('XGBoost Model RMSE:' ,rmse_xgb)
print('R2 Score:', r2_xgb)"""
    st.code(code,language='python')
    st.markdown("""<h5 style= 'margin-bottom:0;'>Output:</h5>""",unsafe_allow_html=True)
    st.markdown("""
<p style='margin-bottom:0'>XGBoost MSE: 2.808798604503647</p>
<p style='margin-bottom:0'>XGBoost Model RMSE: 1.6759470768803073</p>
<p style='margin-bottom:0'>R2 Score: 0.9997900746201435</p>""", unsafe_allow_html=True)
    
