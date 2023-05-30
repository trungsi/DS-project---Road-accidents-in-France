'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
from utils import utils

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def create_regressor(model):
    if model == 'LinearRegression':
        return LinearRegression()
    if model == 'Ridge':
        return RidgeCV(alphas=[0.001, 0.01, 1, 10, 50, 1000, 100, 0.1, 0.3, 0.7])
    if model == 'Lasso':
        return LassoCV(cv=10)
    if model == 'DecisionTree':
        return DecisionTreeRegressor()
    if model == 'RandomForest':
        return RandomForestRegressor()
    if model == 'KNN':
        return KNeighborsRegressor()
    
def train(model):
    df = utils.load_csv('./models/Road accidents - model v1.csv')
    
    df = utils.prepare_data_model_v1(df)
    
    regressor = create_regressor(model)

    train_result_slr = utils.train(df, regressor, scaler)
    return train_result_slr
    
def save_result(model, train_result):    
    utils.save_train_summary(train_result, './train_results/train_summaries.json')
    utils.save_train_result_charts(train_result, './train_results/' + model)
    utils.save_trained_model(train_result, './train_results/' + model)
    
st.markdown("# Train the model")
st.sidebar.markdown("# Train the model")

model_list = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomRorest', 'KNN']
with st.form('my form'):
    radio_val = st.radio('Please select a model', model_list)
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner('Training the model with ' + radio_val + '...'):
            train_result = train(radio_val)
        
        st.write(train_result)
        st.pyplot(utils.plot_results_histogram(train_result))
        st.pyplot(utils.plot_results_scatter_chart(train_result))
        
        save_result(radio_val, train_result)