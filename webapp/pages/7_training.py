'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
from utils import utils

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, RidgeClassifier
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
    
def create_classifier(model):
    if model == 'LinearRegression':
        raise Exception('Not yet implemented')
    if model == 'Ridge':
        return RidgeClassifier()
    if model == 'Lasso':
        raise Exception('Not yet implemented')
    if model == 'DecisionTree':
        raise Exception('Not yet implemented')
    if model == 'RandomForest':
        raise Exception('Not yet implemented')
    if model == 'KNN':
        raise Exception('Not yet implemented')

def train(model, regression=True):
    if regression:
        estimator = create_regressor(model)
        
        df = utils.load_csv('./models/Road accidents - model v1.csv')
        df = utils.prepare_data_model_v1(df)
        
        train_result_slr = utils.train(df, estimator, scaler)
        
    else:
        estimator = create_classifier(model)
        
        df = utils.load_csv('./models/Road accidents - model v2.csv')
        df = utils.prepare_data_model_v2(df)
        
        train_result_slr = utils.train(df, estimator, None, target_col='grav')
        
    return train_result_slr
    
def save_result(model, train_result, regression=True):
    path = './train_results/' if regression else './train_results_classification/'
       
    utils.save_train_summary(train_result, path + 'train_summaries.json')
    utils.save_train_result_charts(train_result, path + model)
    utils.save_trained_model(train_result, path + model)
    
st.markdown("# Train the model")

model_list = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'KNN']
with st.form('my form'):
    radio_val = st.radio('Please select a model', model_list)
    regression = st.radio('', options=['Regression', 'Classification'])
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner('Training the model with ' + radio_val + '...'):
            train_result = train(radio_val, regression=='Regression')
        
        st.write(train_result)
        st.pyplot(utils.plot_results_histogram(train_result))
        st.pyplot(utils.plot_results_scatter_chart(train_result))
        
        save_result(radio_val, train_result, regression=='Regression')
        'Done.'