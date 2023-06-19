'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
#import json
import pandas as pd
import re
from utils import utils
import io

def get_name(estimator):
    return re.split('CV|Regressor|\(', estimator)[0]

@st.cache_data()
def load_train_result_summary():
    #with open('./train_results/train_summaries.json') as f:
    #    return json.load(f)
    df = pd.read_json('./train_results/train_summaries.json')
    df.drop_duplicates(subset=['estimator'], keep='last', inplace=True)
    df.insert(0, 'name', df.estimator.apply(get_name))
    df = df[['name', 'rmse', 'train_score', 'test_score', 'train_duration', 'test_duration']]
    return df

@st.cache_data()
def load_train_result_summary_classification():
    #with open('./train_results/train_summaries.json') as f:
    #    return json.load(f)
    df = pd.read_json('./train_results_classification/train_summary_classification.json')
    df.drop_duplicates(subset=['estimator'], keep='last', inplace=True)
    df.insert(0, 'name', df.estimator.apply(get_name))
    df = df[['name', 'train_score', 'test_score', 'train_duration', 'test_duration', 'classification_report']]
    return df

@st.cache_data()
def get_model_v1():
    return utils.cleans_data(utils.load_csv('./models/Road accidents - model v1.csv'))

st.set_page_config(layout="wide")
st.sidebar.markdown("# Modeling")

tab1, tab2 = st.tabs(['Regression', 'Classification'])

with tab1:
    st.markdown("# Regression")
    st.subheader('Input data')
    
    df = get_model_v1()
    
    buff = io.StringIO()
    df.info(buf=buff)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.text(buff.getvalue())
    with col2:
        st.markdown('- Target variable is grav_mean, most of the columns (except nb_usagers, nb_vehicules) are categorical.')
        #st.markdown('- Data scaling is applied to harmonize numerical values.')
        
    st.subheader('3 linear (Linear, Ridge[CV] and Lasso[CV]), 2 non parametric (DecisionTree and KNN) and 1 ensemble models have been used during training.')
    st.divider()
    
    train_result_summ = load_train_result_summary()
    '# Train result summary'
    train_result_summ
    
    st.subheader('All models give more or less the same result on accuracy and error.')
    st.subheader('Accuracy score is rather low for all (~30%). RMSE is around 1.9 within real range [0-13].')
    st.subheader('RandomForest was very slow in training and KNN very slow on testing.')
    
    '# Train result charts'
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(train_result_summ, x='name', y=['rmse'])
    
    with col2:
        st.line_chart(train_result_summ, x='name', y=['train_score', 'test_score'])
        
    #with col3:
    #    st.line_chart(train_result_summ, x='name', y=['train_duration', 'test_duration'])

    st.divider()    
    '# Train results detail'
    model_list = ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'KNN']
    for model in model_list:
        st.subheader(model)
        col1, col2 = st.columns(2)
        with col1:
            st.image('./train_results/' + model + '/train_result_hist.png')
        with col2:
            st.image('./train_results/' + model + '/train_result_chart.png')
            
    st.subheader('Even all models give the same performance but non parametric models return predicted values within the true range [0-13].')
    st.subheader('While linear models sometimes return out-of-range values.')

with tab2:
    train_result_summ = load_train_result_summary_classification()
    '# Classification'
    
    st.image('./train_results_classification/severity_hist.png')
    st.subheader('Data is imbalanced. Death class is only 2%. It will be difficult to predict this class')
    
    df = train_result_summ.drop(columns=['classification_report'])
    df
    st.subheader('DecisionTree and RandomForest performed over-fitting.')
    
    '# Train results detail'
    model_list = ['Ridge', 'DecisionTree', 'RandomForest']
    for model in model_list:
        st.subheader(model)
        col1, col2 = st.columns(2)
        with col1:
            st.image('./train_results_classification/' + model + '/confusion_matrix.png')
        with col2:
            st.code('c'+ train_result_summ[train_result_summ.name.str.startswith(model)][['classification_report']].to_numpy()[0, 0])
    
    st.subheader('Only DesicionTree could predict some Death cases but the score is lower on other classes.')
    st.subheader('RandomForest gave a little better score. However, the training time is much longer.')
