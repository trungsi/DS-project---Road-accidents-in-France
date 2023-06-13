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

def load_train_result_summary():
    #with open('./train_results/train_summaries.json') as f:
    #    return json.load(f)
    df = pd.read_json('./train_results/train_summaries.json')
    df.drop_duplicates(subset=['estimator'], keep='last', inplace=True)
    df.insert(0, 'name', df.estimator.apply(get_name))
    df = df[['name', 'rmse', 'train_score', 'test_score', 'train_duration', 'test_duration']]
    return df

st.set_page_config(layout="wide")
st.markdown("# Modeling")
st.sidebar.markdown("# Modeling")

st.subheader('Input data')

df = utils.cleans_data(utils.load_csv('./models/Road accidents - model v1.csv'))

buff = io.StringIO()
df.info(buf=buff)

col1, col2 = st.columns([1, 3])
with col1:
    st.text(buff.getvalue())
with col2:
    st.markdown('- Target variable is grav_mean, most of the columns (except nb_usagers, nb_vehicules) are categorical.')
    st.markdown('- Data scaling is applied to harmonize numerical values.')
    
st.subheader('3 linear (Linear, Ridge[CV] and Lasso[CV]), 2 non parametric (DecisionTree and KNN) and 1 ensemble models have been used during training.')
st.divider()

train_result_summ = load_train_result_summary()
'# Train result summary'
train_result_summ

st.subheader('All models give more or less the same result on accuracy and error.')
st.subheader('Accuracy score is rather low for all (~30%). RMSE is around 1.9 within real range [0-13].')
st.subheader('RandomForest was very slow in training and KNN very slow on testing.')

'# Train result charts'
col1, col2, col3 = st.columns(3)

with col1:
    st.line_chart(train_result_summ, x='name', y=['rmse'])

with col2:
    st.line_chart(train_result_summ, x='name', y=['train_score', 'test_score'])
    
with col3:
    st.line_chart(train_result_summ, x='name', y=['train_duration', 'test_duration'])

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

