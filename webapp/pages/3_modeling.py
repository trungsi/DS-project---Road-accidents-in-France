'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
#import json
import pandas as pd
import re

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

st.markdown("# Modeling")
st.sidebar.markdown("# Modeling")


train_result_summ = load_train_result_summary()
'# Train result summary'
train_result_summ

'# Train result charts'
col1, col2, col3 = st.columns(3)

with col1:
    st.line_chart(train_result_summ, x='name', y=['rmse'])

with col2:
    st.line_chart(train_result_summ, x='name', y=['train_score', 'test_score'])
    
with col3:
    st.line_chart(train_result_summ, x='name', y=['train_duration', 'test_duration'])

