'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
from utils import utils
import pandas as pd
import random

def get_form_values(columns):
    return {col: st.session_state[col] for col in columns}

def load_model():
    model = st.session_state['model']
    return utils.load_model('./train_results/' + model)

def set_values_randomly(df):
    columns = filter(lambda col: col not in ['grav_mean'], df.columns)
    for col in columns:
        values = df[col].unique()
        values.sort()
        val = random.choice(values)
        st.session_state[col] = val
    st.session_state['grav_mean'] = None
    
def set_values_from_existing(df):
    columns = filter(lambda col: col not in ['grav_mean'], df.columns)
    idx = random.choice(df.index)
    selected = df.iloc[idx]
    
    for col in columns:
        st.session_state[col] = selected[col]
    
    st.session_state['grav_mean'] = selected['grav_mean']
    return selected['grav_mean']

@st.cache_data()
def load_csv():
    df = utils.load_csv('./models/Road accidents - model v1.csv')
    return utils.cleans_data(df)
     
st.markdown("# Testing")
st.sidebar.markdown("# Testing")


df = load_csv()

col1, col2 = st.columns([0.25, 1])
with col1:    
    random_button = st.button('Select random')
    if random_button:
        set_values_randomly(df)

with col2:
    select_exists = st.button('Select random existing')
    if select_exists:
        grav_mean = set_values_from_existing(df)
        'Existing grav_mean', grav_mean
    
with st.form('myform'):
    disp_cols = st.columns(4)
    columns = filter(lambda col: col not in ['grav_mean', 'nb_usagers', 'nb_vehicules'], df.columns)
    for idx, col in enumerate(columns):
        values = df[col].unique()
        values.sort()
        with disp_cols[idx // 3]:
            st.select_slider('Select a value of ' + col, options=values, key=col)
    
    
    disp_cols = st.columns(2)
    for idx, col in enumerate(['nb_usagers', 'nb_vehicules']):
        values = df[col].unique()
        values.sort()
        with disp_cols[idx // 1]:
            st.select_slider('Select a value of ' + col, options=values, key=col)
    
    st.radio('Select a model', ['LinearRegression', 'Ridge', 'Lasso', 'DecisionTree', 'RandomForest', 'KNN'], key='model', horizontal=True)    
    
    submitted = st.form_submit_button('Predict')
    if submitted:
        form_values = get_form_values(filter(lambda col: col != 'grav_mean', df.columns))
        #st.write(form_values)
        df = pd.DataFrame(data=form_values, index=[0])
        df = utils.convert_categorical(df)
        df
        model = load_model()
        df2 = pd.DataFrame([], columns=model.feature_names_in_)
        df3 = pd.concat([df2, df])
        df3 = df3.fillna(0)
        predicted = model.predict(df3)
        'Predicted value', predicted[0]
        if st.session_state['grav_mean']:
            'True value', st.session_state['grav_mean']
    
