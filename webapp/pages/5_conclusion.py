'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("# Conclusion")
st.sidebar.markdown("# Conclusion")
st.subheader('Regression problem')
st.markdown('- Same and low performance with all models')
st.markdown('- Errors (RMSE) is nevertheless low (2/13)')
st.markdown('- Is it due to target selection ? Or (potentially important) features were ignored ?')
st.markdown('- Some models run slower with respect to others but give the same performance at the end. How can it be avoided ?')

st.subheader('Classification problem')
st.markdown('- Due to imbalanced data, minor class could not be predicted with high score.')
