'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
from utils import utils

'Hello World!!!'

df_carac = utils.read_caracteristiques_files()
st.write(df_carac.head())