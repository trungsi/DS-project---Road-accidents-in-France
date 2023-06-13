'''
Created on 13 juin 2023

@author: TRAND
'''
import streamlit as st
from utils import utils

st.markdown("# Model v2 generation")

button = st.button('Generate classification model')
if button:
    'Load carateristiques files'
    df_carac = utils.read_caracteristiques_files()
    st.write(df_carac.head())
    #utils.save_csv(df_carac.head(), './analysis/caracteristiques_first_5_rows.csv')
    df_carac.info()
    
    'convert department column to correct format'
    df_carac = utils.cleans_departments(df_carac)
    st.write(df_carac.dep.unique())
    
    'add new date related infos (date, month, days_in_year, weeks_in_year) for later analysis'
    df_carac = utils.enrich_dates_info(df_carac)
    
    'load usagers files'
    df_usagers = utils.read_usagers_files()
    st.write(df_usagers.head())
    #utils.save_csv(df_usagers.head(), './analysis/usagers_first_5_rows.csv')
    
    'load vehicules files'
    df_vehicules = utils.read_vehicules_files()
    st.write(df_vehicules.head())
    #utils.save_csv(df_vehicules.head(), './analysis/vehicules_first_5_rows.csv')
    
    'load lieux files'
    df_lieux = utils.read_lieux_files()
    st.write(df_lieux.head())
    #utils.save_csv(df_lieux.head(), './analysis/lieux_first_5_rows.csv')
    
    'merge data'
    df_merged = df_usagers \
            .merge(df_vehicules, on=['Num_Acc', 'num_veh']) \
            .merge(df_lieux, on='Num_Acc') \
            .merge(df_carac, on='Num_Acc')
            
    fig = utils.plot_severity_histogram(df_merged, x='grav')
    st.pyplot(fig)
    fig.savefig('./analysis/severity_histogram_v2.png')
    
    'remove not useful columns'
    # remove not useful columns
    df_merged = df_merged.drop(columns=[
                  # code, id columns
                  'Num_Acc', 'num_veh', 'id_vehicule_x', 'id_vehicule_y',
                  # address is free text so will most likely be unique. Remove it for now
                  'adr', 'voie', 'nbv', 'vosp',
                  # there're around 1000 communes (categories). So training will be very slow. Remove it for now
                  'com',
                  # remove time variables for now
                  'an', 'mois', 'jour', 'hrmn', 'an_nais'
                  ])
    
    'remove more columns'
    df_merged = df_merged.drop(columns=['secu1', 'secu2', 'secu3', 'motor', 'v1', 'v2', 'pr', 'pr1', 'vma', 'gps', 'lat', 'long'])

    'data cleansing'
    str_columns = df_merged.columns.drop(['lartpc', 'larrout', 'occutc'])
    print(str_columns)
    
    df_merged[str_columns] = df_merged[str_columns].astype(str)
    df_merged = df_merged.replace(['-1', ' -1', 'nan', '-1.0', '0'], '0.0').fillna(0)
    
    def transform_row(row):
        return row.apply(lambda v: float(v.replace(',', '.')) if type(v) == str else v)
    
    df_merged[['lartpc', 'larrout', 'occutc']] = (
        df_merged[['lartpc', 'larrout', 'occutc']].apply(transform_row))
    
    'reorder columns'
    columns = df_merged.columns
    re_ordered_columns = columns.drop('grav').insert(0, 'grav')
    df_merged = df_merged[re_ordered_columns]
    
    'save the model'
    path = './models/Road accidents - model v2.csv'
    'done'
    utils.save_csv(df_merged, path)
    