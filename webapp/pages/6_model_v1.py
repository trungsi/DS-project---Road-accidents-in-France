'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st
from utils import utils

st.markdown("# Model v1 generation 🎈")


button = st.button('Generate regression model')
if button:

    # load carateristiques files
    'Load carateristiques files'
    df_carac = utils.read_caracteristiques_files()
    st.write(df_carac.head())
    utils.save_csv(df_carac.head(), './analysis/caracteristiques_first_5_rows.csv')
    df_carac.info()
    
    'convert department column to correct format'
    df_carac = utils.cleans_departments(df_carac)
    st.write(df_carac.dep.unique())
    
    'add new date related infos (date, month, days_in_year, weeks_in_year) for later analysis'
    df_carac = utils.enrich_dates_info(df_carac)
    
    'load usagers files'
    df_usagers = utils.read_usagers_files()
    st.write(df_usagers.head())
    utils.save_csv(df_usagers.head(), './analysis/usagers_first_5_rows.csv')
    
    'load vehicules files'
    df_vehicules = utils.read_vehicules_files()
    st.write(df_vehicules.head())
    utils.save_csv(df_vehicules.head(), './analysis/vehicules_first_5_rows.csv')
    
    'load lieux files'
    df_lieux = utils.read_lieux_files()
    st.write(df_lieux.head())
    utils.save_csv(df_lieux.head(), './analysis/lieux_first_5_rows.csv')
    
    'plot number of accidents by department'
    fig = utils.plot_accident_count_by_departments(df_carac)
    st.pyplot(fig)
    fig.savefig('./analysis/acc_count_by_dep.png')
    'Paris (75) is the region with the highest frequency of road accidents'
    
    'plot number of accidents by atmosphere condition'
    fig = utils.plot_accident_count_by_atmosphere_conditions(df_carac)
    st.pyplot(fig)
    fig.savefig('./analysis/acc_count_by_atm.png')
    '''Atmosphere conditions doesn’t seem to impact the road accidents frequency.
    Most accidents happen in normal condition (1).'''
    
    '''Multiple passengers can be impacted by an accident. The severity of each passenger is classified by
    
    1: Ininjured
    2: Killed
    3: Injured, in hospital
    4: Sligtly injured
    -1: Uknown
    The overall severity of an accident can be seen as an aggregation of severity of all passengers impacted.
    
    The severity can be converted to numerical values as below.
    
    Severity    Value
    Ininjured (1), Unknown (-1)    0
    Slightly injured (4)    3
    Injured, in hispital (3)    7
    Killed (2)    13
    It could be an naive approach of using values in Fibonacci suite for severity calculation.
    
    Then we can use regression to predict the severity. It could then be used in insurance to predict the claim costs.'''
    
    'calculate severity (total and mean)'
    df_acc = utils.calculate_severity_for_accident(df_usagers, df_carac)
    fig = utils.plot_severity_by_number_of_pessengers(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_by_number_of_passengers.png')
    
    'In the above graphs, the average severity looks less dependent on number of passengers than total severity. We can use grav_mean as target variable for prediction.'
    
    'relation between atmosphere and severity'
    fig = utils.plot_severity_by_atmosphere_condition(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_by_atm.png')
    
    'relation between type of collision and severity'
    fig = utils.plot_severity_by_collision_type(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_by_col.png')
    'From above graphs, we can see that outlier effect is less with grav_mean than with grav_total.'
    
    'No collision (7) accident showed the highest road accident severity, followed by the frontal collision (1) of two vehicles'
    
    '''Detailed view of number of accidents by month and year.
    It could help to detect seasonality aspects.'''
    fig = utils.plot_accident_count_by_month_and_year(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/acc_count_by_month_year.png')
    '''
    We can observe some trend on number of accidents over period of 12 months.
    
    Uptrend: from Feb to June and from Aug to Oct.
    
    Downtren: from Oct to Feb and from June to Aug.
    '''
    
    'weekly view on severity of accidents'
    fig = utils.plot_severity_by_week(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_by_week.png')
    
    '''
    The period of the year apparently has no influence on the severity whereas the frequency of accidents appears to be higher in the period before and after the summer season with the highest peak the first week of October
    '''
    
    'add lieux info'
    df_acc = utils.add_lieux_info_to_accidents(df_lieux, df_acc)
    
    'relation between road surface and severity'
    fig = utils.plot_severity_by_road_surface(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_by_surf.png')
    
    'The severity is likely to be higher when there is flood or snow in the road surface'
    
    'add vehicules info'
    df_acc = utils.add_vehicule_info_to_accidents(df_vehicules, df_acc)
    
    '''
    some columns (adr, gps, lat, long and vma) contains a lot of NA. So dropna will largely reduce data size.
    It's better ignoring theses (drop columns) when preparing data for machine learning.
    '''
    
    
    grav_by_deps = df_acc[['dep', 'grav_mean', 'grav_total', 'nb_usagers']].groupby('dep').agg(['count', 'sum', 'mean'])
    grav_by_deps = grav_by_deps.drop(columns=[grav_by_deps.columns[0], grav_by_deps.columns[1], 
                                               grav_by_deps.columns[3], grav_by_deps.columns[8]])
    grav_by_deps.columns = ['grav_mean_mean', 'grav_total_total', 'grav_total_mean', 'acc_count', 'nb_usagers_total']
    grav_by_deps = grav_by_deps.reset_index()
                           
    st.write(grav_by_deps.sort_values(by='grav_total_mean', ascending=False).head())
    
    fig = utils.plot_severity_histogram(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/severity_hist.png')
    
    '''
    mean gravity varies from 0 to 13 with mean around 3 and majority (75%) are less than 4
    Data is skewed It may be difficult to detect outlier.
    '''
    fig = utils.plot_correlation_heatmap(df_acc)
    st.pyplot(fig)
    fig.savefig('./analysis/corr_heatmap.png')
    
    '''
    Number of usagers is highly influent on gravity.
    The Location (agglomeration or hors agglomeration) is the second most important feature.
    Collision type comes next as the third important feature.
    All other features are not important. Can they be excluded from the model ?
    '''
    
    'Save the model'
    utils.save_csv(df_acc, './models/Road accidents - model v1.csv')