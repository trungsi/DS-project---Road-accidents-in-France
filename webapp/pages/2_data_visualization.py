'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st

st.set_page_config(layout="wide")
st.markdown("# Data visualization")
st.sidebar.markdown("# Data visualization")

st.image('./analysis/acc_count_by_dep.png', caption='Top ten departments with most accidents')
st.subheader('''Paris (75) is the region with the highest frequency of road accidents''')
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.image('./analysis/acc_count_by_atm.png', caption='Accident count by atmosphere condition')
with col2:
    st.markdown(''' 
    - 1: Normal
    - 2: Slight rain
    - 3: Heavy rain
    - 4: Snow
    - 5: Fog
    - 6: Strong wind / storm
    - 7: Dazzling
    - 8: Cloudy
    - 9: Other
    ''')
st.subheader('''Atmosphere conditions doesn't seem to impact the road accidents frequency. \nMost accidents happen in normal condition (1).''')
st.divider()

st.image('./analysis/acc_count_by_month_year.png', caption='Accident count by month and year')
st.subheader('We can observe some trend on number of accidents over period of 12 months.')

st.subheader('Uptrend: from Feb to June and from Aug to Oct.')
st.subheader('Downtren: from Oct to Feb and from June to Aug.')
st.divider()

st.image('./analysis/severity_by_number_of_passengers.png', caption='Severity by number of passengers')
st.image('./analysis/severity_by_atm.png', caption='Severity by atmosphere condition')
st.subheader('In the above charts which shows the histogram of severity (total and mean) as function of number of passengers and atmosphere condition.')
st.subheader('We can observe the outlier effect is more important with total value than with mean value. So it might be better to use mean value as target variable for prediction.')
#st.image('./analysis/severity_by_col.png', caption='Severity by collision type')
st.divider()


st.image('./analysis/severity_by_week.png', caption='Severity by week')
st.subheader('The period of the year apparently has no influence on the severity whereas the frequency of accidents appears to be higher in the period before and after the summer season with the highest peak the first week of October')
st.divider()

st.image('./analysis/severity_by_surf.png', caption='Severity by road surface')
st.subheader('The severity is likely to be higher when there is flood or snow in the road surface')
st.divider()

st.image('./analysis/severity_hist.png', caption='Severity histogram')
st.subheader('mean gravity varies from 0 to 13 with mean around 3 and majority (75%) are less than 4. Data is skewed It may be difficult to detect outlier.')

st.image('./analysis/corr_heatmap.png', caption='Correlation heatmap')

