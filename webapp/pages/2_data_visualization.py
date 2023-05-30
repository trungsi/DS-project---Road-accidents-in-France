'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st

st.markdown("# Data visualization")
st.sidebar.markdown("# Data visualization")

st.image('./analysis/acc_count_by_dep.png', caption='Accident count by department')
'Paris (75) is the region with the highest frequency of road accidents'
'plot number of accidents by atmosphere condition'


st.image('./analysis/acc_count_by_atm.png', caption='Accident count by atmosphere condition')
'''Atmosphere conditions doesn't seem to impact the road accidents frequency.
Most accidents happen in normal condition (1).'''

st.image('./analysis/severity_by_number_of_passengers.png', caption='Severity by number of passengers')

st.image('./analysis/severity_by_atm.png', caption='Severity by atmosphere condition')

st.image('./analysis/severity_by_col.png', caption='Severity by collision type')

st.image('./analysis/acc_count_by_month_year.png', caption='Accident count by month and year')
'''
We can observe some trend on number of accidents over period of 12 months.

Uptrend: from Feb to June and from Aug to Oct.

Downtren: from Oct to Feb and from June to Aug.
'''

st.image('./analysis/severity_by_week.png', caption='Severity by week')
'''
The period of the year apparently has no influence on the severity whereas the frequency of accidents appears to be higher in the period before and after the summer season with the highest peak the first week of October
'''

st.image('./analysis/severity_by_surf.png', caption='Severity by road surface')
'The severity is likely to be higher when there is flood or snow in the road surface'

st.image('./analysis/severity_hist.png', caption='Severity histogram')
'''
mean gravity varies from 0 to 13 with mean around 3 and majority (75%) are less than 4
Data is skewed It may be difficult to detect outlier.
'''

st.image('./analysis/corr_heatmap.png', caption='Correlation heatmap')

