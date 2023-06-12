'''
Created on 30 mai 2023

@author: trand
'''
import streamlit as st

st.sidebar.markdown("# Introduction")

"# Road accidents in France"

'''
The objective of the project is to predict the severity of road accidents in France.

French gorvement is providing open datasets on road accidents in France from 2005. 
The data is available at this link https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2021/.

A road accident is defined by criteria below:
- happens on public road or private but be open to public traffic
- involve at least a victim
- involve at least a vehicule

Accident information are composed of 4 files:
- CARATERISTIQUES (Characteristics in French): which contains general information about accident 
    - date
    - time
    - department
    - longitude, latitude
    - collision type
    - ...
- LIEUX (Location in French): which provide more details about location 
    - road types : highway, secondary road, country-side road,...
    - circulation types: single track, 2-lane, 3-lane,...
    - road surface condition: wet, flooded, snowed,...
    - ...
- VEHICULES: which describe all vehicules involed in the accidents
    - vehicule category: car, bus, moto, bicycle,...
    - fixed obstacle: (not moving) vehicule, tree, wall,...
    - mobile obstacle: (moving) vehicule, pedestrian, animal,...
    - initial hit position: in-front, back, left side, right side,...
    - ...
- USAGERS (Passengers in French): describe all passengers impacted by the accident
    - category: driver, passenger, pedestrian
    - severity:
        - 1-Indemne (Unharmed) : does not need any medical care
        - 2-Tue (Died)
        - 3-Hospitalise (Injured): injured who need care in hospital
        - 4-Blesse (Slightly injured): injured but do not require particular care in hospital
    - place: exact location inside car, bus, train,...
    - sexe
    - birth year
    - ...
    
These files grouped by year are in CSV format and are linked by Num_Acc column.    
'''

st.image(image='./csv-files.PNG')

'''
In order to facilitate the processing and also the collaboration, all csv files were downloaded and put into Github repository https://github.com/trungsi/DS-project---Road-accidents-in-France/tree/master/webapp/files.
All source codes (processing and modeling) are also put in the same Github. 
'''

'# Data processing'
'The csv files even produced by French authority still contain some issues which will require custom processing'
'''
- file name contains sometimes underscore, sometimes hyphen
- csv column delimiter are sometimes semi-column (;), virgule (,) or tab (\t)
- department column format is not consistent: 75 vs 075, 201 vs 2A,...
- date column format is not consistent : year 2006 vs 06 vs 6,...
'''

'# Modeling'
'''
As you can see above, an accident may imply one or several victims with different severity degree. 
In order to observe the severity of an accident as a whole, we try to extrapolate the severity of each passenger to calculate overall severity (score) with the rule below.
|Severity Category|Severity Score|
|Ininjured (1)|0|
|Slightly injured (4)|3|
|Injured, in hospital (3)|7|
|Died (2)|13|

Hence to overall severity score of an accident can be the sum of severity score of all victims or the average (mean) score.
Example: an accident having 4 victims with 4 different severity categories will total score = 23 (0+3+7+13) and mean score = 5,75 (23/4)

Then the first attempt is to predict the severity score which is a regression problem. 
As information are aggregated at accident level, information about vehicules (except vehicule count) and passengers (except passengers count) are then ignored. 
'''

'''
Another idea is to instead of predicting the overall severity, we can predict the severity of each passenger involved in the accident.
Then the prediction will be classification problem and we can use all information (vehicules and passengers) in the model.
'''

