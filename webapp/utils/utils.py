'''
Created on 30 mai 2023

@author: trand
'''
import time
from datetime import date
from time import perf_counter

import os
import sys
import math
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# from google.colab import drive

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import joblib


base_url = "https://github.com/trungsi/DS-project---Road-accidents-in-France/blob/master/"

# all csv files are uploaded to GitHub
# They are all merged to create data frames
# some custom processing due to slight change in the format
def is_local():
    return ~('google.colab' in sys.modules)

def get_file_path(prefix, year):
    if is_local():
        return './files/' + prefix + str(year) + '.csv'
    
    return base_url + prefix + str(year) + '.csv?raw=true'

def read_csv(prefix, year, sep):  
    return pd.read_csv(get_file_path(prefix, year), sep=sep, encoding='ISO-8859-1')

def read_csvs(prefix, rang, sep):
    return [read_csv(prefix, year, sep) for year in rang]

def read_all_csvs(configs):
    dfs = list(map(lambda config: read_csvs(config['prefix'], config['range'], config['sep']), configs))
    dfs = [item for sublist in dfs for item in sublist]
    return pd.concat(dfs)

def read_caracteristiques_files():
    carac_configs = [
        {'prefix': 'caracteristiques_', 'range': range(2006, 2009), 'sep': ','},
        {'prefix': 'caracteristiques_', 'range': range(2009, 2010), 'sep': '\t'},
        {'prefix': 'caracteristiques_', 'range': range(2010, 2017), 'sep': ','},
        {'prefix': 'caracteristiques-', 'range': range(2017, 2019), 'sep': ','},
        {'prefix': 'caracteristiques-', 'range': range(2019, 2022), 'sep': ';'},
    ]

    return read_all_csvs(carac_configs)

def cleans_departments(df_carac):
    print('Departments before cleansing', len(df_carac.dep.unique()), df_carac.dep.unique())
    df_carac = df_carac.astype({"dep": str})

    def clean_deps(r):
        if ((len(r) == 3) & (r.endswith('0'))):
            return r[0:-1]
        if (len(r) == 1):
            return '0' + r
        if r == '201':
            return '2A'
        if r == '202':
            return '2B'
        return r

    df_carac.dep = df_carac.dep.apply(clean_deps)
    print('Departments after cleansing', len(df_carac.dep.unique()), df_carac.dep.unique())
    return df_carac

def enrich_dates_info(df_carac):
    def convert_date_scalar(an, mois, jour):
        '''
        return (date, first_day_of_month, nb_of_days_in_year, nb_of_weeks_in_year)
        '''
        if an < 2000: an = 2000+an
        the_date = date(an, mois, jour)
        return [the_date, the_date.replace(day=1), the_date.strftime('%j'), the_date.strftime('%V')]

    start = perf_counter()
    #df_acc[['date', 'nb_jour', 'nb_semaine']] = df_acc.apply(convert_date, axis=1, result_type='expand')
    #df_acc[['date', 'nb_jour', 'nb_semaine']] = df_acc[['an', 'mois', 'jour']].apply(convert_date_array, raw=True, axis=1, result_type='expand')
    #df_acc['date'] = df_acc[['an', 'mois', 'jour']].apply(convert_date_array, raw=True, axis=1)
    #df_acc['nb_jour'] = df_acc[['date']].apply(lambda r: r.date.strftime('%j'), axis=1)
    # map version is much faster than apply
    df_carac2 = pd.DataFrame(map(convert_date_scalar, df_carac.an, df_carac.mois, df_carac.jour), 
                          columns=['date', 'month', 'nb_jour', 'nb_semaine'], index=df_carac.index)
    #df_carac2 = df_carac2.astype({'date': 'datetime64[ns]', 'month': 'datetime64[ns]', 'nb_jour': 'int32', 'nb_semaine': 'int32'})
    #print(df_carac2.dtypes)
    df_carac = pd.concat([df_carac, df_carac2], axis=1)
    end = perf_counter()
    print('Elapsed time ', (end - start))
    return df_carac

def read_usagers_files():
    usager_configs = [
        {'prefix': 'usagers_', 'range': range(2006, 2019), 'sep': ','},
        {'prefix': 'usagers_', 'range': range(2019, 2022), 'sep': ';'},
    ]
    return read_all_csvs(usager_configs)

def read_vehicules_files():
    vehicule_configs = [
        {'prefix': 'vehicules_', 'range': range(2006, 2019), 'sep': ','},
        {'prefix': 'vehicules_', 'range': range(2019, 2022), 'sep': ';'},
    ]
    return read_all_csvs(vehicule_configs)

def read_lieux_files():
    lieu_configs = [
        {'prefix': 'lieux_', 'range': range(2006, 2019), 'sep': ','},
        {'prefix': 'lieux_', 'range': range(2019, 2022), 'sep': ';'},
    ]
    return read_all_csvs(lieu_configs)

def calculate_severity_for_accident(df_usagers, df_carac):
    df_usagers = df_usagers.replace({'grav': {1: 0, 2: 13, 3: 7, 4: 3, -1: 0}})

    df_grav = (df_usagers[['Num_Acc', 'grav']]
               .groupby('Num_Acc').agg(['mean', 'sum', 'count'])
               .reset_index())

    # only add severity calculated form usagers data
    # How to aggregate other usagers data ?
    df_grav.columns = ['Num_Acc', 'grav_mean', 'grav_total', 'nb_usagers']
    df_acc = df_carac.merge(df_grav, on='Num_Acc')

    return df_acc

def add_lieux_info_to_accidents(df_lieux, df_acc):
    df_acc = df_acc.merge(df_lieux[['Num_Acc', 'catr', 'circ', 'prof', 'plan', 'infra', 'situ', 'surf', 'vma']], on='Num_Acc')
    return df_acc

def add_vehicule_info_to_accidents(df_vehicules, df_acc):
    # add aggregated vehicules info
    # At the moment, only number of vehicules involved in accident is available.
    # How to aggregate other info ?
    df_veh_count = df_vehicules.groupby('Num_Acc')['num_veh'].count().reset_index()
    df_veh_count.columns = ['Num_Acc', 'nb_vehicules']

    df_acc = df_acc.merge(df_veh_count, on='Num_Acc')
    return df_acc

def plot_accident_count_by_departments(df_carac):
    dep_count = df_carac.dep.value_counts()
    dep_count = dep_count.to_frame().reset_index().rename(columns={'dep': 'acc_count', 'index': 'department'})
    print(dep_count.head())
    fig = plt.figure()
    sns.barplot(x='department', y='acc_count', data=dep_count.head(10), order=dep_count.head(10).department);
  
    return fig

def plot_accident_count_by_atmosphere_conditions(df_carac):
    atm_count = df_carac.atm.value_counts()
    atm_count = atm_count.to_frame().reset_index().rename(columns={'atm': 'acc_count', 'index': 'atm'})
    fig = plt.figure()
    sns.barplot(x='atm', y='acc_count', data=atm_count, order=atm_count.atm);
    return fig  


def plot_severity_by_number_of_pessengers(df_carac):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.lineplot(df_carac, x='nb_usagers', y='grav_total', ax=axes[0])
    sns.lineplot(df_carac, x='nb_usagers', y='grav_mean', ax=axes[1]);
    return fig

def plot_severity_by_atmosphere_condition(df_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(data=df_acc, y='grav_mean', x='atm', ax=axes[0])
    sns.boxplot(data=df_acc, y='grav_total', x='atm', ax=axes[1]);
    return fig

def plot_severity_by_collision_type(df_acc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    sns.boxplot(data=df_acc, y='grav_mean', x='col', ax=axes[0])
    sns.boxplot(data=df_acc, y='grav_total', x='col', ax=axes[1])
    return fig

def plot_accident_count_by_month_and_year(df_acc):
    df_month_count = df_acc.month.value_counts().to_frame().reset_index().rename(columns={'index': 'month', 'month': 'acc_count'})
    df_month_count['year'] = pd.to_datetime(df_month_count['month']).dt.year
    years = df_month_count['year'].unique()
    years.sort()

    fig = plt.figure(figsize=(10, 10))
    #fig, axes = plt.subplots(len(years) // 4 + 1, 4, figsize=(15,5))
    row = len(years) // 4 + 1
    for idx, year in enumerate(years):
        df_year_count = df_month_count[df_month_count.year == year]
        plt.subplot(row, 4, idx + 1)
        b = sns.lineplot(data=df_year_count, x='month', y='acc_count')
        b.set(title=str(year))
        b.tick_params(labelsize=7)
        plt.xticks(rotation=45);

    plt.subplots_adjust(hspace=1, wspace=0.5);

    return fig

def plot_severity_by_week(df_acc):
    fig = plt.figure(figsize=(15,8))
    sns.boxplot(data=df_acc, y='grav_mean', x='nb_semaine')
    return fig

def plot_severity_by_road_surface(df_acc):
    ax = sns.boxplot(data=df_acc, y='grav_mean', x='surf');
    return ax.get_figure()

def plot_severity_histogram(df_acc):
    ax = sns.histplot(df_acc, x='grav_mean', bins=12);
    return ax.get_figure()

def plot_correlation_heatmap(df_acc):
    fig = plt.figure(figsize=(13,13))
    df_acc_reduced = df_acc.drop(columns=['Num_Acc', 'an', 'mois', 'jour', 'grav_total'])
    sns.heatmap(df_acc_reduced.corr(),  annot=True, cmap="RdBu_r", center =0);
    return fig

def save_csv(df, path):
    with open(path, 'w', encoding = 'ISO-8859-1') as f:
        df.to_csv(f)

def load_csv(path):
    #drive.mount('/content/drive')
    with open(path, 'r', encoding='ISO-8859-1') as f:
        return pd.read_csv(f, index_col=0)

def cleans_data(df):
    df_prepared = df.drop(
        columns=['Num_Acc', 
            'an', 'mois', 'jour', 'hrmn',
            'adr', 'gps', 'lat', 'long', 
            'com', 'dep',
            'grav_total', 
            'date', 'month', 'nb_jour', 'nb_semaine', 
            'vma'])

    df_prepared = df_prepared.dropna()
    return df_prepared

def convert_categorical(df_prepared):
    # convert categorical variables which are currently encoded as numeric to string type
    str_cols = set(df_prepared.columns) - set(['grav_mean', 'nb_usagers', 'nb_vehicules'])
    str_cols = {col: 'str' for col in str_cols}
    df_prepared = df_prepared.astype(str_cols)

    # convert categorical variables
    return pd.get_dummies(df_prepared)
    
def prepare_data_model_v1(df):
    '''
    As first attempt, take hypothese that time and 
    geographic location do not have (important) impact on accidents (number and severity) 

    Remove columns not useful for the model.

    Convert categorical variables to numeric using dummification
    '''
    df_prepared = cleans_data(df)
  
    df_prepared = convert_categorical(df_prepared)

    return df_prepared

def train(df, estimator, scaler, target_col='grav_mean'):
    targets = df[target_col]
    feats = df.drop(columns=[target_col])

    train_result = {}
    train_result['estimator'] = estimator
    train_result['scaler'] = scaler

    if (scaler):
        feats = pd.DataFrame(
          scaler.fit_transform(feats), 
          columns=feats.columns, 
          index=feats.index)

    X_train, X_test, y_train, y_test = train_test_split(feats, targets, 
                                                      test_size=0.2, 
                                                      random_state=123)
    start = time.perf_counter()  
    estimator.fit(X_train, y_train)
    duration = time.perf_counter() - start
    train_result['train_duration'] = duration

    train_result['train_score'] = estimator.score(X_train, y_train)
    train_result['test_score'] = estimator.score(X_test, y_test)

    start = time.perf_counter()
    y_test_pred = estimator.predict(X_test)
    duration = time.perf_counter() - start
    train_result['test_duration'] = duration
  
    train_result['y_test_pred'] = y_test_pred
    train_result['y_test'] = y_test
    train_result['errors'] = y_test_pred - y_test
  
    train_result['rmse'] = math.sqrt(mean_squared_error(y_test_pred, y_test))

    return train_result

def plot_results_histogram(train_result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(train_result['y_test_pred'], bins=12, ax=axes[1], shrink=0.8).set(title='Distribution of predicted values')
    sns.histplot(train_result['y_test'], bins=12, ax=axes[0], shrink=0.8).set(title='Distribution of true values')
    return fig
 
def plot_results_scatter_chart(train_result):
    #fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    fig = plt.figure(figsize=(10, 5))
    y_test_pred = train_result['y_test_pred']
    y_test = train_result['y_test']
    indexes = np.random.choice(len(y_test_pred), size=20000)
    indexes.sort()
    plt.subplot(121)
    plt.scatter(np.take(y_test_pred, indexes), np.take(y_test, indexes), label='Predicted / True values')
    plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color='red', label='True value line')
    plt.legend()
    
    plt.subplot(122)
    plt.scatter(np.take(y_test, indexes), np.take(train_result['errors'], indexes), color='#980a10', label='Residues / True values')
    plt.plot((y_test.min(), y_test.max()), (0, 0))
    plt.legend()
    
    #plt.subplot()
    #plt.plot(range(200), y_test[:200], color='yellow')
    #plt.plot(range(200), np.take(y_test_pred, indexes), color='yellow')
    #plt.plot(range(200), np.take(train_result['errors'], indexes), color='red')
    
    return fig

def print_train_result(train_result):
    print('Estimator', train_result['estimator'])
    print('Scaler', train_result['scaler'])
    print('\tScore on train data:', train_result['train_score'])
    print('\tScore on test data:', train_result['test_score'])
    print('\tSRMSE=', train_result['rmse'])
    print('Train duration=', train_result['train_duration'])

def save_train_summary(train_result, file):
    summaries = load_json(file) if os.path.exists(file) else []
    copy = train_result.copy()
    copy['estimator'] = str(copy['estimator'])
    copy['scaler'] = str(copy['scaler'])
    
    copy.pop('y_test_pred')
    copy.pop('y_test')
    copy.pop('errors')
    
    summaries.append(copy)
    save_json(summaries, file)

def load_json(file):
    with open(file) as f:
        return json.load(f)
    
def save_json(value, file):
    with open(file, 'w') as f:
        json.dump(value, f)        

def save_train_result_charts(train_result, model_folder):
    if not (os.path.exists(model_folder)):
        os.makedirs(model_folder)
        
    hist = plot_results_histogram(train_result)
    hist.savefig(model_folder + '/train_result_hist.png')
    
    chart = plot_results_scatter_chart(train_result)
    chart.savefig(model_folder + '/train_result_chart.png')
    
def save_trained_model(train_result, model_folder):
    if not (os.path.exists(model_folder)):
        os.makedirs(model_folder)
        
    joblib.dump(train_result['estimator'], model_folder + '/model.joblib')
    
def load_model(model_folder):
    return joblib.load(model_folder + '/model.joblib')