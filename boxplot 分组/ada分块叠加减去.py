#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 08:10:44 2021

@author: HaoLI
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score  ###计算roc和auc
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import datetime
import time
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# check and set the working directory
os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')
df = pd.read_csv('data1210rename_use.csv')
col_names = list(df.columns.values[3:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
base_col_names = col_names[0:13] # for baseline model 仅仅包含银行数据+早中晚，而不包含消费数据
df_fillna = df.fillna(0) # fill NA with 0. 无消费以0计
X = df_fillna[col_names]
y = df_fillna.default_geq_1 # Target variable
X_base = df_fillna[base_col_names]
y_base = df_fillna.default_geq_1 # Target variable

######## adaboost

n_estimators=50
max_depth=2 
min_samples_split=10 
min_samples_leaf=4
learning_rate=0.1

# check and set the working directory
os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')
df = pd.read_csv('data1210rename_use.csv')
col_names = list(df.columns.values[3:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
base_col_names = col_names[0:13] # for baseline model 仅仅包含银行数据+早中晚，而不包含消费数据
demo_col_names = ['Gender',
                 'Age',
                 'Education level 1',
                 'Education level 2',
                 'Education level 3',
                 'Education level 5',
                 'Education level 6',
                 'Education level 7',
                 'Housing flag',
                 'Salary']
time_col_names = [ 'Midnight amount mean',
                  'Morning amount mean',
                  'Night amount mean']
chnl_col_names = ['Debit amount mean',
                  'Credit amount mean']
cate_col_names = ['Realestate expenditure',
                 'Building material',
                 'Commonwealth merchant count',
                 'Commonwealth',
                 'Cigarrete/wine/tea amount/transaction',
                 'Cigarrete/wine/tea orders/month',
                 'Lottery expenditure',
                 'Short video effective months',
                 'Water/electricity/towngas effective months']
df_fillna = df.fillna(0) # fill NA with 0. 无消费以0计
X = df_fillna[col_names]
y = df_fillna.default_geq_1 # Target variable

X_base = df_fillna[base_col_names]
y_base = df_fillna.default_geq_1 # Target variable

X_demo = df_fillna[demo_col_names]
y_demo = y

X_time = df_fillna[time_col_names]
y_time = y

X_chnl = df_fillna[chnl_col_names]
y_chnl = y

X_cate = df_fillna[cate_col_names]
y_cate = y

auc_full = []
auc_base = []
auc_demo = []
auc_time = []
auc_chnl = []
auc_cate = []

ros = RandomOverSampler(random_state=0)
for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30, random_state=random_state)
    X_base_train, y_base_train = ros.fit_resample(X_base_train, y_base_train)
    X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(X_demo, y_demo, test_size = 0.30, random_state=random_state)
    X_demo_train, y_demo_train = ros.fit_resample(X_demo_train, y_demo_train)
    X_time_train, X_time_test, y_time_train, y_time_test = train_test_split(X_time, y_time, test_size = 0.30, random_state=random_state)
    X_time_train, y_time_train = ros.fit_resample(X_time_train, y_time_train)
    X_chnl_train, X_chnl_test, y_chnl_train, y_chnl_test = train_test_split(X_chnl, y_chnl, test_size = 0.30, random_state=random_state)
    X_chnl_train, y_chnl_train = ros.fit_resample(X_chnl_train, y_chnl_train)
    X_cate_train, X_cate_test, y_cate_train, y_cate_test = train_test_split(X_cate, y_cate, test_size = 0.30, random_state=random_state)
    X_cate_train, y_cate_train = ros.fit_resample(X_cate_train, y_cate_train)

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    # define the model
    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf),
        n_estimators=n_estimators, learning_rate=learning_rate
        )
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5

    classifier.fit(X_base_train, y_base_train)
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1]

    classifier.fit(X_demo_train, y_demo_train)
    y_demo_test_pred = classifier.predict_proba(X_demo_test)[:, 1]

    classifier.fit(X_time_train, y_time_train)
    y_time_test_pred = classifier.predict_proba(X_time_test)[:, 1]

    classifier.fit(X_chnl_train, y_chnl_train)
    y_chnl_test_pred = classifier.predict_proba(X_chnl_test)[:, 1]

    classifier.fit(X_cate_train, y_cate_train)
    y_cate_test_pred = classifier.predict_proba(X_cate_test)[:, 1]

    #### ROC curve and Area-Under-Curve (AUC)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    test_fpr_base, test_tpr_base, te_thresholds_base = roc_curve(y_base_test, y_base_test_pred)
    test_fpr_demo, test_tpr_demo, te_thresholds_demo = roc_curve(y_demo_test, y_demo_test_pred)
    test_fpr_time, test_tpr_time, te_thresholds_time = roc_curve(y_time_test, y_time_test_pred)
    test_fpr_chnl, test_tpr_chnl, te_thresholds_chnl = roc_curve(y_chnl_test, y_chnl_test_pred)
    test_fpr_cate, test_tpr_cate, te_thresholds_cate = roc_curve(y_cate_test, y_cate_test_pred)
    auc_full.append(auc(test_fpr, test_tpr))
    auc_base.append(auc(test_fpr_base, test_tpr_base))
    auc_demo.append(auc(test_fpr_demo, test_tpr_demo))
    auc_time.append(auc(test_fpr_time, test_tpr_time))
    auc_chnl.append(auc(test_fpr_chnl, test_tpr_chnl))
    auc_cate.append(auc(test_fpr_cate, test_tpr_cate))

df_auc = pd.DataFrame({'demographic':auc_demo,
                       'pay time':auc_time,
                       'pay channel':auc_chnl,
                       'category':auc_cate})    
df_auc.to_csv('ada分块.csv') 



#### 叠加 base+1 group ######
n_estimators=200  
max_depth = 8 
min_samples_split = 4
min_samples_leaf = 2
# check and set the working directory
os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')
df = pd.read_csv('data1210rename_use.csv')
col_names = list(df.columns.values[3:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
base_col_names = col_names[0:13] # for baseline model 仅仅包含银行数据+早中晚，而不包含消费数据
demo_col_names = ['Gender',
                 'Age',
                 'Education level 1',
                 'Education level 2',
                 'Education level 3',
                 'Education level 5',
                 'Education level 6',
                 'Education level 7',
                 'Housing flag',
                 'Salary']
time_col_names = [ 'Midnight amount mean',
                  'Morning amount mean',
                  'Night amount mean']
chnl_col_names = ['Debit amount mean',
                  'Credit amount mean']
cate_col_names = ['Realestate expenditure',
                 'Building material',
                 'Commonwealth merchant count',
                 'Commonwealth',
                 'Cigarrete/wine/tea amount/transaction',
                 'Cigarrete/wine/tea orders/month',
                 'Lottery expenditure',
                 'Short video effective months',
                 'Water/electricity/towngas effective months']
df_fillna = df.fillna(0) # fill NA with 0. 无消费以0计
X = df_fillna[col_names]
y = df_fillna.default_geq_1 # Target variable

X_demo = df_fillna[demo_col_names]
y_demo = y

X_base = df_fillna[demo_col_names+time_col_names]
y_base = y

X_demo_chnl = df_fillna[demo_col_names+chnl_col_names]
y_demo_chnl = y

X_demo_cate = df_fillna[demo_col_names+cate_col_names]
y_demo_cate = y

auc_full = []
auc_demo = []
auc_base = []
auc_demo_chnl = []
auc_demo_cate = []


ros = RandomOverSampler(random_state=0)
for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(X_demo, y_demo, test_size = 0.30, random_state=random_state)
    X_demo_train, y_demo_train = ros.fit_resample(X_demo_train, y_demo_train)
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30, random_state=random_state)
    X_base_train, y_base_train = ros.fit_resample(X_base_train, y_base_train)
    X_demo_chnl_train, X_demo_chnl_test, y_demo_chnl_train, y_demo_chnl_test = train_test_split(X_demo_chnl, y_demo_chnl, test_size = 0.30, random_state=random_state)
    X_demo_chnl_train, y_demo_chnl_train = ros.fit_resample(X_demo_chnl_train, y_demo_chnl_train)
    X_demo_cate_train, X_demo_cate_test, y_demo_cate_train, y_demo_cate_test = train_test_split(X_demo_cate, y_demo_cate, test_size = 0.30, random_state=random_state)
    X_demo_cate_train, y_demo_cate_train = ros.fit_resample(X_demo_cate_train, y_demo_cate_train)

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    # define the model
    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf),
        n_estimators=n_estimators, learning_rate=learning_rate
        )
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5

    classifier.fit(X_demo_train, y_demo_train)
    y_demo_test_pred = classifier.predict_proba(X_demo_test)[:, 1]

    classifier.fit(X_base_train, y_base_train)
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1]

    classifier.fit(X_demo_chnl_train, y_demo_chnl_train)
    y_demo_chnl_test_pred = classifier.predict_proba(X_demo_chnl_test)[:, 1]

    classifier.fit(X_demo_cate_train, y_demo_cate_train)
    y_demo_cate_test_pred = classifier.predict_proba(X_demo_cate_test)[:, 1]

    #### ROC curve and Area-Under-Curve (AUC)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    test_fpr_demo, test_tpr_demo, te_thresholds_demo = roc_curve(y_demo_test, y_demo_test_pred)
    test_fpr_base, test_tpr_base, te_thresholds_base = roc_curve(y_base_test, y_base_test_pred)
    test_fpr_demo_chnl, test_tpr_demo_chnl, te_thresholds_demo_chnl = roc_curve(y_demo_chnl_test, y_demo_chnl_test_pred)
    test_fpr_demo_cate, test_tpr_demo_cate, te_thresholds_demo_cate = roc_curve(y_demo_cate_test, y_demo_cate_test_pred)
    auc_full.append(auc(test_fpr, test_tpr))
    auc_demo.append(auc(test_fpr_demo, test_tpr_demo))
    auc_base.append(auc(test_fpr_base, test_tpr_base))
    auc_demo_chnl.append(auc(test_fpr_demo_chnl, test_tpr_demo_chnl))
    auc_demo_cate.append(auc(test_fpr_demo_cate, test_tpr_demo_cate))

df_auc = pd.DataFrame({'demographic':auc_demo,
                       '+ pay time':auc_base,
                       '+ pay channel':auc_demo_chnl,
                       '+ category':auc_demo_cate})    
df_auc.to_csv('ada叠加.csv') 

######full减去某group########
df = pd.read_csv('data1210rename_use.csv')
col_names = list(df.columns.values[3:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
base_col_names = col_names[0:13] # for baseline model 仅仅包含银行数据+早中晚，而不包含消费数据
demo_col_names = ['Gender',
                 'Age',
                 'Education level 1',
                 'Education level 2',
                 'Education level 3',
                 'Education level 5',
                 'Education level 6',
                 'Education level 7',
                 'Housing flag',
                 'Salary']
time_col_names = [ 'Midnight amount mean',
                  'Morning amount mean',
                  'Night amount mean']
chnl_col_names = ['Debit amount mean',
                  'Credit amount mean']
cate_col_names = ['Realestate expenditure',
                 'Building material',
                 'Commonwealth merchant count',
                 'Commonwealth',
                 'Cigarrete/wine/tea amount/transaction',
                 'Cigarrete/wine/tea orders/month',
                 'Lottery expenditure',
                 'Short video effective months',
                 'Water/electricity/towngas effective months']
df_fillna = df.fillna(0) # fill NA with 0. 无消费以0计
X = df_fillna[col_names]
y = df_fillna.default_geq_1 # Target variable

X_ndemo = df_fillna[time_col_names+chnl_col_names+cate_col_names]
y_ndemo = y

X_ntime = df_fillna[demo_col_names+chnl_col_names+cate_col_names]
y_ntime = y

X_nchnl = df_fillna[demo_col_names+time_col_names+cate_col_names]
y_nchnl = y

X_ncate = df_fillna[demo_col_names+time_col_names+chnl_col_names]
y_ncate = y

auc_full = []
auc_ndemo = []
auc_ntime = []
auc_nchnl = []
auc_ncate = []


ros = RandomOverSampler(random_state=0)
for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_ndemo_train, X_ndemo_test, y_ndemo_train, y_ndemo_test = train_test_split(X_ndemo, y_ndemo, test_size = 0.30, random_state=random_state)
    X_ndemo_train, y_ndemo_train = ros.fit_resample(X_ndemo_train, y_ndemo_train)
    X_ntime_train, X_ntime_test, y_ntime_train, y_ntime_test = train_test_split(X_ntime, y_ntime, test_size = 0.30, random_state=random_state)
    X_ntime_train, y_ntime_train = ros.fit_resample(X_ntime_train, y_ntime_train)
    X_nchnl_train, X_nchnl_test, y_nchnl_train, y_nchnl_test = train_test_split(X_nchnl, y_nchnl, test_size = 0.30, random_state=random_state)
    X_nchnl_train, y_nchnl_train = ros.fit_resample(X_nchnl_train, y_nchnl_train)
    X_ncate_train, X_ncate_test, y_ncate_train, y_ncate_test = train_test_split(X_ncate, y_ncate, test_size = 0.30, random_state=random_state)
    X_ncate_train, y_ncate_train = ros.fit_resample(X_ncate_train, y_ncate_train)

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    # define the model
    classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf),
        n_estimators=n_estimators, learning_rate=learning_rate
        )
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5

    classifier.fit(X_ndemo_train, y_ndemo_train)
    y_ndemo_test_pred = classifier.predict_proba(X_ndemo_test)[:, 1]

    classifier.fit(X_ntime_train, y_ntime_train)
    y_ntime_test_pred = classifier.predict_proba(X_ntime_test)[:, 1]

    classifier.fit(X_nchnl_train, y_nchnl_train)
    y_nchnl_test_pred = classifier.predict_proba(X_nchnl_test)[:, 1]

    classifier.fit(X_ncate_train, y_ncate_train)
    y_ncate_test_pred = classifier.predict_proba(X_ncate_test)[:, 1]

    #### ROC curve and Area-Under-Curve (AUC)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    test_fpr_ndemo, test_tpr_ndemo, te_thresholds_ndemo = roc_curve(y_ndemo_test, y_ndemo_test_pred)
    test_fpr_ntime, test_tpr_ntime, te_thresholds_ntime = roc_curve(y_ntime_test, y_ntime_test_pred)
    test_fpr_nchnl, test_tpr_nchnl, te_thresholds_nchnl = roc_curve(y_nchnl_test, y_nchnl_test_pred)
    test_fpr_ncate, test_tpr_ncate, te_thresholds_ncate = roc_curve(y_ncate_test, y_ncate_test_pred)
    auc_full.append(auc(test_fpr, test_tpr))
    auc_ndemo.append(auc(test_fpr_ndemo, test_tpr_ndemo))
    auc_ntime.append(auc(test_fpr_ntime, test_tpr_ntime))
    auc_nchnl.append(auc(test_fpr_nchnl, test_tpr_nchnl))
    auc_ncate.append(auc(test_fpr_ncate, test_tpr_ncate))

df_auc = pd.DataFrame({'full model':auc_full,
                       '- demographic':auc_ndemo,
                       '- pay time':auc_ntime,
                       '- pay channel':auc_nchnl,
                       '- category': auc_ncate})    
df_auc.to_csv('ada减去.csv') 

