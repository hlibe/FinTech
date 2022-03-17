#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 15:19:34 2021

@author: HaoLI
"""
# evaluate gradient boosting algorithm for classification
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

penalty='none'
# 'none' for normal LR, 'l1' for L1 regularization

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

    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.fit_transform(X_test)
    # define the model
    classifier = LogisticRegression(penalty= 'none', dual=False,
                                    tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,   
                                    class_weight=None, random_state=None, solver='saga',  
                                    max_iter=100, verbose=0,   
                                    warm_start=False, n_jobs=None, l1_ratio=None) #'none' for no penalty
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
df_auc.to_csv('LR0减去.csv') 