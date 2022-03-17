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

X_demo = df_fillna[demo_col_names]
y_demo = y

X_base = df_fillna[base_col_names]
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
    y_train_pred = classifier.predict_proba(X_train)[:, 1]
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5

    classifier.fit(X_demo_train, y_demo_train)
    y_demo_train_pred = classifier.predict_proba(X_demo_train)[:, 1]
    y_demo_test_pred = classifier.predict_proba(X_demo_test)[:, 1]

    classifier.fit(X_base_train, y_base_train)
    y_base_train_pred = classifier.predict_proba(X_base_train)[:, 1]
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1]

    classifier.fit(X_demo_chnl_train, y_demo_chnl_train)
    y_demo_chnl_train_pred = classifier.predict_proba(X_demo_chnl_train)[:, 1]
    y_demo_chnl_test_pred = classifier.predict_proba(X_demo_chnl_test)[:, 1]

    classifier.fit(X_demo_cate_train, y_demo_cate_train)
    y_demo_cate_train_pred = classifier.predict_proba(X_demo_cate_train)[:, 1]
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
df_auc.to_csv('LR0叠加.csv') 