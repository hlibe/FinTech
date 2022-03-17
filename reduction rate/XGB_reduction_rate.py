#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:45:27 2021

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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from xgboost import Booster
from xgboost import DMatrix
import datetime
import time
from sklearn.preprocessing import StandardScaler
# plot feature importance manually
from numpy import loadtxt
from matplotlib import pyplot
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


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


n_estimators=90
subsample=0.7
max_depth=2
gamma=4
reg_alpha=7
reg_lambda=2
learning_rate=0.1
min_child_weight=1

reduction_rate=[]
for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = random_state)
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30, random_state = random_state)
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_base_train, y_base_train = ros.fit_resample(X_base_train, y_base_train)
    #min_max_scaler = MinMaxScaler()
    #X_train = min_max_scaler.fit_transform(X_train)
    #X_test = min_max_scaler.fit_transform(X_test)
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.fit_transform(X_test)    

    classifier = XGBClassifier(objective='binary:logistic',
        eval_metric='auc',
        n_estimators=n_estimators,
        subsample=subsample,
        learning_rate=learning_rate,
        max_depth=max_depth,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=2,  # parallel threads
        random_state=999,
        )
    classifier.fit(X_train, y_train)
    #list_feaimp.append(classifier.feature_importances_)
    #print(classifier.feature_importances_) 
    y_train_pred = classifier.predict_proba(X_train)[:, 1]
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5
    fullmodelperc = np.percentile(y_test_pred,[95,90,80,70,60,50] )
    full_rej_perc_5 = fullmodelperc[0]
    full_rej_perc_10 = fullmodelperc[1]
    full_rej_perc_20 = fullmodelperc[2]
    full_rej_perc_30 = fullmodelperc[3]
    full_rej_perc_40 = fullmodelperc[4]
    full_rej_perc_50 = fullmodelperc[5]
    
    classifier.fit(X_base_train, y_base_train)
    y_base_train_pred = classifier.predict_proba(X_base_train)[:, 1]
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1] #可以加weight 0.5
    basemodelperc = np.percentile(y_base_test_pred,[95,90,80,70,60,50] )
    base_rej_perc_5 = basemodelperc[0]
    base_rej_perc_10 = basemodelperc[1]
    base_rej_perc_20 = basemodelperc[2]
    base_rej_perc_30 = basemodelperc[3]
    base_rej_perc_40 = basemodelperc[4]
    base_rej_perc_50 = basemodelperc[5]
    print("full model rejection rate[5,10,20,30,40,50]: %s"%fullmodelperc )# get percentile of array y_test_pred
    print("baseline model rejection rate[5,10,20,30,40,50]: %s"%basemodelperc )# get percentile of array y_test_pred
    
    #记录base model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_base = np.vstack((y_test,y_base_test_pred))
    df_base = pd.DataFrame(df_base)
    df_base = df_base.transpose()
    df_base.columns = ["label", "pred_prob"]
    def_rate_5_base = df_base[df_base["pred_prob"]<=base_rej_perc_5]['label'].sum()/(df_base.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_base = df_base[df_base["pred_prob"]<=base_rej_perc_10]['label'].sum()/(df_base.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_base = df_base[df_base["pred_prob"]<=base_rej_perc_20]['label'].sum()/(df_base.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_base = df_base[df_base["pred_prob"]<=base_rej_perc_30]['label'].sum()/(df_base.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_base = df_base[df_base["pred_prob"]<=base_rej_perc_40]['label'].sum()/(df_base.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_base = df_base[df_base["pred_prob"]<=base_rej_perc_50]['label'].sum()/(df_base.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的

    #记录full model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_full = np.vstack((y_test,y_test_pred))
    df_full = pd.DataFrame(df_full)
    df_full = df_full.transpose()
    df_full.columns = ["label", "pred_prob"]
    def_rate_5_full = df_full[df_full["pred_prob"]<=full_rej_perc_5]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_full = df_full[df_full["pred_prob"]<=full_rej_perc_10]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_20_full = df_full[df_full["pred_prob"]<=full_rej_perc_20]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_30_full = df_full[df_full["pred_prob"]<=full_rej_perc_30]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_40_full = df_full[df_full["pred_prob"]<=full_rej_perc_40]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_50_full = df_full[df_full["pred_prob"]<=full_rej_perc_50]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的

    reduction_rate_5 = -(def_rate_5_full-def_rate_5_base)/def_rate_5_base
    reduction_rate_10 = -(def_rate_10_full-def_rate_10_base)/def_rate_10_base
    reduction_rate_20 = -(def_rate_20_full-def_rate_20_base)/def_rate_20_base
    reduction_rate_30 = -(def_rate_30_full-def_rate_30_base)/def_rate_30_base
    reduction_rate_40 = -(def_rate_40_full-def_rate_40_base)/def_rate_40_base
    reduction_rate_50 = -(def_rate_50_full-def_rate_50_base)/def_rate_50_base

    reduction_rate.append( [reduction_rate_5, reduction_rate_10, reduction_rate_20,
                          reduction_rate_30, reduction_rate_40, reduction_rate_50])
   
df_reduction_rate = pd.DataFrame(reduction_rate)
df_reduction_rate.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_mean = df_reduction_rate.mean()
df_reduction_rate_mean = pd.DataFrame(df_reduction_rate_mean)
df_reduction_rate_mean = df_reduction_rate_mean.transpose()
df_reduction_rate_mean.to_csv('XGB_reduction_rate.csv')
