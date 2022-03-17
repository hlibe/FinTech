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

n_estimators=90
subsample=0.7
max_depth=2
gamma=4
reg_alpha=7
reg_lambda=2
learning_rate=0.1
min_child_weight=1

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
# check and set the working directory
os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')
df = pd.read_csv('data1210rename_use.csv')

col_names = list(df.columns.values[1:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
col_names.remove('default_flag_6m')
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
X_all = df_fillna[col_names]
X = df_fillna[demo_col_names+time_col_names+chnl_col_names+cate_col_names]
y = df_fillna['default_geq_1'] # Target variable

X_base = df_fillna[demo_col_names+time_col_names]
y_base = y

def cal_auc_diff(X_train,y_train,X_test,y_test,X_base_train,y_base_train,X_base_test):
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict_proba(X_train)[:, 1]
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5
    classifier.fit(X_base_train, y_base_train)
    y_base_train_pred = classifier.predict_proba(X_base_train)[:, 1]
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1] #可以加weight 0.5
    
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    test_fpr_base, test_tpr_base, te_thresholds_base = roc_curve(y_test, y_base_test_pred)
    auc_diff = auc(test_fpr, test_tpr) - auc(test_fpr_base, test_tpr_base)
    return auc_diff

ros = RandomOverSampler(random_state=0)

# create dataframes each education level = 1 or 0
X_edu11 = X[X['Education level 1']==1]
for i in [1,2,3,5,6,7]:
    X_edu11 = X_edu11.drop([str('Education level %s')%i], axis=1)
X_base_edu11 = X_base[X_base['Education level 1']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu11 = X_base_edu11.drop([str('Education level %s')%i], axis=1)
y_edu11 = df_fillna[df_fillna['Education level 1']==1]['default_geq_1']
X_edu10 = X[X['Education level 1']==0]
for i in [1,2,3,5,6,7]:
    X_edu10 = X_edu10.drop([str('Education level %s')%i], axis=1)
X_base_edu10 = X_base[X_base['Education level 1']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu10 = X_base_edu10.drop([str('Education level %s')%i], axis=1)
y_edu10 = df_fillna[df_fillna['Education level 1']==0]['default_geq_1']

X_edu21 = X[X['Education level 2']==1]
for i in [1,2,3,5,6,7]:
    X_edu21 = X_edu21.drop([str('Education level %s')%i], axis=1)
X_base_edu21 = X_base[X_base['Education level 2']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu21 = X_base_edu21.drop([str('Education level %s')%i], axis=1)
y_edu21 = df_fillna[df_fillna['Education level 2']==1]['default_geq_1']
X_edu20 = X[X['Education level 2']==0]
for i in [1,2,3,5,6,7]:
    X_edu20 = X_edu20.drop([str('Education level %s')%i], axis=1)
X_base_edu20 = X_base[X_base['Education level 2']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu20 = X_base_edu20.drop([str('Education level %s')%i], axis=1)
y_edu20 = df_fillna[df_fillna['Education level 2']==0]['default_geq_1']

X_edu31 = X[X['Education level 3']==1]
for i in [1,2,3,5,6,7]:
    X_edu31 = X_edu31.drop([str('Education level %s')%i], axis=1)
X_edu30 = X[X['Education level 3']==0]
for i in [1,2,3,5,6,7]:
    X_edu30 = X_edu30.drop([str('Education level %s')%i], axis=1)
X_base_edu31 = X_base[X_base['Education level 3']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu31 = X_base_edu31.drop([str('Education level %s')%i], axis=1)
X_base_edu30 = X_base[X_base['Education level 3']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu30 = X_base_edu30.drop([str('Education level %s')%i], axis=1)
y_edu31 = df_fillna[df_fillna['Education level 3']==1]['default_geq_1']
y_edu30 = df_fillna[df_fillna['Education level 3']==0]['default_geq_1']

X_edu51 = X[X['Education level 5']==1]
for i in [1,2,3,5,6,7]:
    X_edu51 = X_edu51.drop([str('Education level %s')%i], axis=1)
X_edu50 = X[X['Education level 5']==0]
for i in [1,2,3,5,6,7]:
    X_edu50 = X_edu50.drop([str('Education level %s')%i], axis=1)
X_base_edu51 = X_base[X_base['Education level 5']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu51 = X_base_edu51.drop([str('Education level %s')%i], axis=1)
X_base_edu50 = X_base[X_base['Education level 5']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu50 = X_base_edu50.drop([str('Education level %s')%i], axis=1)
y_edu51 = df_fillna[df_fillna['Education level 5']==1]['default_geq_1']
y_edu50 = df_fillna[df_fillna['Education level 5']==0]['default_geq_1']

X_edu61 = X[X['Education level 6']==1]
for i in [1,2,3,5,6,7]:
    X_edu61 = X_edu61.drop([str('Education level %s')%i], axis=1)
X_edu60 = X[X['Education level 6']==0]
for i in [1,2,3,5,6,7]:
    X_edu60 = X_edu60.drop([str('Education level %s')%i], axis=1)
X_base_edu61 = X_base[X_base['Education level 6']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu61 = X_base_edu61.drop([str('Education level %s')%i], axis=1)
X_base_edu60 = X_base[X_base['Education level 6']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu60 = X_base_edu60.drop([str('Education level %s')%i], axis=1)
y_edu61 = df_fillna[df_fillna['Education level 6']==1]['default_geq_1']
y_edu60 = df_fillna[df_fillna['Education level 6']==0]['default_geq_1']

X_edu71 = X[X['Education level 7']==1]
for i in [1,2,3,5,6,7]:
    X_edu71 = X_edu71.drop([str('Education level %s')%i], axis=1)
X_edu70 = X[X['Education level 7']==0]
for i in [1,2,3,5,6,7]:
    X_edu70 = X_edu70.drop([str('Education level %s')%i], axis=1)
X_base_edu71 = X_base[X_base['Education level 7']==1]
for i in [1,2,3,5,6,7]:
    X_base_edu71 = X_base_edu71.drop([str('Education level %s')%i], axis=1)
X_base_edu70 = X_base[X_base['Education level 7']==0]
for i in [1,2,3,5,6,7]:
    X_base_edu70 = X_base_edu70.drop([str('Education level %s')%i], axis=1)
y_edu71 = df_fillna[df_fillna['Education level 7']==1]['default_geq_1']
y_edu70 = df_fillna[df_fillna['Education level 7']==0]['default_geq_1']

for random_state in range(0,14): #跑多个random state取平均值
    locals()['auc_diff_array'+str(random_state)] = []
    for which_edu_level in [1,2,3,5,6,7]: # 对于各个education level计算full高于base的AUC的比例
        locals()['auc_diff'+str(which_edu_level)] = []
        for dummy_edu_level in [1,0]: #该edu level有或者没有
            XX = locals()['X_edu'+str(which_edu_level)+str(dummy_edu_level)]
            yy = locals()['y_edu'+str(which_edu_level)+str(dummy_edu_level)]
            XX_base = locals()['X_base_edu'+str(which_edu_level)+str(dummy_edu_level)]
            XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size = 0.30, random_state=random_state)
            XX_train, yy_train = ros.fit_resample(XX_train, yy_train)
            XX_base_train, XX_base_test, yy_base_train, yy_base_test = train_test_split(XX_base, yy, test_size = 0.30, random_state=random_state)
            XX_base_train, yy_base_train = ros.fit_resample(XX_base_train, yy_base_train)
        auc_diff = cal_auc_diff(XX_train, yy_train, XX_test, yy_test, XX_base_train,yy_base_train, XX_base_test)
        locals()['auc_diff%s'%which_edu_level].append(auc_diff)
    for which_edu_level in [1,2,3,5,6,7]:
        locals()['auc_diff_array%s'%random_state].append(locals()['auc_diff%s'%which_edu_level])

for i in range(0,14):
    locals()['auc_diff_array%s'%i] = sum(locals()['auc_diff_array%s'%i],[])
df_auc_diff_edu = pd.DataFrame({'0':auc_diff_array0,'1':auc_diff_array1,
                            '2':auc_diff_array2,'3':auc_diff_array3,
                            '4':auc_diff_array4,'5':auc_diff_array5,
                            '6':auc_diff_array6,'7':auc_diff_array7,
                            '8':auc_diff_array8,'9':auc_diff_array9,
                            '10':auc_diff_array10,'11':auc_diff_array11,
                            '12':auc_diff_array12,'13':auc_diff_array13})
df_auc_diff_edu = df_auc_diff_edu.transpose()
df_auc_diff_edu.columns = ['Edu1','Edu2','Edu3','Edu4','Edu5','Edu6']
df_auc_diff_edu.to_csv('hete_edu.csv')


###################

# create dataframes housing flag = 1 or 0
X_house1 = X[X['Housing flag']==1]
X_house1 = X_house1.drop(['Housing flag'], axis=1)
X_house0 = X[X['Housing flag']==0]
X_house0 = X_house0.drop(['Housing flag'], axis=1)
X_base_house1 = X_base[X_base['Housing flag']==1]
X_base_house1 = X_base_house1.drop(['Housing flag'], axis=1)
X_base_house0 = X_base[X_base['Housing flag']==0]
X_base_house0 = X_base_house0.drop(['Housing flag'], axis=1)
y_house1 = df_fillna[df_fillna['Housing flag']==1]['default_geq_1']
y_house0 = df_fillna[df_fillna['Housing flag']==0]['default_geq_1']


# create dataframes city super and new first tier= 1 or 0
X_city1 = X_all[(X_all['city_level']==1) | (X_all['city_level']==2) ] #一线城市+新一线城市
X_city1 = X_city1.drop(['city_level'], axis=1)
X_city0 = X_all[(X_all['city_level']==3) | (X_all['city_level']==4)]
X_city0 = X_city0.drop(['city_level'], axis=1)
col_base_city = demo_col_names+time_col_names
col_base_city.insert(0,'city_level')
X_base_city=X_all[col_base_city]
X_base_city1 = X_base_city[(X_base_city['city_level']==1) | (X_base_city['city_level']==2) ] #一线城市+新一线城市
X_base_city1 = X_base_city1.drop(['city_level'], axis=1)
X_base_city0 = X_base_city[(X_base_city['city_level']==3) | (X_base_city['city_level']==4)]
X_base_city0 = X_base_city0.drop(['city_level'], axis=1)
y_city1 = df_fillna[(df_fillna['city_level']==1) | (df_fillna['city_level']==2)]['default_geq_1']
y_city0 = df_fillna[(df_fillna['city_level']==3) | (df_fillna['city_level']==4)]['default_geq_1']


auc_diff_house1_list = []
auc_diff_house0_list = []
auc_diff_city1_list = []
auc_diff_city0_list = []

for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)

    X_train_house1, X_test_house1, y_train_house1, y_test_house1 = train_test_split(X_house1, y_house1, test_size = 0.30, random_state=random_state)
    X_train_house1, y_train_house1 = ros.fit_resample(X_train_house1, y_train_house1)
    X_base_train_house1, X_base_test_house1, y_base_train_house1, y_base_test_house1 = train_test_split(X_base_house1, y_house1, test_size = 0.30, random_state=random_state)
    X_base_train_house1, y_base_train_house1 = ros.fit_resample(X_base_train_house1, y_base_train_house1)
    X_train_house0, X_test_house0, y_train_house0, y_test_house0 = train_test_split(X_house0, y_house0, test_size = 0.30, random_state=random_state)
    X_train_house0, y_train_house0 = ros.fit_resample(X_train_house0, y_train_house0)
    X_base_train_house0, X_base_test_house0, y_base_train_house0, y_base_test_house0 = train_test_split(X_base_house0, y_house0, test_size = 0.30, random_state=random_state)
    X_base_train_house0, y_base_train_house0 = ros.fit_resample(X_base_train_house0, y_base_train_house0)
    auc_diff_house1 = cal_auc_diff(X_train_house1, y_train_house1, 
                       X_test_house1, y_test_house1, 
                       X_base_train_house1, y_base_train_house1, 
                       X_base_test_house1)
    auc_diff_house0 = cal_auc_diff(X_train_house0, y_train_house0, 
                       X_test_house0, y_test_house0, 
                       X_base_train_house0, y_base_train_house0, 
                       X_base_test_house0)    
    X_train_city1, X_test_city1, y_train_city1, y_test_city1 = train_test_split(X_city1, y_city1, test_size = 0.30, random_state=random_state)
    X_train_city1, y_train_city1 = ros.fit_resample(X_train_city1, y_train_city1)
    X_base_train_city1, X_base_test_city1, y_base_train_city1, y_base_test_city1 = train_test_split(X_base_city1, y_city1, test_size = 0.30, random_state=random_state)
    X_base_train_city1, y_base_train_city1 = ros.fit_resample(X_base_train_city1, y_base_train_city1)
    X_train_city0, X_test_city0, y_train_city0, y_test_city0 = train_test_split(X_city0, y_city0, test_size = 0.30, random_state=random_state)
    X_train_city0, y_train_city0 = ros.fit_resample(X_train_city0, y_train_city0)
    X_base_train_city0, X_base_test_city0, y_base_train_city0, y_base_test_city0 = train_test_split(X_base_city0, y_city0, test_size = 0.30, random_state=random_state)
    X_base_train_city0, y_base_train_city0 = ros.fit_resample(X_base_train_city0, y_base_train_city0)
    auc_diff_city1 = cal_auc_diff(X_train_city1, y_train_city1, 
                       X_test_city1, y_test_city1, 
                       X_base_train_city1, y_base_train_city1, 
                       X_base_test_city1)
    auc_diff_city0 = cal_auc_diff(X_train_city0, y_train_city0, 
                       X_test_city0, y_test_city0, 
                       X_base_train_city0, y_base_train_city0, 
                       X_base_test_city0)    
    
    auc_diff_house1_list.append(auc_diff_house1)
    auc_diff_house0_list.append(auc_diff_house0)
    auc_diff_city1_list.append(auc_diff_city1)
    auc_diff_city0_list.append(auc_diff_city0)
    print('round:'+str(random_state))
    
df_auc_diff_house = pd.DataFrame({'House = 1':auc_diff_house1_list, 
                                   'House = 0':auc_diff_house0_list})
df_auc_diff_city = pd.DataFrame({'super or first tier cities':auc_diff_city1_list, 
                                   'lower tier cities':auc_diff_city0_list})
df_auc_diff_house.to_csv('house.csv')
df_auc_diff_city.to_csv('city.csv')







