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
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
    y_train_pred = classifier.predict_proba(X_train)[:, 1]
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5

    classifier.fit(X_base_train, y_base_train)
    y_base_train_pred = classifier.predict_proba(X_base_train)[:, 1]
    y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1]

    classifier.fit(X_demo_train, y_demo_train)
    y_demo_train_pred = classifier.predict_proba(X_demo_train)[:, 1]
    y_demo_test_pred = classifier.predict_proba(X_demo_test)[:, 1]

    classifier.fit(X_time_train, y_time_train)
    y_time_train_pred = classifier.predict_proba(X_time_train)[:, 1]
    y_time_test_pred = classifier.predict_proba(X_time_test)[:, 1]

    classifier.fit(X_chnl_train, y_chnl_train)
    y_chnl_train_pred = classifier.predict_proba(X_chnl_train)[:, 1]
    y_chnl_test_pred = classifier.predict_proba(X_chnl_test)[:, 1]

    classifier.fit(X_cate_train, y_cate_train)
    y_cate_train_pred = classifier.predict_proba(X_cate_train)[:, 1]
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

auc_full_mean = format(sum(auc_full)/len(auc_full),'.3f')
auc_base_mean = format(sum(auc_base)/len(auc_base),'.3f')
auc_demo_mean = format(sum(auc_demo)/len(auc_demo),'.3f')
auc_time_mean = format(sum(auc_time)/len(auc_time),'.3f')
auc_chnl_mean = format(sum(auc_chnl)/len(auc_chnl),'.3f')
auc_cate_mean = format(sum(auc_cate)/len(auc_cate),'.3f')

plt.figure(0).clf()
plt.plot(test_fpr, test_tpr,label="Full model (AUC="+str(auc_full_mean)+")",
         linestyle="-",
         markerfacecolor='white', markersize=8, 
         color='grey', linewidth=2)
plt.plot(test_fpr_base, test_tpr_base,label="Base model (AUC="+str(auc_base_mean)+")",
         linestyle=":",
         markerfacecolor='white', markersize=8, 
         color='green', linewidth=2)
plt.plot(test_fpr_demo, test_tpr_demo,label="Demographic (AUC="+str(auc_demo_mean)+")",
         linestyle=":",
         markerfacecolor='white', markersize=8, 
         color='blue', linewidth=2)
plt.plot(test_fpr_time,test_tpr_time,label="Pay time (AUC="+str(auc_time_mean)+")",
         linestyle="--",
         markerfacecolor='white', markersize=8, 
         color='red', linewidth=2)
plt.plot(test_fpr_chnl,test_tpr_chnl,label="Pay channel (AUC="+str(auc_chnl_mean)+")",
         linestyle="-.",
         markerfacecolor='white', markersize=8, 
         color='orange', linewidth=2)
plt.plot([0, 1], [0, 1], color = 'black', linewidth = 1,linestyle="--") # 45度虚线
plt.plot(test_fpr_cate, test_tpr_cate,label="Category (AUC="+str(auc_cate_mean)+")")
plt.xlabel("1-specificity")
plt.ylabel("specificity")    
plt.title('Feature group AUC by AdaBoost')
plt.legend(loc=0)
plt.savefig("/Users/HaoLI/Stata/credit/out/AUC_group_ada.pdf", bbox_inches = 'tight')                        
plt.show()