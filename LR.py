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

penalty='none'

list_rec = [] #记录参数
for random_state in range(0,20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30, random_state=random_state)
    ros = RandomOverSampler(random_state=0)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_base_train, y_base_train = ros.fit_resample(X_base_train, y_base_train)
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

    # use trained model and testing data to predict
    y_train_pred = classifier.decision_function(X_train)
    y_test_pred = classifier.decision_function(X_test)
    fullmodelperc = np.percentile(y_test_pred,[5,10,20,30,40,50] )

    classifier.fit(X_base_train, y_base_train)
    y_base_train_pred = classifier.decision_function(X_base_train)
    y_base_test_pred = classifier.decision_function(X_base_test)#可以加weight 0.5
    basemodelperc = np.percentile(y_base_test_pred,[5,10,20,30,40,50] )
    #print("full model percentile[5,10,20,30,40,50]: %s"%fullmodelperc )# get percentile of array y_test_pred
    #print("baseline model percentile[5,10,20,30,40,50]: %s"%basemodelperc )# get percentile of array y_test_pred


    #### ROC curve and Area-Under-Curve (AUC)
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    print(auc(train_fpr, train_tpr))
    print(auc(test_fpr, test_tpr))
    
    plt.grid()
    plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.title("AUC(LR ROC curve)")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    time1 = datetime.datetime.now()
    #对现在时间格式化，以此作为文件名
    time2 = time1.strftime('%Y-%m-%d-%H%M%S')
    plt.savefig("/Users/HaoLI/Stata/credit/out/ROC figure/Figure_"+time2+".png", bbox_inches = 'tight')                        
    plt.show()
    list_rec.append([auc(train_fpr, train_tpr), auc(test_fpr, test_tpr)])

list_rec_1 = list_rec
df = pd.DataFrame(list_rec, columns = ['IS_AUC','OOS_AUC'])
df.to_csv('LR'+penalty+'_AUC_parameter_record.csv')
