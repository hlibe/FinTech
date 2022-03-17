#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:48:50 2021

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
base_col_names = col_names[0:13] # for baseline model 包含银行数据+早中晚数据
df_fillna = df.fillna(0) # fill NA with 0. 无消费以0计
X = df_fillna[col_names]
y = df_fillna.default_geq_1 # Target variable
X_base = df_fillna[base_col_names]
y_base = df_fillna.default_geq_1 # Target variable


n_estimators=100
subsample=0.7
max_depth=6
gamma=6
reg_alpha=7
reg_lambda=1
learning_rate=0.1
min_child_weight=3

list_rec = [] #记录参数
list_feaimp = [] #feature importance, 最终会取平均 
#n_estimators(20,310,10), subsample [0.5,0.7,0.8,1], max_depth (3,11), 
#gamma(0,8,2), reg_alpha(1,9,2), reg_lambda(1,4)


for n_estimators in range(90,101,100):
    for max_depth in [4]:
        for gamma in [4,6,8,10,20]:
            for reg_alpha in [2,5,7,10]:
                for reg_lambda in [5,7,10]: 
                    for min_child_weight in [1,3,5,7,9]:
                        for random_state in range(0,20):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = random_state)
                            #如果 random_state = None (默认值），会随机选择一个种子，这样每次都会得到不同的数据划分。给 random_state 设置相同的值，那么当别人重新运行你的代码的时候就能得到完全一样的结果，复现和你一样的过程。
                            X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30)
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
                                reg_lambda=reg_alpha,
                                n_jobs=2,  # parallel threads
                                #random_state=999,
                                min_child_weight=min_child_weight
                                )
                            classifier.fit(X_train, y_train)
                            list_feaimp.append(classifier.feature_importances_)
                            #print(classifier.feature_importances_) 
                            y_train_pred = classifier.predict_proba(X_train)[:, 1]
                            y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5
                            fullmodelperc = np.percentile(y_test_pred,[5,10,20,30,40,50] )
                            
                            classifier.fit(X_base_train, y_base_train)
                            y_base_train_pred = classifier.predict_proba(X_base_train)[:, 1]
                            y_base_test_pred = classifier.predict_proba(X_base_test)[:, 1] #可以加weight 0.5
                            basemodelperc = np.percentile(y_base_test_pred,[5,10,20,30,40,50] )
                            #print("full model percentile[5,10,20,30,40,50]: %s"%fullmodelperc )# get percentile of array y_test_pred
                            #print("baseline model percentile[5,10,20,30,40,50]: %s"%basemodelperc )# get percentile of array y_test_pred
                        
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
                            t = ''' 
                            n_estimators = %s, subsample = %s, max_depth = %s, 
                            gamma = %s, reg_alpha = %s, reg_lambda = %s,
                            learning_rate = %s, random_state = %s,
                            min_child_weight = %s
                            '''%(n_estimators,subsample,max_depth,gamma,reg_alpha,
                            reg_lambda,learning_rate, random_state, min_child_weight )
                            plt.title("AUC(XGBoost ROC curve)"+t)
                            plt.grid(color='black', linestyle='-', linewidth=0.5)
                            time1 = datetime.datetime.now()
                            #对现在时间格式化，以此作为文件名
                            time2 = time1.strftime('%Y-%m-%d-%H%M%S')
                            plt.savefig("/Users/HaoLI/Stata/credit/out/ROC figure/Figure_"+time2+".png", bbox_inches = 'tight')                        
                            plt.show()
                            list_rec.append([auc(train_fpr, train_tpr), auc(test_fpr, test_tpr),
                                             n_estimators,subsample,
                                             max_depth,gamma,reg_alpha,
                                             reg_lambda,learning_rate,random_state,
                                             min_child_weight])
                    
list_rec_1 = list_rec
df = pd.DataFrame(list_rec, columns = ['IS_AUC','OOS_AUC','n_estimators', 
                                       'subsample', 'max_depth','gamma', 
                                       'reg_alpha', 'reg_lambda','learning_rate', 
                                       'random_state', 'min_child_weight'])
df.to_csv('XGB_AUC_maxdepth4.csv')
