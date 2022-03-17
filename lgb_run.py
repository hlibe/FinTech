#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:56:38 2021

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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import time
import datetime
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


#numerical_columns=['id', 'loanAmnt', 'term', 'interestRate', 'installment', 'employmentTitle', 'homeOwnership']

#Specifying the parameter
n_estimators=100
learning_rate=0.1
max_depth=6
num_leaves=16
feature_fraction=1
bagging_fraction=1
verbosity=20
num_boost_round=20000
verbose_eval=1000
early_stopping_rounds=200
reg_alpha=2
reg_lambda=15 

list_rec = [] #记录参数
list_feature_importance = []
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
    
    #converting the dataset into proper LGB format 
    train_matrix=lgb.Dataset(X_train, label=y_train)
    valid_matrix= lgb.Dataset(X_test, label=y_test)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        "n_estimators":n_estimators,
        'learning_rate': learning_rate,#较小的学习率，较大的决策树个数
        'max_depth': max_depth,#树的最大深度，防止过拟合
        'num_leaves': num_leaves,
        'feature_fraction': feature_fraction, #每次选择所有的特征训练树
        'bagging_fraction': bagging_fraction,
    }
    classifier=lgb.train(params, train_set=train_matrix, valid_sets=valid_matrix, num_boost_round=num_boost_round, verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds)
 
    ax=lgb.plot_importance(classifier, figsize=(15,15))
    plt.show()  
    importance = classifier.feature_importance(importance_type='split')
    feature_name = col_names
    importance = importance/sum(importance)
    list_feature_importance.append(importance)

    # use trained model and testing data to predict
    y_train_pred = classifier.predict(X_train)
    y_test_pred=classifier.predict(X_test)
    #### ROC curve and Area-Under-Curve (AUC)
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    print(auc(train_fpr, train_tpr))
    print(auc(test_fpr, test_tpr))
    
    plt.figure(figsize=(12,6))
    lgb.plot_importance(classifier)
    plt.title("Feature Importances")
    plt.show()
    
    plt.grid()
    plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    t = ''' 
    n_estimators = %s, learning_rate = %s, max_depth = %s
    num_leaves = %s, feature_fraction = %s, bagging_fraction = %s
    verbosity = %s, num_boost_round = %s
    verbose_eval = %s, early_stopping_rounds = %s, random_state = %s
    '''%(n_estimators,learning_rate,max_depth, num_leaves, feature_fraction, 
    bagging_fraction, verbosity, num_boost_round, 
    verbose_eval, early_stopping_rounds, random_state)    
    plt.title("AUC(LightGBM ROC curve)"+t)
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    time1 = datetime.datetime.now()
    #对现在时间格式化，以此作为文件名
    time2 = time1.strftime('%Y-%m-%d-%H%M%S')
    plt.savefig("/Users/HaoLI/Stata/credit/out/ROC figure/Figure_"+time2+".png", bbox_inches = 'tight')                        
    plt.show()
    list_rec.append([auc(train_fpr, train_tpr), auc(test_fpr, test_tpr),
                    n_estimators,
                    learning_rate,
                    max_depth,
                    num_leaves,
                    feature_fraction,
                    bagging_fraction,
                    verbosity,
                    num_boost_round,
                    verbose_eval,
                    early_stopping_rounds,
                    random_state
                     ])

list_rec_1 = list_rec
df = pd.DataFrame(list_rec, columns = ['IS_AUC','OOS_AUC', 'n_estimators',
                                        'learning_rate',
                                        'max_depth',
                                        'num_leaves',
                                        'feature_fraction',
                                        'bagging_fraction',
                                        'verbosity',
                                        'num_boost_round',
                                        'verbose_eval',
                                        'early_stopping_rounds',
                                        'random_state'])
df.to_csv('lightGBM_AUC.csv')

list_feature_importance_1 = list_feature_importance
feature_importance = pd.DataFrame(list_feature_importance, columns = col_names )
feature_importance.to_csv('lighGBM_feature_importance.csv',index=False)
