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

# check and set the working directory
os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')
df = pd.read_csv('data1210rename_use.csv')

col_names = list(df.columns.values[3:30]) 
col_names.remove('default_geq_1') #X中不能包含目标函数y
col_names.remove('default_geq_2')
col_names.remove('default_geq_3')
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

X_demo = df_fillna[time_col_names+chnl_col_names+cate_col_names]
y_demo = y

X_deti = df_fillna[demo_col_names+chnl_col_names+cate_col_names]
y_deti = y

X_dech = df_fillna[demo_col_names+time_col_names+cate_col_names]
y_dech = y

X_deca = df_fillna[demo_col_names+time_col_names+chnl_col_names]
y_deca = y

auc_demo = []
auc_deti = []
auc_dech = []
auc_deca = []

ros = RandomOverSampler(random_state=0)
reduction_rate_demo=[]
reduction_rate_deti=[]
reduction_rate_dech=[]
reduction_rate_deca=[]
for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=random_state)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(X_demo, y_demo, test_size = 0.30, random_state=random_state)
    X_demo_train, y_demo_train = ros.fit_resample(X_demo_train, y_demo_train)
    X_deti_train, X_deti_test, y_deti_train, y_deti_test = train_test_split(X_deti, y_deti, test_size = 0.30, random_state=random_state)
    X_deti_train, y_deti_train = ros.fit_resample(X_deti_train, y_deti_train)
    X_dech_train, X_dech_test, y_dech_train, y_dech_test = train_test_split(X_dech, y_dech, test_size = 0.30, random_state=random_state)
    X_dech_train, y_dech_train = ros.fit_resample(X_dech_train, y_dech_train)
    X_deca_train, X_deca_test, y_deca_train, y_deca_test = train_test_split(X_deca, y_deca, test_size = 0.30, random_state=random_state)
    X_deca_train, y_deca_train = ros.fit_resample(X_deca_train, y_deca_train)

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
    y_test_pred = classifier.predict_proba(X_test)[:, 1] #可以加weight 0.5
    fullmodelperc = np.percentile(y_test_pred,[95,90,80,70,60,50] )
    full_rej_perc_5 = fullmodelperc[0]
    full_rej_perc_10 = fullmodelperc[1]
    full_rej_perc_20 = fullmodelperc[2]
    full_rej_perc_30 = fullmodelperc[3]
    full_rej_perc_40 = fullmodelperc[4]
    full_rej_perc_50 = fullmodelperc[5]
    df_full = np.vstack((y_test,y_test_pred)) # 合并两个向量
    df_full = pd.DataFrame(df_full)
    df_full = df_full.transpose()
    df_full.columns = ["label", "pred_prob"]
    def_rate_5_full = df_full[df_full["pred_prob"]<=full_rej_perc_5]['label'].sum()/(df_full.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_full = df_full[df_full["pred_prob"]<=full_rej_perc_10]['label'].sum()/(df_full.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_full = df_full[df_full["pred_prob"]<=full_rej_perc_20]['label'].sum()/(df_full.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_full = df_full[df_full["pred_prob"]<=full_rej_perc_30]['label'].sum()/(df_full.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_full = df_full[df_full["pred_prob"]<=full_rej_perc_40]['label'].sum()/(df_full.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_full = df_full[df_full["pred_prob"]<=full_rej_perc_50]['label'].sum()/(df_full.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的


    #fulul – demographic
    classifier.fit(X_demo_train, y_demo_train)
    y_demo_test_pred = classifier.predict_proba(X_demo_test)[:, 1] #可以加weight 0.5
    basemodelperc = np.percentile(y_demo_test_pred,[95,90,80,70,60,50] )
    base_rej_perc_5 = basemodelperc[0]
    base_rej_perc_10 = basemodelperc[1]
    base_rej_perc_20 = basemodelperc[2]
    base_rej_perc_30 = basemodelperc[3]
    base_rej_perc_40 = basemodelperc[4]
    base_rej_perc_50 = basemodelperc[5]
    print("baseline model rejection rate[5,10,20,30,40,50]: %s"%basemodelperc )# get percentile of array y_test_pred
    #记录base model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_demo = np.vstack((y_test,y_demo_test_pred))
    df_demo = pd.DataFrame(df_demo)
    df_demo = df_demo.transpose()
    df_demo.columns = ["label", "pred_prob"]
    def_rate_5_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_5]['label'].sum()/(df_demo.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_10]['label'].sum()/(df_demo.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_20]['label'].sum()/(df_demo.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_30]['label'].sum()/(df_demo.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_40]['label'].sum()/(df_demo.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_demo = df_demo[df_demo["pred_prob"]<=base_rej_perc_50]['label'].sum()/(df_demo.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的

    #full – pay time
    classifier.fit(X_deti_train, y_deti_train)
    y_deti_test_pred = classifier.predict_proba(X_deti_test)[:, 1] #可以加weight 0.5
    basemodelperc = np.percentile(y_deti_test_pred,[95,90,80,70,60,50] )
    base_rej_perc_5 = basemodelperc[0]
    base_rej_perc_10 = basemodelperc[1]
    base_rej_perc_20 = basemodelperc[2]
    base_rej_perc_30 = basemodelperc[3]
    base_rej_perc_40 = basemodelperc[4]
    base_rej_perc_50 = basemodelperc[5]
    #记录base model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_deti = np.vstack((y_test,y_deti_test_pred))
    df_deti = pd.DataFrame(df_deti)
    df_deti = df_deti.transpose()
    df_deti.columns = ["label", "pred_prob"]
    def_rate_5_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_5]['label'].sum()/(df_deti.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_10]['label'].sum()/(df_deti.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_20]['label'].sum()/(df_deti.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_30]['label'].sum()/(df_deti.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_40]['label'].sum()/(df_deti.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_deti = df_deti[df_deti["pred_prob"]<=base_rej_perc_50]['label'].sum()/(df_deti.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的

    #full – pay channel
    classifier.fit(X_dech_train, y_dech_train)
    y_dech_test_pred = classifier.predict_proba(X_dech_test)[:, 1] #可以加weight 0.5
    basemodelperc = np.percentile(y_dech_test_pred,[95,90,80,70,60,50] )
    base_rej_perc_5 = basemodelperc[0]
    base_rej_perc_10 = basemodelperc[1]
    base_rej_perc_20 = basemodelperc[2]
    base_rej_perc_30 = basemodelperc[3]
    base_rej_perc_40 = basemodelperc[4]
    base_rej_perc_50 = basemodelperc[5]
    #记录base model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_dech = np.vstack((y_test,y_dech_test_pred))
    df_dech = pd.DataFrame(df_dech)
    df_dech = df_dech.transpose()
    df_dech.columns = ["label", "pred_prob"]
    def_rate_5_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_5]['label'].sum()/(df_dech.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_10]['label'].sum()/(df_dech.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_20]['label'].sum()/(df_dech.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_30]['label'].sum()/(df_dech.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_40]['label'].sum()/(df_dech.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_dech = df_dech[df_dech["pred_prob"]<=base_rej_perc_50]['label'].sum()/(df_dech.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的

    #full – categories
    classifier.fit(X_deca_train, y_deca_train)
    y_deca_test_pred = classifier.predict_proba(X_deca_test)[:, 1] #可以加weight 0.5
    basemodelperc = np.percentile(y_deca_test_pred,[95,90,80,70,60,50] )
    base_rej_perc_5 = basemodelperc[0]
    base_rej_perc_10 = basemodelperc[1]
    base_rej_perc_20 = basemodelperc[2]
    base_rej_perc_30 = basemodelperc[3]
    base_rej_perc_40 = basemodelperc[4]
    base_rej_perc_50 = basemodelperc[5]
    #记录base model该循环中的rejection rate为5%，10%，20%，30%，40%，50%时候的违约率
    df_deca = np.vstack((y_test,y_deca_test_pred))
    df_deca = pd.DataFrame(df_deca)
    df_deca = df_deca.transpose()
    df_deca.columns = ["label", "pred_prob"]
    def_rate_5_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_5]['label'].sum()/(df_deca.shape[0]*0.95) #计算rejection rate为5%时候的违约率，test中的
    def_rate_10_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_10]['label'].sum()/(df_deca.shape[0]*0.9) #计算rejection rate为10%时候的违约率，test中的
    def_rate_20_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_20]['label'].sum()/(df_deca.shape[0]*0.8) #计算rejection rate为20%时候的违约率，test中的
    def_rate_30_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_30]['label'].sum()/(df_deca.shape[0]*0.7) #计算rejection rate为30%时候的违约率，test中的
    def_rate_40_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_40]['label'].sum()/(df_deca.shape[0]*0.6) #计算rejection rate为40%时候的违约率，test中的
    def_rate_50_deca = df_deca[df_deca["pred_prob"]<=base_rej_perc_50]['label'].sum()/(df_deca.shape[0]*0.5) #计算rejection rate为50%时候的违约率，test中的


    reduction_rate_5_demo = (def_rate_5_demo-def_rate_5_full)/def_rate_5_full
    reduction_rate_10_demo = (def_rate_10_demo-def_rate_10_full)/def_rate_10_full
    reduction_rate_20_demo = (def_rate_20_demo-def_rate_20_full)/def_rate_20_full
    reduction_rate_30_demo = (def_rate_30_demo-def_rate_30_full)/def_rate_30_full
    reduction_rate_40_demo = (def_rate_40_demo-def_rate_40_full)/def_rate_40_full
    reduction_rate_50_demo = (def_rate_50_demo-def_rate_50_full)/def_rate_50_full

    reduction_rate_demo.append( [reduction_rate_5_demo,reduction_rate_10_demo, 
                                 reduction_rate_20_demo,reduction_rate_30_demo, 
                                 reduction_rate_40_demo,reduction_rate_50_demo])

    reduction_rate_5_deti = (def_rate_5_deti-def_rate_5_full)/def_rate_5_full
    reduction_rate_10_deti = (def_rate_10_deti-def_rate_10_full)/def_rate_10_full
    reduction_rate_20_deti = (def_rate_20_deti-def_rate_20_full)/def_rate_20_full
    reduction_rate_30_deti = (def_rate_30_deti-def_rate_30_full)/def_rate_30_full
    reduction_rate_40_deti = (def_rate_40_deti-def_rate_40_full)/def_rate_40_full
    reduction_rate_50_deti = (def_rate_50_deti-def_rate_50_full)/def_rate_50_full

    reduction_rate_deti.append( [reduction_rate_5_deti,reduction_rate_10_deti, 
                                 reduction_rate_20_deti,reduction_rate_30_deti, 
                                 reduction_rate_40_deti,reduction_rate_50_deti])

    reduction_rate_5_dech = (def_rate_5_dech-def_rate_5_full)/def_rate_5_full
    reduction_rate_10_dech = (def_rate_10_dech-def_rate_10_full)/def_rate_10_full
    reduction_rate_20_dech = (def_rate_20_dech-def_rate_20_full)/def_rate_20_full
    reduction_rate_30_dech = (def_rate_30_dech-def_rate_30_full)/def_rate_30_full
    reduction_rate_40_dech = (def_rate_40_dech-def_rate_40_full)/def_rate_40_full
    reduction_rate_50_dech = (def_rate_50_dech-def_rate_50_full)/def_rate_50_full

    reduction_rate_dech.append( [reduction_rate_5_dech,reduction_rate_10_dech, 
                                 reduction_rate_20_dech,reduction_rate_30_dech, 
                                 reduction_rate_40_dech,reduction_rate_50_dech])   

    reduction_rate_5_deca = (def_rate_5_deca-def_rate_5_full)/def_rate_5_full
    reduction_rate_10_deca = (def_rate_10_deca-def_rate_10_full)/def_rate_10_full
    reduction_rate_20_deca = (def_rate_20_deca-def_rate_20_full)/def_rate_20_full
    reduction_rate_30_deca = (def_rate_30_deca-def_rate_30_full)/def_rate_30_full
    reduction_rate_40_deca = (def_rate_40_deca-def_rate_40_full)/def_rate_40_full
    reduction_rate_50_deca = (def_rate_50_deca-def_rate_50_full)/def_rate_50_full

    reduction_rate_deca.append( [reduction_rate_5_deca,reduction_rate_10_deca, 
                                 reduction_rate_20_deca,reduction_rate_30_deca, 
                                 reduction_rate_40_deca,reduction_rate_50_deca])   
    print("round: "+str(random_state))
    
df_reduction_rate_demo = pd.DataFrame(reduction_rate_demo)
df_reduction_rate_demo.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_demo_mean = df_reduction_rate_demo.mean()
df_reduction_rate_demo_mean = pd.DataFrame(df_reduction_rate_demo_mean)
df_reduction_rate_demo_mean = df_reduction_rate_demo_mean.transpose()

df_reduction_rate_deti = pd.DataFrame(reduction_rate_deti)
df_reduction_rate_deti.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_deti_mean = df_reduction_rate_deti.mean()
df_reduction_rate_deti_mean = pd.DataFrame(df_reduction_rate_deti_mean)
df_reduction_rate_deti_mean = df_reduction_rate_deti_mean.transpose()

df_reduction_rate_dech = pd.DataFrame(reduction_rate_dech)
df_reduction_rate_dech.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_dech_mean = df_reduction_rate_dech.mean()
df_reduction_rate_dech_mean = pd.DataFrame(df_reduction_rate_dech_mean)
df_reduction_rate_dech_mean = df_reduction_rate_dech_mean.transpose()

df_reduction_rate_deca = pd.DataFrame(reduction_rate_deca)
df_reduction_rate_deca.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_deca_mean = df_reduction_rate_deca.mean()
df_reduction_rate_deca_mean = pd.DataFrame(df_reduction_rate_deca_mean)
df_reduction_rate_deca_mean = df_reduction_rate_deca_mean.transpose()

#################
# importing package
import matplotlib.pyplot as plt
from matplotlib import ticker
# create data
x = [5,10,20	,30,40,50]

y_demo = df_reduction_rate_demo_mean.iloc[-1].values.tolist()
y_deti = df_reduction_rate_deti_mean.iloc[-1].values.tolist()
y_dech = df_reduction_rate_dech_mean.iloc[-1].values.tolist()
y_deca = df_reduction_rate_deca_mean.iloc[-1].values.tolist()

fig, ax = plt.subplots()  
# plot lines
plt.plot(x, y_demo, label = "Full – demographic",linestyle="-.",
         marker='D', markerfacecolor='white', markersize=8, 
         color='orange', linewidth=2)
plt.plot(x, y_deti, label = "Full – pay time",linestyle="-",
         marker='v', markerfacecolor='white', markersize=8, 
         color='blue', linewidth=2)
plt.plot(x, y_dech, label = "Full – pay channel",linestyle="-.",
         marker='d', markerfacecolor='white', markersize=8, 
         color='red', linewidth=2)
plt.plot(x, y_deca, label = "Full – category",linestyle="--",
         marker='o', markerfacecolor='white', markersize=8, 
         color='cyan', linewidth=2)
plt.xlabel('Rejection rate (%)')
# Set the y axis label of the current axis.
plt.ylabel('Reduction rate')
# Set a title of the current axes.
plt.title('Full model – single feature group')
#设置坐标轴刻度
my_x_ticks = [5,10,20,30,40,50]
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.legend()
plt.savefig("/Users/HaoLI/Stata/credit/out/XGB_3red_m1.pdf", bbox_inches = 'tight')                        
plt.show()