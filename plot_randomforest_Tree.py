#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 19:03:45 2021

@author: HaoLI
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from imblearn.over_sampling import RandomOverSampler
from itertools import product
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn import datasets
from IPython.display import Image
import os       

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


n_estimators=200  
max_depth = 8 
min_samples_split = 4
min_samples_leaf = 2

list_rec = [] #记录参数            
list_feaimp = [] #feature importance, 最终会取平均 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
#如果 random_state = None (默认值），会随机选择一个种子，这样每次都会得到不同的数据划分。给 random_state 设置相同的值，那么当别人重新运行你的代码的时候就能得到完全一样的结果，复现和你一样的过程。
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)
######## Create the model
classifier = RandomForestClassifier(n_estimators=n_estimators, max_features = 'sqrt', 
                             max_depth = max_depth, 
                             min_samples_split = min_samples_split,
                             min_samples_leaf = min_samples_leaf,  
                             criterion='entropy', bootstrap=False)
classifier.fit(X_train, y_train)

col_names_array = np.array(col_names)

Estimators = classifier.estimators_
for index, model in enumerate(Estimators):
    # 输出完全树
    filename = '/Users/HaoLI/Stata/credit/out/RF_plot_tree/fulltree_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=col_names,
                         class_names=y.name,
                         #max_depth=3, #设置输出的树的最大深度
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)
    # 输出限制深度的树
    filename = '/Users/HaoLI/Stata/credit/out/RF_plot_tree/tree_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=col_names,
                         class_names=y.name,
                         max_depth=3, #设置输出的树的最大深度
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)
    
