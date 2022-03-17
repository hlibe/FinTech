#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:32:19 2020

@author: HaoLI
"""
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import datasets, metrics, model_selection, svm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os #for working directory
import datetime
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score  ###计算roc和auc
from sklearn.preprocessing import StandardScaler
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


n_estimators=200  
max_depth = 8 
min_samples_split = 4
min_samples_leaf = 2

list_rec = [] #记录参数            
list_feaimp = [] #feature importance, 最终会取平均 
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

    ######## Create the model
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_features = 'sqrt', 
                                 max_depth = max_depth, 
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,  
                                 criterion='entropy', bootstrap=False)

    classifier.fit(X_train, y_train)
    list_feaimp.append(classifier.feature_importances_)
    print(classifier.feature_importances_) 

    # Predicting the test set results and calculating the accuracy
    ############### Receiver Operating Characteristic(ROC) auc ###############
    y_train_pred = classifier.predict_proba(X_train)[:, 1]
    y_test_pred = classifier.predict_proba(X_test)[:, 1]
    #首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    
    plt.grid()
    plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0,1],[0,1],'g--')
    plt.legend()
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    t = ''' 
    n_estimators = %s, max_depth = %s, 
    min_samples_split = %s, min_samples_leaf = %s,
    random_state = %s
        '''%(n_estimators,max_depth,min_samples_split,min_samples_leaf, random_state)
    plt.title("AUC(Random Forest ROC curve)"+t)
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    time0 = datetime.datetime.now()
    #对现在时间格式化，以此作为文件名
    time1 = time0.strftime('%Y-%m-%d-%H%M%S')
    plt.savefig("/Users/HaoLI/Stata/credit/out/ROC figure/Figure_"+time1+".png", bbox_inches = 'tight')
    plt.show()
    list_rec.append([auc(train_fpr, train_tpr), auc(test_fpr, test_tpr),
                     n_estimators,max_depth,min_samples_split,min_samples_leaf,
                     random_state])

list_rec_1 = list_rec
df = pd.DataFrame(list_rec, columns = ['IS_AUC','OOS_AUC', 'n_estimators', 
                                       'max_depth','min_samples_split', 
                                       'min_samples_leaf', 'random_state'])
df.to_csv('RF_AUC.csv')

list_feaimp_1 = list_feaimp
df = pd.DataFrame(list_feaimp, columns = col_names)
df.to_csv('RF_feature_importance.csv')





#############################################
#############################################
'''
# Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.
# you got a classification rate of 80%, considered as good accuracy.
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))   # 0.9804153492709778
# Precision: Precision is about being precise, i.e., how accurate your model is. 
print("Precision:",metrics.precision_score(y_test, y_pred))
# Recall: If there are patients who have diabetes in the test set and your Logistic Regression model can identify it 58% of the time.
print("Recall:",metrics.recall_score(y_test, y_pred))


### Create the Confusion Matrix 
#cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

ax = plt.subplots()
sns.heatmap(cnf_matrix, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);
'''

