#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:52:56 2021

@author: HaoLI
"""
import pandas as pd
import numpy as np
import os #for working directory
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc, roc_auc_score  ###计算roc和auc
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix

################################################

# check and set the working directory
os.getcwd()
os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')

df = pd.read_csv("data_bylasso0307_0327.csv")
col_names = list(df.columns.values) #get the column names of the dataframe
cons_cols = col_names[20:57] #the feature columns are the independent variables
demo_cols = ['gender','age','edu_level_1','edu_level_2', 'edu_level_3', 'edu_level_5','edu_level_6','edu_level_7','housing_flag','log_salary_level']
y = df.default_geq_2 # Target variable
X = df[demo_cols + cons_cols ]
#X.isnull().sum() #check the number of nan
X = X.fillna(0) # fill nan with 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

####################  Polynomial Kernel 
svcpoly1 = SVC(kernel='linear') #  kernel type polynomial
svcpoly1.fit(X_train, y_train)

###通过decision_function()计算得到的test_predict_label的值，用在roc_curve()函数中
y_train_pred = svcpoly1.decision_function(X_train)
y_test_pred = svcpoly1.decision_function(X_test)
#首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
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
plt.title("AUC(Linear Kernel ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
#plt.savefig('SVM_poly5.png')

##########  Gaussian Kernel Radial Basis Function 

svcRBF = SVC(kernel='rbf', C=0.8)
svcRBF.fit(X_train, y_train)

y_train_pred = svcRBF.decision_function(X_train)
y_test_pred = svcRBF.decision_function(X_test)
#首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
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
plt.title("AUC(Radial Basis Function Kernel ROC curve) gamma=default, C=0.8")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()
plt.savefig('SVM_RBF gamma default C10.png')




########### Sigmoid kernel
from sklearn.svm import SVC
svcsigmoid = SVC(kernel='sigmoid')
svcsigmoid.fit(X_train, y_train)
y_pred = svcsigmoid.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

###通过decision_function()计算得到的y_test_pred的值，用在roc_curve()函数中
y_train_pred = svcsigmoid.decision_function(X_train)
y_test_pred = svcsigmoid.decision_function(X_test)
#首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
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
plt.title("AUC(Sigmoid Kernel ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()

#################### 1. Polynomial Kernel 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)
X = irisdata.drop('Class', axis=1) # drop column 'Class'
y = irisdata['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

########## 2. Gaussian Kernel Radial Basis Function 
from sklearn.svm import SVC
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_train, y_train)

# Prediction and Evaluation
y_pred = svclassifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

########## 3. Sigmoid Kernel
from sklearn.svm import SVC
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)

#Prediction and Evaluation
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

############### example SVM + ROC AUC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import model_selection

# Import some data to play with
iris = datasets.load_iris()
X = iris.data#得到样本集
y = iris.target#得到标签集
 
##变为2分类
X, y = X[y != 2], y[y != 2]#通过取y不等于2来取两种类别
 
# Add noisy features to make the problem harder添加扰动
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
 
# shuffle and split training and test sets划分样本集
train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y, test_size=.3,random_state=0)
#train_data用于训练的样本集, test_data用于测试的样本集, train_label训练样本对应的标签集, test_label测试样本对应的标签集
 
# Learn to predict each class against the other分类器设置
svm = svm.SVC(kernel='linear', probability=True)#使用核函数为线性核，参数默认，创建分类器
 
###通过decision_function()计算得到的test_predict_label的值，用在roc_curve()函数中
test_predict_label = svm.fit(train_data, train_label).decision_function(test_data)
#首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集
print(test_predict_label)
 
# Compute ROC curve and ROC area for each class#计算tp,fp
#通过测试样本输入的标签集和模型预测的标签集进行比对，得到fp,tp,不同的fp,tp是算法通过一定的规则改变阈值获得的
fpr,tpr,threshold = roc_curve(test_label, test_predict_label) ###计算真正率和假正率
print(fpr)
print(tpr)
print(threshold)
roc_auc = auc(fpr,tpr) ###计算auc的值，auc就是曲线包围的面积，越大越好
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
