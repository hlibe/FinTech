#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:48:41 2021

@author: HaoLI
"""
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
import os #for working directory
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score  # 计算roc和auc
import time
import datetime
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance
from tensorboardX import SummaryWriter

use_gpu = torch.cuda.is_available()
print("GPU",use_gpu)

layer1=48
layer2=8
training_epochs = 150
minibatch_size = 1000
learning_rate=0.001
penalty=2 #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
weight_decay=0.011 #weight_decay 就是 L2 正则项
dropout=0.0

#os.getcwd()
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

layer0=len(X.columns) # input层的神经元个数
layer0_base=len(X_base.columns) # input层的神经元个数

#min_max_scaler = MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
sc = StandardScaler()# transform X into standard normal distribution for each column. X from dataframe to array
X = sc.fit_transform(X) 
X_base = sc.fit_transform(X_base) 

#X = np.array(X)
reduction_rate=[]
auc_full=[]
auc_base=[]
ros = RandomOverSampler(random_state=0)

for random_state in range(0,15):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state) # data types are dataframe
    X_train, y_train = ros.fit_resample(X_train, y_train)
    y_train = y_train.values
    y_test = np.array(y_test)

    X_base_train, X_base_test, y_base_train, y_base_test = train_test_split(X_base, y_base, test_size = 0.30, random_state = random_state)    
    X_base_train, y_base_train = ros.fit_resample(X_base_train, y_base_train)
    y_base_train = y_base_train.values
    y_base_test = np.array(y_base_test)   
    # construct NN
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0, layer1) # fc: fully connected
            #self.bn1 = nn.BatchNorm1d(num_features=64, momentum=0.1) #default momentum = 0.1
            self.fc2 = nn.Linear(layer1, layer2)
            #self.fc3 = nn.Linear(layer2, layer3)
            #self.bn3 = nn.BatchNorm1d(num_features=32)
            #self.fc4 = nn.Linear(28, 24)
            self.fc5 = nn.Linear(layer2, 1)
        # x represents our data
        def forward(self, x): # x is the data
            x = F.relu(self.fc1(x)) # first x pass through 
            #x = self.bn1(x)
            #x = F.dropout(x, p=0.1)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=dropout)
            #x = F.relu(self.fc3(x))
            #x = self.bn3(x)
            #x = F.dropout(x, p=0.25)
            #x = F.relu(self.fc4(x))
            #x = F.softmax(self.fc5(x),dim=0) 
            x = torch.sigmoid(self.fc5(x)) 
            return x
    net = CreditNet().double()  # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #或device = torch.device("cuda:0")
    device1 = torch.device("cuda:1") 
    if torch.cuda.is_available():
        #net = net.cuda()
        net = net.to(device1)   #使用序号为0的GPU
        #或model.to(device1) #使用序号为1的GPU
    
    ########### Train #################
    # X_train, y_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_train = torch.from_numpy(X_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_train = torch.from_numpy(y_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_train = X_train.to(device1)
        y_train = y_train.to(device1)
    train = data_utils.TensorDataset(X_train, y_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_train_labels = [] # create an empty array 
        y_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                #weights = weights.to(device1)
            #print("inputs shape", inputs.shape,  labels.shape)
            #print("inputs", inputs)
            #print("labels", labels)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            #print("y_pred", y_pred)
            y_train_labels = np.append(y_train_labels, labels.cpu().numpy())
            y_train_pred = np.append(y_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3',loss, epoch)
    writer.close()
    #X_test y_test
    X_test = torch.from_numpy(X_test) # check the tested results
    y_test = torch.from_numpy(y_test).double()
    if torch.cuda.is_available():
        X_test = X_test.to(device1)
        y_test = y_test.to(device1)
    test = data_utils.TensorDataset(X_test, y_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_test_labels = []
    y_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            #inputs = inputs.to(device1)
            #labels = labels.to(device1)
            #print("inputs", inputs)
            #print("labels", labels)
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            #print("outputs", outputs)
            #print("predicted", predicted.numpy())
            y_test_labels = np.append(y_test_labels,labels.cpu().numpy())
            y_test_pred = np.append(y_test_pred,outputs.cpu().numpy())
            #print("Y_test_labels", Y_test_labels)
            #print("Y_test_pred", Y_test_pred)

    fullmodelperc = np.percentile(y_test_pred,[95,90,80,70,60,50] )
    full_rej_perc_5 = fullmodelperc[0]
    full_rej_perc_10 = fullmodelperc[1]
    full_rej_perc_20 = fullmodelperc[2]
    full_rej_perc_30 = fullmodelperc[3]
    full_rej_perc_40 = fullmodelperc[4]
    full_rej_perc_50 = fullmodelperc[5]
    print("full model rejection rate[5,10,20,30,40,50]: %s"%fullmodelperc )# get percentile of array y_test_pred


    # construct NN for base, 因为base的输入层神经元个数少于full model
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0_base, layer1) # fc: fully connected
            #self.bn1 = nn.BatchNorm1d(num_features=64, momentum=0.1) #default momentum = 0.1
            self.fc2 = nn.Linear(layer1, layer2)
            #self.fc3 = nn.Linear(layer2, layer3)
            #self.bn3 = nn.BatchNorm1d(num_features=32)
            #self.fc4 = nn.Linear(28, 24)
            self.fc5 = nn.Linear(layer2, 1)
        # x represents our data
        def forward(self, x): # x is the data
            x = F.relu(self.fc1(x)) # first x pass through 
            #x = self.bn1(x)
            #x = F.dropout(x, p=0.1)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=dropout)
            #x = F.relu(self.fc3(x))
            #x = self.bn3(x)
            #x = F.dropout(x, p=0.25)
            #x = F.relu(self.fc4(x))
            #x = F.softmax(self.fc5(x),dim=0) 
            x = torch.sigmoid(self.fc5(x)) 
            return x
    net = CreditNet().double()  # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
 
    
    ### X_base_train, y_base_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_base_train = torch.from_numpy(X_base_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_base_train = torch.from_numpy(y_base_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_base_train = X_base_train.to(device1)
        y_base_train = y_base_train.to(device1)
    train = data_utils.TensorDataset(X_base_train, y_base_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_base_train_labels = [] # create an empty array 
        y_base_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                #weights = weights.to(device1)
            #print("inputs shape", inputs.shape,  labels.shape)
            #print("inputs", inputs)
            #print("labels", labels)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            #print("y_pred", y_pred)
            y_base_train_labels = np.append(y_base_train_labels, labels.cpu().numpy())
            y_base_train_pred = np.append(y_base_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3_base',loss, epoch)
    writer.close()
    ### X_base_test y_base_test
    X_base_test = torch.from_numpy(X_base_test) # check the tested results
    y_base_test = torch.from_numpy(y_base_test).double()
    if torch.cuda.is_available():
        X_base_test = X_base_test.to(device1)
        y_base_test = y_base_test.to(device1)
    test = data_utils.TensorDataset(X_base_test, y_base_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_base_test_labels = []
    y_base_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            #inputs = inputs.to(device1)
            #labels = labels.to(device1)
            #print("inputs", inputs)
            #print("labels", labels)
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            #print("outputs", outputs)
            #print("predicted", predicted.numpy())
            y_base_test_labels = np.append(y_base_test_labels,labels.cpu().numpy())
            y_base_test_pred = np.append(y_base_test_pred,outputs.cpu().numpy())
            #print("Y_test_labels", Y_test_labels)
            #print("Y_test_pred", Y_test_pred)
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
    test_fpr, test_tpr, te_thresholds = roc_curve(y_true = y_test_labels, y_score = y_test_pred)
    test_base_fpr, test_base_tpr, te_base_thresholds = roc_curve(y_true = y_base_test_labels, y_score = y_base_test_pred)
    print("AUC full = ", auc(test_fpr, test_tpr))
    print("AUC base = ", auc(test_base_fpr, test_base_tpr))
    auc_full.append(auc(test_fpr, test_tpr))
    auc_base.append(auc(test_base_fpr, test_base_tpr))
    
df_reduction_rate = pd.DataFrame(reduction_rate)
df_reduction_rate.columns = ['5%','10%','20%','30%','40%','50%']
df_reduction_rate_mean = df_reduction_rate.mean()
df_reduction_rate_mean = pd.DataFrame(df_reduction_rate_mean)
df_reduction_rate_mean = df_reduction_rate_mean.transpose()
df_reduction_rate_mean.to_csv('NN_reduction_rate.csv')                     