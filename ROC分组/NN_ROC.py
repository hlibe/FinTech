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

layer0=len(X.columns) # input层的神经元个数
layer0_base=len(X_base.columns) # input层的神经元个数
layer0_demo=len(X_demo.columns) # input层的神经元个数
layer0_time=len(X_time.columns) # input层的神经元个数
layer0_chnl=len(X_chnl.columns) # input层的神经元个数
layer0_cate=len(X_cate.columns) # input层的神经元个数

sc = StandardScaler()# transform X into standard normal distribution for each column. X from dataframe to array
X = sc.fit_transform(X) 
X_base = sc.fit_transform(X_base) 
X_demo = sc.fit_transform(X_demo) 
X_time = sc.fit_transform(X_time) 
X_chnl = sc.fit_transform(X_chnl) 
X_cate = sc.fit_transform(X_cate) 

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

    X_demo_train, X_demo_test, y_demo_train, y_demo_test = train_test_split(X_demo, y_demo, test_size = 0.30, random_state = random_state)    
    X_demo_train, y_demo_train = ros.fit_resample(X_demo_train, y_demo_train)
    y_demo_train = y_demo_train.values
    y_demo_test = np.array(y_demo_test)   

    X_time_train, X_time_test, y_time_train, y_time_test = train_test_split(X_time, y_time, test_size = 0.30, random_state = random_state)    
    X_time_train, y_time_train = ros.fit_resample(X_time_train, y_time_train)
    y_time_train = y_time_train.values
    y_time_test = np.array(y_time_test)   

    X_chnl_train, X_chnl_test, y_chnl_train, y_chnl_test = train_test_split(X_chnl, y_chnl, test_size = 0.30, random_state = random_state)    
    X_chnl_train, y_chnl_train = ros.fit_resample(X_chnl_train, y_chnl_train)
    y_chnl_train = y_chnl_train.values
    y_chnl_test = np.array(y_chnl_test)   

    X_cate_train, X_cate_test, y_cate_train, y_cate_test = train_test_split(X_cate, y_cate, test_size = 0.30, random_state = random_state)    
    X_cate_train, y_cate_train = ros.fit_resample(X_cate_train, y_cate_train)
    y_cate_train = y_cate_train.values
    y_cate_test = np.array(y_cate_test)   

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
        net = net.to(device1)   #使用序号为0的GPU
    ########### Full model #################
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
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_test_labels = np.append(y_test_labels,labels.cpu().numpy())
            y_test_pred = np.append(y_test_pred,outputs.cpu().numpy())

    ############## base  ###########
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
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
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
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_base_test_labels = np.append(y_base_test_labels,labels.cpu().numpy())
            y_base_test_pred = np.append(y_base_test_pred,outputs.cpu().numpy())

    ########### demo  ##############
    # construct NN for demo, 因为demo的输入层神经元个数少于full model
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0_demo, layer1) # fc: fully connected
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
    ### X_demo_train, y_demo_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_demo_train = torch.from_numpy(X_demo_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_demo_train = torch.from_numpy(y_demo_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_demo_train = X_demo_train.to(device1)
        y_demo_train = y_demo_train.to(device1)
    train = data_utils.TensorDataset(X_demo_train, y_demo_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_demo_train_labels = [] # create an empty array 
        y_demo_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            y_demo_train_labels = np.append(y_demo_train_labels, labels.cpu().numpy())
            y_demo_train_pred = np.append(y_demo_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3_demo',loss, epoch)
    writer.close()
    ### X_demo_test y_demo_test
    X_demo_test = torch.from_numpy(X_demo_test) # check the tested results
    y_demo_test = torch.from_numpy(y_demo_test).double()
    if torch.cuda.is_available():
        X_demo_test = X_demo_test.to(device1)
        y_demo_test = y_demo_test.to(device1)
    test = data_utils.TensorDataset(X_demo_test, y_demo_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_demo_test_labels = []
    y_demo_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_demo_test_labels = np.append(y_demo_test_labels,labels.cpu().numpy())
            y_demo_test_pred = np.append(y_demo_test_pred,outputs.cpu().numpy())

    ############### Pay time ###########
    # construct NN for time, 因为time的输入层神经元个数少于full model
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0_time, layer1) # fc: fully connected
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
    ### X_time_train, y_time_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_time_train = torch.from_numpy(X_time_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_time_train = torch.from_numpy(y_time_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_time_train = X_time_train.to(device1)
        y_time_train = y_time_train.to(device1)
    train = data_utils.TensorDataset(X_time_train, y_time_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_time_train_labels = [] # create an empty array 
        y_time_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            y_time_train_labels = np.append(y_time_train_labels, labels.cpu().numpy())
            y_time_train_pred = np.append(y_time_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3_time',loss, epoch)
    writer.close()
    ### X_time_test y_time_test
    X_time_test = torch.from_numpy(X_time_test) # check the tested results
    y_time_test = torch.from_numpy(y_time_test).double()
    if torch.cuda.is_available():
        X_time_test = X_time_test.to(device1)
        y_time_test = y_time_test.to(device1)
    test = data_utils.TensorDataset(X_time_test, y_time_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_time_test_labels = []
    y_time_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_time_test_labels = np.append(y_time_test_labels,labels.cpu().numpy())
            y_time_test_pred = np.append(y_time_test_pred,outputs.cpu().numpy())

    ############### Pay channel ###########
    # construct NN for chnl, 因为chnl的输入层神经元个数少于full model
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0_chnl, layer1) # fc: fully connected
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
    ### X_chnl_train, y_chnl_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_chnl_train = torch.from_numpy(X_chnl_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_chnl_train = torch.from_numpy(y_chnl_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_chnl_train = X_chnl_train.to(device1)
        y_chnl_train = y_chnl_train.to(device1)
    train = data_utils.TensorDataset(X_chnl_train, y_chnl_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_chnl_train_labels = [] # create an empty array 
        y_chnl_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            y_chnl_train_labels = np.append(y_chnl_train_labels, labels.cpu().numpy())
            y_chnl_train_pred = np.append(y_chnl_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3_chnl',loss, epoch)
    writer.close()
    ### X_chnl_test y_chnl_test
    X_chnl_test = torch.from_numpy(X_chnl_test) # check the tested results
    y_chnl_test = torch.from_numpy(y_chnl_test).double()
    if torch.cuda.is_available():
        X_chnl_test = X_chnl_test.to(device1)
        y_chnl_test = y_chnl_test.to(device1)
    test = data_utils.TensorDataset(X_chnl_test, y_chnl_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_chnl_test_labels = []
    y_chnl_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_chnl_test_labels = np.append(y_chnl_test_labels,labels.cpu().numpy())
            y_chnl_test_pred = np.append(y_chnl_test_pred,outputs.cpu().numpy())

    ########### category ########
    # construct NN for cate, 因为cate的输入层神经元个数少于full model
    class CreditNet(nn.Module):
        def __init__(self): #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
            super().__init__()
            self.fc1 = nn.Linear(layer0_cate, layer1) # fc: fully connected
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
    ### X_cate_train, y_cate_train
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
    X_cate_train = torch.from_numpy(X_cate_train) # transfer to Tensor, no need to add .double(), because it is already float data type
    y_cate_train = torch.from_numpy(y_cate_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
    if torch.cuda.is_available():
        X_cate_train = X_cate_train.to(device1)
        y_cate_train = y_cate_train.to(device1)
    train = data_utils.TensorDataset(X_cate_train, y_cate_train) # adjust format. 打包X Y 用于训练 
    train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，
    writer = SummaryWriter()
    for epoch in range(training_epochs):
        y_cate_train_labels = [] # create an empty array 
        y_cate_train_pred = []   
        for b, data in enumerate(train_loader, 0): # 取batch
            inputs, labels = data#.cuda() # inputs and labels follows that when loaded
            if torch.cuda.is_available():
                inputs = inputs.to(device1)
                labels = labels.to(device1)
            optimizer.zero_grad() #reset gradients, i.e. zero the gradient buffers
            y_pred = net(inputs) # obtain the predicted values, a Tensor
            y_pred = y_pred.view(y_pred.size()[0])
            y_cate_train_labels = np.append(y_cate_train_labels, labels.cpu().numpy())
            y_cate_train_pred = np.append(y_cate_train_pred,y_pred.detach().cpu().numpy())
            loss_fn = nn.BCELoss() # binary cross entropy loss, with weights
            if torch.cuda.is_available():
                loss_fn = loss_fn.to(device1)
            loss = loss_fn(y_pred, labels) # 2 tensors in, 1 value out
            loss.backward() # backward pass
            optimizer.step() # update weights
            if b % 100 == 0: # if b整除10, then output loss
                #print('Epochs: {}, batch: {} loss: {}'.format(epoch, b, loss))
                writer.add_scalar('train_loss_NN_resample_3_cate',loss, epoch)
    writer.close()
    ### X_cate_test y_cate_test
    X_cate_test = torch.from_numpy(X_cate_test) # check the tested results
    y_cate_test = torch.from_numpy(y_cate_test).double()
    if torch.cuda.is_available():
        X_cate_test = X_cate_test.to(device1)
        y_cate_test = y_cate_test.to(device1)
    test = data_utils.TensorDataset(X_cate_test, y_cate_test)
    test_loader = data_utils.DataLoader(test, batch_size=minibatch_size, shuffle=True)
    y_cate_test_labels = []
    y_cate_test_pred = []
    with torch.no_grad(): #上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            outputs = outputs.view(outputs.size()[0])
            y_cate_test_labels = np.append(y_cate_test_labels,labels.cpu().numpy())
            y_cate_test_pred = np.append(y_cate_test_pred,outputs.cpu().numpy())

    #### ROC curve and Area-Under-Curve (AUC)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)
    test_fpr_base, test_tpr_base, te_thresholds_base = roc_curve(y_base_test_labels, y_base_test_pred)
    test_fpr_demo, test_tpr_demo, te_thresholds_demo = roc_curve(y_demo_test_labels, y_demo_test_pred)
    test_fpr_time, test_tpr_time, te_thresholds_time = roc_curve(y_time_test_labels, y_time_test_pred)
    test_fpr_chnl, test_tpr_chnl, te_thresholds_chnl = roc_curve(y_chnl_test_labels, y_chnl_test_pred)
    test_fpr_cate, test_tpr_cate, te_thresholds_cate = roc_curve(y_cate_test_labels, y_cate_test_pred)
    auc_full.append(auc(test_fpr, test_tpr))
    auc_base.append(auc(test_fpr_base, test_tpr_base))
    auc_demo.append(auc(test_fpr_demo, test_tpr_demo))
    auc_time.append(auc(test_fpr_time, test_tpr_time))
    auc_chnl.append(auc(test_fpr_chnl, test_tpr_chnl))
    auc_cate.append(auc(test_fpr_cate, test_tpr_cate))
    print("round:"+str(random_state))
    
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
plt.title('Feature group AUC by NN')
plt.legend(loc=0)
plt.savefig("/Users/HaoLI/Stata/credit/out/AUC_group_NN.pdf", bbox_inches = 'tight')                        
plt.show()
    
    
    
    