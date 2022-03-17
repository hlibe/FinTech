#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 09:52:31 2021

@author: HaoLI
"""
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
import random
use_gpu = torch.cuda.is_available()
print("GPU",use_gpu)
list_rec = [] #记录参数
randomseed = 22
random.seed(randomseed)

layer1=196
layer2=196
oversample_ratio=0.5
training_epochs = 80
minibatch_size = 5000
learning_rate=2e-4
penalty=2 #p=1 for L1; p=0 for L2, weight_decay only for L2 ; p=2 for default. 范数计算中的幂指数值，默认求2范数. 当p=0为L2正则化,p=1为L1正则化
weight_decay=0.0125 #weight_decay 就是 L2 正则项
dropout=0.0

#os.getcwd()
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

layer0=len(X.columns) # input层的神经元个数

#min_max_scaler = MinMaxScaler()
#X = min_max_scaler.fit_transform(X)
sc = StandardScaler()# transform X into standard normal distribution for each column. X from dataframe to array
X = sc.fit_transform(X) 

ros = RandomOverSampler(random_state=0)
list_rec_NN = [] #记录参数

for layer1 in [196]:
    for layer2 in [196]:
        for weight_decay in [0.0125]:
            for training_epochs in [80]:
                for minibatch_size in [5000]:
                    for random_state in range(0,20):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state) # data types are dataframe
                        X_train, y_train = ros.fit_resample(X_train, y_train)
                        y_train = y_train.values
                        y_test = np.array(y_test)                       
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
                                x = F.dropout(x, p=dropout)
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
                        #loss_fn = nn.CrossEntropyLoss()
                        #loss_fn = nn.BCELoss() # binary cross entropy loss
                        #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) # auto adjust lr, better than sgd
                        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.9) # auto adjust lr, better than sgd; sgd stable 
                        #优化器采用Adam，并且设置参数weight_decay=0.0，即无正则化的方法 
                        #优化器采用Adam，并且设置参数weight_decay=10.0，即正则化的权重lambda =10.0
                        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay) # auto adjust lr, better than sgd
                        # if we use L2 regularization, apply the following line
                        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)



                        X_train = torch.from_numpy(X_train) # transfer to Tensor, no need to add .double(), because it is already float data type
                        y_train = torch.from_numpy(y_train).double() # .double() makes the data type float, 在pytorch中，只有浮点类型的数才有梯度
                        #weights_tensor = torch.from_numpy(overwt_arr_y_lossfn)

                        if torch.cuda.is_available():
                            X_train = X_train.to(device1)
                            y_train = y_train.to(device1)
                            #weights_tensor = weights_tensor.to(device1)

                        train = data_utils.TensorDataset(X_train, y_train) # adjust format. 打包X Y 用于训练 
                        train_loader = data_utils.DataLoader(train, batch_size=minibatch_size, shuffle=True) # 在PyTorch中训练模型经常要使用它. batch_size定义每次喂给神经网络多少行数据. shuffle在每次迭代训练时是否将数据洗牌，默认设置是False。将输入数据的顺序打乱，是为了使数据更有独立性，

                        # !tensorboard --logdir './runs' #远程的notebook中如果使用魔法函数, 可能会导致你无法打开tensorboard的http服务

                        from tensorboardX import SummaryWriter
                        writer = SummaryWriter()

                        #%reload_ext tensorboard
                         # Load the TensorBoard notebook extension

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
                                    writer.add_scalar('NN_oversample',loss, epoch)

                        writer.close()
                        #%tensorboard --logdir   #定位tensorboard读取的文件目录

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

                        #### plot ROC, compute AUC ###
                        # y_true is ground truth labels, y_score is predicted probabilities generated by sklearn classifier
                        test_fpr, test_tpr, te_thresholds = roc_curve(y_true = y_test_labels, y_score = y_test_pred)
                        #print("AUC TEST = ", auc(test_fpr, test_tpr))
                        train_fpr, train_tpr, tr_thresholds = roc_curve(y_true = y_train_labels, y_score = y_train_pred) # /w_ytrain, such that return the array to 0,1 array
                        #print("AUC TRAIN = ", auc(train_fpr, train_tpr))

                        #print('resample: {}, Epochs: {}, batch size: {}, '.format(oversample_ratio, training_epochs, minibatch_size))
                        #print(net)

                        plt.grid()
                        plt.plot(train_fpr, train_tpr, label=" AUC TRAIN ="+str(auc(train_fpr, train_tpr)))
                        plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
                        plt.plot([0,1],[0,1],'g--')
                        plt.legend()
                        plt.xlabel("True Positive Rate")
                        plt.ylabel("False Positive Rate")
                        t='''
                        training_epochs=%s, minibatch_size=%s,
                        learning_rate=%s, penalty=L%s, weight_decay=%s, 
                        dropout=%s, 24=>%s=>%s=>1, myoversampling, random_state=%s, 
                        randomseed=%s
                        '''%(training_epochs,minibatch_size,learning_rate, 
                        penalty, weight_decay, dropout, layer1, layer2, random_state,randomseed)
                        plt.title("AUC(Neural Network ROC curve)"+t)
                        plt.grid(color='black', linestyle='-', linewidth=0.5)
                        time1 = datetime.datetime.now()
                            #对现在时间格式化，以此作为文件名
                        time2 = time1.strftime('%Y-%m-%d-%H%M%S')
                        plt.savefig("/Users/HaoLI/Stata/credit/out/ROC figure/Figure_"+time2+".png", bbox_inches = 'tight')                        
                        plt.show()
                        list_rec.append([auc(train_fpr, train_tpr), auc(test_fpr, test_tpr),
                                   training_epochs,minibatch_size,learning_rate, 
                                   penalty, weight_decay, dropout, layer1, layer2,
                                   random_state, randomseed
                                   ])

list_rec_1 = list_rec
df = pd.DataFrame(list_rec, columns = ['IS_AUC','OOS_AUC','training_epochs',
                                       'minibatch_size','learning_rate', 
                                       'penalty', 'weight_decay', 'dropout', 
                                       'layer1', 'layer2', 'random_state','randomseed'])
df.to_csv('NN_adj.csv')              