#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:32:14 2021

@author: HaoLI
"""

# importing package
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
# create data
x = [5,10,20	,30,40,50]
y_LR0 = [0.025950498,0.095450502,0.211364747,0.309495269,0.431343679,0.529475391]
y_XGB = [0.044033,0.106569,0.209072,0.332866,0.427476,0.522393]
y_lightGBM = [0.039457831,0.101177375,0.204518847,0.325524337,0.432782302,0.528707172]
y_ada = [0.033139185,0.098403563,0.198324472,0.307053518,0.418654037,0.507337749]
y_RF = [0.036183221,0.100177327,0.20353803,0.322442248,0.42606804,0.520754796]

fig, ax = plt.subplots()  
# plot lines

plt.plot(x, y_XGB, label = "XGBoost",linestyle="-",
         marker='v', markerfacecolor='white', markersize=8, 
         color='blue', linewidth=2)
plt.plot(x, y_lightGBM, label = "lightGBM",linestyle="--",
         marker='d', markerfacecolor='white', markersize=8, 
         color='red', linewidth=2)
plt.plot(x, y_ada, label = "adaBoost",linestyle="-.",
         marker='D', markerfacecolor='white', markersize=8, 
         color='orange', linewidth=2)
plt.plot(x, y_RF, label = "RF",linestyle="-",
         marker='h', markerfacecolor='white', markersize=8, 
         color='grey', linewidth=2)
plt.plot(x, y_LR0, label = "LR0",linestyle="--",
         marker='o', markerfacecolor='white', markersize=8, 
         color='cyan', linewidth=2)
plt.xlabel('Rejection rate (%)')
# Set the y axis label of the current axis.
plt.ylabel('Reduction rate')
# Set a title of the current axes.
plt.title('Tree-based models')
#设置坐标轴刻度
my_x_ticks = [5,10,20,30,40,50]
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.legend()
plt.savefig("/Users/HaoLI/Stata/credit/out/rejectionrate_tree.pdf", bbox_inches = 'tight')                        
plt.show()

#################
x = [5,10,20	,30,40,50]
y_LR0 = [0.025950498,0.095450502,0.211364747,0.309495269,0.431343679,0.529475391]
y_LR1 = [0.025957064, 0.095129649,0.211480506,0.309317393,0.431149031,0.529291548]
y_LR2 = [0.025844846,0.094709035,0.211064729,0.309129572,0.43126421,0.529513741]

fig, ax = plt.subplots()  
# plot lines

plt.plot(x, y_LR1, label = "LR1",linestyle="-",
         marker='v', markerfacecolor='white', markersize=8, 
         color='blue', linewidth=2)
plt.plot(x, y_LR2, label = "LR2",linestyle="-.",
         marker='d', markerfacecolor='white', markersize=8, 
         color='red', linewidth=2)
plt.plot(x, y_LR0, label = "LR0",linestyle="--",
         marker='o', markerfacecolor='white', markersize=8, 
         color='cyan', linewidth=2)
plt.xlabel('Rejection rate (%)')
# Set the y axis label of the current axis.
plt.ylabel('Reduction rate')
# Set a title of the current axes.
plt.title('Linear combination with regularization')
#设置坐标轴刻度
my_x_ticks = [5,10,20,30,40,50]
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.legend()
plt.savefig("/Users/HaoLI/Stata/credit/out/rejectionrate_LR.pdf", bbox_inches = 'tight')                        
plt.show()

#########
x = [5,10,20	,30,40,50]
y_NN = [0.000490293,0.052727525,0.159766327,0.260855313,0.36672587,0.472045426]
fig, ax = plt.subplots()  
# plot lines

plt.plot(x, y_NN, label = "NN",linestyle="-",
         marker='v', markerfacecolor='white', markersize=8, 
         color='blue', linewidth=2)
plt.plot(x, y_LR0, label = "LR0",linestyle="--",
         marker='o', markerfacecolor='white', markersize=8, 
         color='cyan', linewidth=2)
plt.xlabel('Rejection rate (%)')
# Set the y axis label of the current axis.
plt.ylabel('Reduction rate')
# Set a title of the current axes.
plt.title('Neural networks')
#设置坐标轴刻度
my_x_ticks = [5,10,20,30,40,50]
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.xticks(my_x_ticks)
#plt.yticks(my_y_ticks)
plt.legend()
plt.savefig("/Users/HaoLI/Stata/credit/out/rejectionrate_NN.pdf", bbox_inches = 'tight')                        
plt.show()
