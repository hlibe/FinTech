#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:24:00 2021

@author: HaoLI
"""
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#from graphviz import Digraph
import os

os.getcwd()
#os.chdir('/Users/HaoLI/Dropbox/FinTech/raw_data')
os.chdir('/Users/HaoLI/Stata/credit/data')

# #### correlation matrix

# #### boxplot 所有算法
df = pd.read_excel('boxplot_pythonResults20211210.xlsx', sheet_name='basemodel')
#df1 = df[df['Neural Networks'] > 0.5].transpose()
df1 = df.transpose()

fig, ax = plt.subplots(figsize = (16, 8))
bp = ax.boxplot(df)
ax.set_xlabel('Prediction models', fontsize = 16) #  fontweight='bold'
ax.set_ylabel('Out-of-sample areas under the roc curve (AUC)', fontsize = 16)
plt.xticks(range(1, 9), list(df1.index), rotation = 45, ha = 'center', fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.savefig('boxplot.pdf')
plt.show()

