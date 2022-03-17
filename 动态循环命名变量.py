#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 22:58:30 2021

@author: HaoLI
"""


for i in range(0,5):
    locals()['x'+str(i)] = i
    print(locals()['x%s'%i])
    