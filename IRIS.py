# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:49:29 2023

@author: fedib
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
label = iris.target

train_data,test_data,train_label,test_label=train_test_split(data,label,test_size = 0.33,random_state = 0)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import losses

model= Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile('adam',loss=losses.mean_squared_error,metrics=['accuracy'])
model.fit(train_data,train_label,batch_size=16,epochs=10,validation_data=(test_data,test_label))
