# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 10:30:33 2022

@author: fedib
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import dataset
dataset=pd.read_csv("C:/STUDY/CIII/Deep learning/tp/tp2/Dataset_spine.csv")

dataset=dataset.drop(['Unnamed: 13'], axis=1) # drop column unnamed 13 because its unnamed with random info

dataset.describe()
label=dataset['Class_att']
data =dataset.drop(['Class_att'], axis=1)


train_data,test_data,train_label,test_label=train_test_split(data,label,test_size = 0.33,random_state = 0)

from sklearn.neural_network import MLPClassifier

clf=MLPClassifier(activation='logistic',hidden_layer_sizes=(50,50,50),max_iter=500,solver='adam',random_state=0)
clf.fit(train_data,train_label)
pred=clf.predict(test_data)
ACC=accuracy_score(test_label, pred)*100
ACC

"""""""""""""""""""""""""""""""""""""""""""""EX2"""""""""""""""""""""""""""""""""""""""""""""
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


