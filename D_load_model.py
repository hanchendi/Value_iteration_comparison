import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import os
cwd=os.getcwd()
import scipy.io
from scipy.io import loadmat
import random
import math
from collections import defaultdict

n=10
N_test=5000
model1 = Sequential()
model1.add(Dense(100, input_dim=6, activation='relu'))
model1.add(Dense(100,activation='relu'))
model1.add(Dense(100,activation='relu'))
model1.add(Dense(100))
model1.add(Dense(1))

model1.load_weights('./training_model')

x = loadmat(cwd+'/x_test.mat')
x_test = x.get('x_test')
x_test = x_test.astype(np.float64)
for i in range(0,N_test):
    for j in range(0,6):
        x_test[i][j]=x_test[i][j]/(n-1)
        
y_pred_test = model1.predict(x_test)