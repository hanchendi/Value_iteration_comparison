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

N_iter=20
N_train=5000
N_test=5000

###########################################
# save matrix
###########################################

y_test=np.zeros([N_test,N_iter+1])
error_train=np.zeros(N_iter)

###########################################
# load the test point and normalized
###########################################

def Q_calculation(xs1,ys1,xs2,ys2,xf1,yf1):
    dis1=abs(xs1-xf1)+abs(ys1-yf1)
    dis2=abs(xs2-xf1)+abs(ys2-yf1)
    return min(dis1,dis2)

x = loadmat(cwd+'/x_test.mat')
x_test = x.get('x_test')
x_test = x_test.astype(np.float64)
for i in range(0,N_test):
    y_test[i][0]=Q_calculation(x_test[i][0],x_test[i][1],x_test[i][2],x_test[i][3],x_test[i][4],x_test[i][5])
    for j in range(0,6):
        x_test[i][j]=x_test[i][j]/(n-1)
        
###########################################
# First iteration
###########################################

temp_x=[0,0,-1,1,0]
temp_y=[-1,1,0,0,0]

x_train=np.zeros([N_train,6])
y_train=np.zeros(N_train)
for i in range(0,N_train):
    
    # random position
    xs1=random.randint(0, n-1)
    ys1=random.randint(0, n-1)
    xs2=random.randint(0, n-1)
    ys2=random.randint(0, n-1)
    xf1=random.randint(0, n-1)
    yf1=random.randint(0, n-1)

    # generate position for fly
    position_fly=[]
    for i1 in range(0,5):
        xt=xf1+temp_x[i1]
        yt=yf1+temp_y[i1]
        if xt>=0 and xt<n and yt>=0 and yt<n:
            position_fly.append([xt,yt])
    len_fly=len(position_fly)
    
    if (xs1==xf1 and ys1==yf1) or (xs2==xf1 and ys2==yf1):
        Q=0
    else:
        Q=math.inf
        
    for i1 in range(0,5):
        for i2 in range(0,5):
            xt1=xs1+temp_x[i1]
            yt1=ys1+temp_y[i1]
            xt2=xs2+temp_x[i2]
            yt2=ys2+temp_y[i2]
            if xt1>=0 and xt1<n and yt1>=0 and yt1<n and xt2>=0 and xt2<n and yt2>=0 and yt2<n:
                temp=0
                for j1 in range(0,len_fly):
                    temp+=Q_calculation(xt1,yt1,xt2,yt2,position_fly[j1][0],position_fly[j1][1])
                temp=temp/len_fly
                Q=min(Q,temp+1)
                
    x_train[i][0]=xs1/(n-1)
    x_train[i][1]=ys1/(n-1)
    x_train[i][2]=xs2/(n-1)
    x_train[i][3]=ys2/(n-1)
    x_train[i][4]=xf1/(n-1)
    x_train[i][5]=yf1/(n-1)
    
    y_train[i]=Q
    

model1 = Sequential()
model1.add(Dense(100, input_dim=6, activation='relu'))
model1.add(Dense(100,activation='relu'))
model1.add(Dense(100,activation='relu'))
model1.add(Dense(100))
model1.add(Dense(1))

model1.compile(loss='mean_squared_error', optimizer='adam')
model1.fit(x_train, y_train, epochs=500, batch_size=10)

y_pred_train = model1.predict(x_train)
y_pred_test = model1.predict(x_test)

error_train[0]=sum([abs(y_train[i]-y_pred_train[i]) for i in range(0,N_train)])/sum(y_train)

for i in range(0,N_test):
    y_test[i][1]=y_pred_test[i]

###########################################
# iteration
###########################################
    
for iteration_step in range(1,N_iter):
    
    x_train=np.zeros([N_train,6])
    y_train=np.zeros(N_train)
    
    # generate input file
    NN_input=[]
    dic=defaultdict(list)
    o=0
    for i in range(0,N_train):
    
        # random position
        xs1=random.randint(0, n-1)
        ys1=random.randint(0, n-1)
        xs2=random.randint(0, n-1)
        ys2=random.randint(0, n-1)
        xf1=random.randint(0, n-1)
        yf1=random.randint(0, n-1)
        
        x_train[i][0]=xs1
        x_train[i][1]=ys1
        x_train[i][2]=xs2
        x_train[i][3]=ys2
        x_train[i][4]=xf1
        x_train[i][5]=yf1

        # generate position for fly
        position_fly=[]
        for i1 in range(0,5):
            xt=xf1+temp_x[i1]
            yt=yf1+temp_y[i1]
            if xt>=0 and xt<n and yt>=0 and yt<n:
                position_fly.append([xt,yt])
        len_fly=len(position_fly)
        
        for i1 in range(0,5):
            for i2 in range(0,5):
                xt1=xs1+temp_x[i1]
                yt1=ys1+temp_y[i1]
                xt2=xs2+temp_x[i2]
                yt2=ys2+temp_y[i2]
                if xt1>=0 and xt1<n and yt1>=0 and yt1<n and xt2>=0 and xt2<n and yt2>=0 and yt2<n:
                    for j1 in range(0,len_fly):
                        dic[(i,i1,i2)].append(o)
                        NN_input.append([xt1/(n-1),yt1/(n-1),xt2/(n-1),yt2/(n-1),position_fly[j1][0]/(n-1),position_fly[j1][1]/(n-1)])
                        o+=1
    
    # calculate out put
    NN_input=np.array(NN_input,'float')
    NN_output=model1.predict(NN_input)
    
    # generate Q factor
    for i in range(0,N_train):
        
        xs1=x_train[i][0]
        ys1=x_train[i][1]
        xs2=x_train[i][2]
        ys2=x_train[i][3]
        xf1=x_train[i][4]
        yf1=x_train[i][5]
        if (xs1==xf1 and ys1==yf1) or (xs2==xf1 and ys2==yf1):
            Q=0
        else:
            Q=math.inf
            for i1 in range(0,5):
                for i2 in range(0,5):
                    
                    xt1=xs1+temp_x[i1]
                    yt1=ys1+temp_y[i1]
                    xt2=xs2+temp_x[i2]
                    yt2=ys2+temp_y[i2]
                    
                    if xt1>=0 and xt1<n and yt1>=0 and yt1<n and xt2>=0 and xt2<n and yt2>=0 and yt2<n:
                        temp=0
                        for j1 in dic[(i,i1,i2)]:
                            temp+=NN_output[j1]
                        temp=temp/len(dic[(i,i1,i2)])
                        Q=min(Q,temp+1)
                
        x_train[i][0]=xs1/(n-1)
        x_train[i][1]=ys1/(n-1)
        x_train[i][2]=xs2/(n-1)
        x_train[i][3]=ys2/(n-1)
        x_train[i][4]=xf1/(n-1)
        x_train[i][5]=yf1/(n-1)
    
        y_train[i]=Q
    
    model2 = Sequential()
    model2.add(Dense(100, input_dim=6, activation='relu'))
    model2.add(Dense(100,activation='relu'))
    model2.add(Dense(100,activation='relu'))
    model2.add(Dense(100))
    model2.add(Dense(1))

    model2.compile(loss='mean_squared_error', optimizer='adam')
    model2.fit(x_train, y_train, epochs=500, batch_size=10)

    y_pred_train = model2.predict(x_train)
    y_pred_test = model2.predict(x_test)

    error_train[iteration_step]=sum([abs(y_train[i]-y_pred_train[i]) for i in range(0,N_train)])/sum(y_train)

    for i in range(0,N_test):
        y_test[i][iteration_step+1]=y_pred_test[i]
    
    model1=model2
    print(iteration_step)
model1.save_weights('./training_model')
scipy.io.savemat(cwd+'/y_test.mat', mdict={'y_test': y_test})
scipy.io.savemat(cwd+'/error_train.mat', mdict={'error_train': error_train})