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

###########################
# Initial matrix
###########################

l=2 # look ahead steps
n=10 # dimension of matrix

model = Sequential()
model.add(Dense(100, input_dim=6, activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100))
model.add(Dense(1))

model.load_weights('./training_model')

T_O=np.zeros([n,n],'int')
O_T=np.zeros([n**2,2],'int')
o=0
for i in range(0,n):
    for j in range(0,n):
        T_O[i][j]=o
        O_T[o][0]=i
        O_T[o][1]=j
        o+=1


N_iter=10**4
time_gather=np.zeros(N_iter)

###########################
# multi run
###########################

for run_step in range(0,N_iter):

    xs1=0
    ys1=0
    
    xs2=0
    ys2=0
    
    xf1=9
    yf1=9
    
    if (xs1==xf1 and ys1==yf1) or (xs2==xf1 and ys2==yf1):
        capture_f1=True
    else:
        capture_f1=False
    
    position_save=[[xs1,ys1,xs2,ys2,xf1,yf1]]
    
    ###########################
    # Rollout
    ###########################
    
    temp_x=[0,0,-1,1,0]
    temp_y=[-1,1,0,0,0]
    while capture_f1 is False:
        
        # generate position of fly
        position_fly=[[[T_O[xf1][yf1]]]]
        for i in range(0,1):
            p=[]
            for path in position_fly:
                cur=path[-1]
                xf1_cur=O_T[cur[0]][0]
                yf1_cur=O_T[cur[0]][1]
                for i1 in range(0,5):
                    xt1=xf1_cur+temp_x[i1]
                    yt1=yf1_cur+temp_y[i1]
                    if xt1>=0 and xt1<n and yt1>=0 and yt1<n:
                        p.append(path+[[T_O[xt1][yt1]]])
                    #else:
                    #    p.append(path+[[T_O[xf1_cur][yf1_cur]]])
            position_fly=[[j2.copy() for j2 in j1] for j1 in p]
        len_fly=len(position_fly)
        
        # generate position of spider
        temp=[]
        for i1 in range(0,5):
            for i2 in range(0,5):
                xt1=xs1+temp_x[i1]
                yt1=ys1+temp_y[i1]
                xt2=xs2+temp_x[i2]
                yt2=ys2+temp_y[i2]
                if xt1>=0 and xt1<n and yt1>=0 and yt1<n and xt2>=0 and xt2<n and yt2>=0 and yt2<n:
                    
                    NN_input=np.zeros([1,6])
                    NN_input[0][0]=xt1/(n-1)
                    NN_input[0][1]=yt1/(n-1)
                    NN_input[0][2]=xt2/(n-1)
                    NN_input[0][3]=yt2/(n-1)
                    NN_input[0][4]=xf1/(n-1)
                    NN_input[0][5]=yf1/(n-1)
                    NN_output=model.predict(NN_input)
                    temp.append([NN_output[0][0],xt1,yt1,xt2,yt2])
        
        
        # find one path with smallest Q
        temp.sort()
        xs1=temp[0][1]
        ys1=temp[0][2]
        xs2=temp[0][3]
        ys2=temp[0][4]
        
        # move fly
        choose_move_fly=position_fly[random.randint(0, len_fly-1)][1]
        xf1=O_T[choose_move_fly[0]][0]
        yf1=O_T[choose_move_fly[0]][1]
        
        if (xs1==xf1 and ys1==yf1) or (xs2==xf1 and ys2==yf1):
            capture_f1=True
        
        position_save.append([xs1,ys1,xs2,ys2,xf1,yf1])        
    #capture_f1=True    
    #capture_f1=True
    time_gather[run_step]=len(position_save)-1
    print(run_step)
import os
cwd=os.getcwd()
import scipy.io
scipy.io.savemat(cwd+'/Rollout_traditional.mat', mdict={'time_gather': time_gather})   