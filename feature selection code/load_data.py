# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:43:21 2018

@author: Zongjian Fan
"""
import scipy.stats as stats
import numpy as np
"""
Convert original data to .csv
"""
#p=np.load('D:\hw1\ecs171train.npy')
##first colume: id number; last colume: loss of this id; else: features of this id(f1~f778, 769 in total)
#p_decode=[]
#for line in p:
#    #convert numpy.byte to string
#    a=line.decode('UTF-8').split(',')
#    p_decode.append(a)
##print(len(p_decode[0]))
#train_y=np.zeros((1,50000),dtype=np.float64)#loss
#train_x=np.zeros((769,50000),dtype=np.float64)#features
#feature=[]#features' name
#idnumber=[]#id numbers
#for i in range(770):
##    temp=[]
#    for j in range(50001):
#        idnumber.append(p_decode[j][0])
#        if j==0:
#            feature.append(p_decode[j][i+1])
#        elif i==769:
#            train_y[0][j-1]=float(p_decode[j][i+1])
#        elif p_decode[j][i+1]=='NA':
#            train_x[i][j-1]=np.nan
#        else:
#            train_x[i][j-1]=float(p_decode[j][i+1])
#print(train_x.shape, train_y.shape)
#np.savetxt("train_x.csv", train_x, delimiter=',')  
#np.savetxt("train_y.csv", train_y, delimiter=',')  

train_x = np.loadtxt(open("train_x.csv","rb"), delimiter=",", skiprows=0)
train_y = np.loadtxt(open("train_y.csv","rb"), delimiter=",", skiprows=0)
"""
Compute NA percentage
"""
NApercent=np.array((range(769),np.zeros(769))).transpose()
for i in range(769):
    count=np.count_nonzero(np.isnan(train_x[i]))
    NApercent[i][1]=count/50000
np.savetxt("NApercent.csv", NApercent, delimiter=',')  
"""
Delete row with NA>=5% and replace all other NAs with mean value of that feature
"""
features=[i for i in range(769)]
delete_list=[]
for i in range(769):
    if NApercent[i][1]>=0.05:
        delete_list.append(i)
        features[i]=np.nan#lable deleted features as np.nan
delete_tuple=tuple(delete_list)
train_x_1=np.delete(train_x, delete_tuple, axis=0)
CurrentFeatureNum=train_x_1.shape[0]
#replace NA with mean value of current feature
for i in range(CurrentFeatureNum):
    nan_index=np.argwhere(np.isnan(train_x_1[i]))
    mean=np.nanmean(train_x_1[i])
    for index in nan_index:
        train_x_1[i][index[0]]=mean
print(train_x_1.shape)
np.savetxt("train_x_1.csv", train_x_1, delimiter=',')  

#train_x_1 = np.loadtxt(open("train_x_1.csv","rb"), delimiter=",", skiprows=0)
"""
Delete duplicate data(if two features have more than 50% same elements, we regard it as duplicate)
"""
CurrentFeatureNum=train_x_1.shape[0]
features_1=[i for i in range(CurrentFeatureNum)]
delete_list_1=[]
for i in range(CurrentFeatureNum):
    for j in range(i+1, CurrentFeatureNum):
        temp=train_x_1[i]-train_x_1[j]
        if np.count_nonzero(temp) <=25000:#nonzero elements<=25000: i,j duplicate
            delete_list_1.append(j)
delete_tuple_1=tuple(delete_list_1)
train_x_2=np.delete(train_x_1, delete_tuple_1, axis=0)
print(train_x_2.shape)
np.savetxt("train_x_2.csv", train_x_1, delimiter=',')  

"""
Find feature pairs with correlation>0.996 and keep their difference as new features
"""
CurrentFeatureNum=train_x_2.shape[0]
GoldenFeatureList=[]
difference_list=[]
for i in range(CurrentFeatureNum):
    for j in range(i+1, CurrentFeatureNum):
        corr=stats.pearsonr(train_x_2[i],train_x_2[j])[0]
        if corr >=0.996:
            difference_list.append([i,j])
            GoldenFeatureList.append(train_x_2[i]-train_x_2[j])
GoldenFeature=np.array(GoldenFeatureList, dtype=np.float64)
np.savetxt("GoldenFeature.csv", train_x_1, delimiter=',') 


