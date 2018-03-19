# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:43:21 2018

@author: Zongjian Fan
"""
import scipy.stats as stats
import numpy as np
"""
Convert original data to .csv(only need to run at the first time)
"""
def ConvertData():
    p1=np.load('D:\hw1\ecs171train.npy')
    p2=np.load('D:\hw1\ecs171test.npy')
    #first colume: id number; last colume: loss of this id; else: features of this id(f1~f778, 769 in total)
    p1_decode=[]
    p2_decode=[]
    for line in p1:
        #convert numpy.byte to string
        a=line.decode('UTF-8').split(',')
        p1_decode.append(a)
    for line in p2:
        #convert numpy.byte to string
        a=line.decode('UTF-8').split(',')
        p2_decode.append(a)
    #print(len(p_decode[0]))
    train_y=np.zeros((50000,1),dtype=np.float64)#loss
    train_x=np.zeros((769, 50000),dtype=np.float64)#features
    test=np.zeros((769, 50000),dtype=np.float64)
    feature=[]#features' name
    idnumber=[]#id numbers
    for i in range(770):
    #    temp=[]
        for j in range(50001):
            idnumber.append(p1_decode[j][0])
            if j==0:
                feature.append(p1_decode[j][i+1])
            elif i==769:
                train_y[j-1][0]=float(p1_decode[j][i+1])
            elif p1_decode[j][i+1]=='NA':
                train_x[i][j-1]=np.nan
            elif p2_decode[j][i+1]=='NA':
                test[i][j-1]=np.nan
            else:
                train_x[i][j-1]=float(p1_decode[j][i+1])
                test[i][j-1]=float(p1_decode[j][i+1])
    print(train_x.shape, train_y.shape, test.shape)
    np.savetxt("train_x.csv", train_x, delimiter=',')  
    np.savetxt("train_y.csv", train_y, delimiter=',')
    np.savetxt("test.csv", test, delimiter=',')  
"""
Load converted .csv data
"""
def LoadData():
    train_x = np.loadtxt(open("train_x.csv","rb"), delimiter=",", skiprows=0)
    train_y = np.loadtxt(open("train_y.csv","rb"), delimiter=",", skiprows=0)
    test = np.loadtxt(open("test.csv","rb"), delimiter=",", skiprows=0)
    return train_x, train_y, test
"""
Compute NA percentage
"""
def NAPercent(train_x):
    NApercent=np.array((range(769),np.zeros(769))).transpose()
    for i in range(769):
        count=np.count_nonzero(np.isnan(train_x[i]))
        NApercent[i][1]=count/50000
    np.savetxt("NApercent.csv", NApercent, delimiter=',')
    return NApercent
"""
Delete row(in both train and test data) with NA>=5% and replace all other NAs with mean value of that feature
"""
def deleteNA(train_x, test, NApercent):
    features=[i for i in range(769)]
    delete_list=[]
    for i in range(769):
        if NApercent[i][1]>=0.05:
            delete_list.append(i)
            features[i]=np.nan#lable deleted features as np.nan
    delete_tuple=tuple(delete_list)
    train_x=np.delete(train_x, delete_tuple, axis=0)
    test=np.delete(test, delete_tuple, axis=0)
    CurrentFeatureNum=train_x.shape[0]
    #replace NA with mean value of current feature
    for i in range(CurrentFeatureNum):
        nan_index=np.argwhere(np.isnan(train_x[i]))
        mean=np.nanmean(train_x[i])
        nan_index_test=np.argwhere(np.isnan(test[i]))
        mean_test=np.nanmean(test[i])
        for index in nan_index:
            train_x[i][index[0]]=mean
        for index in nan_index_test:
            test[i][index[0]]=mean_test
    print(train_x.shape)
    print(test.shape)
    return train_x, test
#    np.savetxt("train_x_1.csv", train_x_1, delimiter=',')  

#train_x_1 = np.loadtxt(open("train_x_1.csv","rb"), delimiter=",", skiprows=0)
"""
Delete duplicate data(if two features have more than 50% same elements, we regard it as duplicate)
"""
def DeleteDuplicate(train_x, test):
    CurrentFeatureNum=train_x.shape[0]
    delete_list=[]
    for i in range(CurrentFeatureNum):
        for j in range(i+1, CurrentFeatureNum):
            temp=train_x[i]-train_x[j]
            if np.count_nonzero(temp) <=20000:#nonzero elements<=20000: i,j duplicate
                delete_list.append(j)
    delete_tuple=tuple(delete_list)
    train_x=np.delete(train_x, delete_tuple, axis=0)
    test=np.delete(test, delete_tuple, axis=0)
    print(train_x.shape)
    print(test.shape)
    return train_x, test
#    np.savetxt("train_x_2.csv", train_x_2, delimiter=',')  

"""
Find feature pairs with correlation>0.996 and keep their difference as new features
"""
def GoldenFeature(train_x, test):
    CurrentFeatureNum=train_x.shape[0]
    GoldenFeatureList=[]
    FinalTestList=[]
    difference_list=[]
    for i in range(CurrentFeatureNum):
        for j in range(i+1, CurrentFeatureNum):
            corr=stats.pearsonr(train_x[i],train_x[j])[0]
            if corr >=0.996:
                difference_list.append([i,j])
                GoldenFeatureList.append(train_x[i]-train_x[j])
                FinalTestList.append(test[i]-test[j])
    train_x=np.array(GoldenFeatureList, dtype=np.float64)
    test=np.array(FinalTestList, dtype=np.float64)
    print(train_x.shape)
    print(test.shape)
    return train_x, test
#    np.savetxt("GoldenFeature.csv", GoldenFeature, delimiter=',') 
#    np.savetxt("GoldenFeatureT.csv", GoldenFeature.transpose(), delimiter=',') 
"""
Remove all feature pairs with correlation>0.99
"""
#GoldenFeature = np.loadtxt(open("GoldenFeature.csv","rb"), delimiter=",", skiprows=0)
def FinalFeature(train_x, test):
    CurrentFeatureNum=train_x.shape[0]
    remove_list=[]
    for i in range(CurrentFeatureNum):
        for j in range(i+1, CurrentFeatureNum):
            corr=stats.pearsonr(train_x[i],train_x[j])[0]
            if corr>0.99:
                if j not in remove_list:
                    remove_list.append(j)
    remove_tuple=tuple(remove_list)
    train_x=np.delete(train_x, remove_tuple, axis=0)
    test=np.delete(test, remove_tuple, axis=0)
#    np.savetxt("FinalTest.csv", FinalTest, delimiter=',') 
#    np.savetxt("FinalTestT.csv", FinalTest.transpose(), delimiter=',') 
#    np.savetxt("FinalFeature.csv", FinalFeature, delimiter=',') 
#    np.savetxt("FinalFeatureT.csv", FinalFeature.transpose(), delimiter=',')
    return train_x, test

"""
Output .csv include training data(FinalFeature: 50000*features; Loss: 50000*1), testing data(FinalTest: 50000*features)
"""
def main():
#    ConvertData()#only need to run it at the first time
    train_x, train_y, test=LoadData()
    NA=NAPercent(train_x)
    train_x, test=deleteNA(train_x, test, NA)
    train_x, test=DeleteDuplicate(train_x, test)
    train_x, test=GoldenFeature(train_x, test)
    train_x, test=FinalFeature(train_x, test)
    np.savetxt("FinalTest.csv", test.transpose(), delimiter=',') 
    np.savetxt("FinalFeature.csv", train_x.transpose(), delimiter=',') 
    np.savetxt("Loss.csv", train_y, delimiter=',')
        
if __name__=='__main__':
    main()
        
        
        
        
        