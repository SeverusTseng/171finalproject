# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:43:21 2018

@author: Zongjian Fan
"""

eps=0.01
dir = '.'
import scipy.stats as stats
import math
import numpy as np
import operator
from sklearn.feature_selection import f_regression
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
    train_y=np.zeros((50000,1),dtype=np.float32)#loss
    train_x=np.zeros((769, 50000),dtype=np.float32)#features
    test=np.zeros((769, 55471),dtype=np.float32)
    feature=[]#features' name
    idnumber=[]#id numbers
    for i in range(770):
    #    temp=[]
        for j in range(len(p1_decode)):
            idnumber.append(p1_decode[j][0])
            if j==0:
                feature.append(p1_decode[j][i+1])
            elif i==769:
                train_y[j-1][0]=float(p1_decode[j][i+1])
            elif p1_decode[j][i+1]=='NA':
                train_x[i][j-1]=np.nan
            else:
                train_x[i][j-1]=float(p1_decode[j][i+1])
    for i in range(769):
    #    temp=[]
        for j in range(len(p2_decode)):
            idnumber.append(p2_decode[j][0])
            if p2_decode[j][i+1]=='NA':
                test[i][j]=np.nan
            else:
                test[i][j]=float(p2_decode[j][i+1])
    print(train_x.transpose().shape, train_y.shape, test.shape)
    np.savetxt("train_x.csv", train_x.transpose(), delimiter=',')  
    np.savetxt("train_y.csv", train_y, delimiter=',')
    np.savetxt("test.csv", test.transpose(), delimiter=',')  
"""
Load converted .csv data
"""
def LoadData():
    print('loading data')
    train_x = np.loadtxt(open("train_x.csv","rb"), delimiter=",", skiprows=0)
    train_y = np.loadtxt(open("train_y.csv","rb"), delimiter=",", skiprows=0)
    test = np.loadtxt(open("test.csv","rb"), delimiter=",", skiprows=0)
    print(train_x.shape, train_y.shape, test.shape)
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
    print('deleting NA...')
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
    print('deleting duplicate...')
    CurrentFeatureNum=train_x.shape[0]
    delete_list=[]
    for i in range(CurrentFeatureNum):
        for j in range(i+1, CurrentFeatureNum):
            temp=train_x[i]-train_x[j]
            if np.count_nonzero(temp) <=20000 and i not in delete_list:#nonzero elements<=20000: i,j duplicate
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
    print('golden feature...')
    CurrentFeatureNum=train_x.shape[0]
    GoldenFeatureList=[]
    FinalTestList=[]
    difference_list=[]
    for i in range(CurrentFeatureNum):
        for j in range(i+1, CurrentFeatureNum):
            corr=stats.pearsonr(train_x[i],train_x[j])[0]
            if corr >=0.993:
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
    print('final feature...')
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
    print(train_x.shape)
    print(test.shape)
    return train_x, test
"""
Generate all index list for feature generating by combination of features and remove high-correlated pairs
"""
# feature_pair_sub_list
def feature_pair_sub_list(train_x, train_y):
    feature_size = len(train_x[0])
    sub_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print (i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] - train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                sub_feature_corr_list.append(feature_corr)
    
    sub_list = [corr for corr in sub_feature_corr_list if abs(corr[2])>eps]
    sorted_sub_list = sorted(sub_list, key=lambda corr:abs(corr[2]), reverse=True)  
    temp_list = get_distinct_feature_pairs(sorted_sub_list)
    dist_sub_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_sub_list = [[520, 521], [271, 521], [271, 520]]
    feature_pair_sub_list.extend(dist_sub_list[1:])
    np.savetxt("feature_pair_sub_list.csv", feature_pair_sub_list, delimiter=',')  
    return feature_pair_sub_list

#feature_pair_plus_list
def feature_pair_plus_list(train_x, train_y):
    feature_size = len(train_x[0])
    plus_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print (i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, corr[0])
                plus_feature_corr_list.append(feature_corr)

    plus_list = [corr for corr in plus_feature_corr_list if abs(corr[2])>eps]
    sorted_plus_list = sorted(plus_list, key=lambda corr:abs(corr[2]), reverse=True)
    temp_list = get_distinct_feature_pairs(sorted_plus_list)
    dist_plus_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_plus_list = dist_plus_list
    np.savetxt("feature_pair_plus_list.csv", feature_pair_plus_list, delimiter=',')  
    return feature_pair_plus_list
#feature_pair_mul_list
def feature_pair_mul_list(train_x, train_y):
    feature_size = len(train_x[0])
    mul_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print (i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] * train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                mul_feature_corr_list.append(feature_corr)

    mul_list = [corr for corr in mul_feature_corr_list if abs(corr[2])>eps]
    sorted_mul_list = sorted(mul_list, key=lambda corr:abs(corr[2]), reverse=True)
    temp_list = get_distinct_feature_pairs(sorted_mul_list)
    dist_mul_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_mul_list = dist_mul_list
    np.savetxt("feature_pair_mul_list.csv", feature_pair_mul_list, delimiter=',')  
    return feature_pair_mul_list
#feature_pair_divide_list
def feature_pair_divide_list(train_x, train_y):
    feature_size = len(train_x[0])
    divide_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print (i)
        for j in range(feature_size):
            if i != j:
                try:
                    res = train_x[:,i] / train_x[:,j]
                    corr = stats.pearsonr(res, train_y)
                    if abs(corr[0]) < eps:
                        continue
                    feature_corr = (i, j, abs(corr[0]))
                    divide_feature_corr_list.append(feature_corr)
                except ValueError:
                    print ('divide 0')    
    divide_list = [corr for corr in divide_feature_corr_list if abs(corr[2])>eps]
    sorted_divide_list = sorted(divide_list, key=lambda corr:abs(corr[2]), reverse=True)
    temp_list = get_distinct_feature_pairs(sorted_divide_list)
    dist_divide_list = [[corr[0],corr[1]] for corr in temp_list]
    feature_pair_divide_list = dist_divide_list
    np.savetxt("feature_pair_divide_list.csv", feature_pair_divide_list, delimiter=',')  
    return feature_pair_divide_list
"""
Get all indexs of not highly correlated features
"""
def get_distinct_feature_pairs(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for i in range(len(sorted_corr_list)):
        if sorted_corr_list[i][0] not in dis_ind and sorted_corr_list[i][1] not in dis_ind:
            dis_ind[sorted_corr_list[i][0]] = 1
            dis_ind[sorted_corr_list[i][1]] = 1
            distinct_list.append(sorted_corr_list[i])
    return distinct_list
"""
Modify feature set based on index lists
"""
def get_data(train_x, feature_indexs, feature_pair_sub_list, feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list):
    sub_train_x = train_x[:,feature_indexs]


    for i in range(len(feature_pair_sub_list)):
        ind_i = feature_pair_sub_list[i][0]
        ind_j = feature_pair_sub_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i]-train_x[:,ind_j]))

    for i in range(len(feature_pair_plus_list)):
        ind_i = feature_pair_plus_list[i][0]
        ind_j = feature_pair_plus_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] + train_x[:,ind_j]))

    for i in range(len(feature_pair_mul_list)):
        ind_i = feature_pair_mul_list[i][0]
        ind_j = feature_pair_mul_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] * train_x[:,ind_j]))

    for i in range(len(feature_pair_divide_list)):
        ind_i = feature_pair_divide_list[i][0]
        ind_j = feature_pair_divide_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] / train_x[:,ind_j]))
        
    return sub_train_x
"""
Generate new feature sets
"""
def FeatureGenerate(train_x, train_y, test_x):
    sub_list=feature_pair_sub_list(train_x, train_y)
    plus_list=feature_pair_sub_list(train_x, train_y)
    mul_list=feature_pair_sub_list(train_x, train_y)
    divide_list=feature_pair_sub_list(train_x, train_y)
    feature_indexs=getTopFeatures(train_x, train_y, n_features=100)
    train_x=get_data(train_x, feature_indexs, sub_list, plus_list, mul_list, divide_list)
    test_x=get_data(test_x, feature_indexs, sub_list, plus_list, mul_list, divide_list)
    return train_x, test_x

"""
Pick features highly correlated with loss
"""
def getTopFeatures(train_x, train_y, n_features=100):
    f_val, p_val = f_regression(train_x,train_y)
    f_val_dict = {}
    p_val_dict = {}
    for i in range(len(f_val)):
        if math.isnan(f_val[i]):
            f_val[i] = 0.0
        f_val_dict[i] = f_val[i]
        if math.isnan(p_val[i]):
            p_val[i] = 0.0
        p_val_dict[i] = p_val[i]
    
    sorted_f = sorted(f_val_dict.items(), key=operator.itemgetter(1), reverse=True)
    
    feature_indexs = []
    for i in range(0,n_features):
        feature_indexs.append(sorted_f[i][0])   
    return feature_indexs


def main():
#    ConvertData()#only need to run it at the first time
    train_x, train_y, test=LoadData()
    NA=NAPercent(train_x)
    train_x, test=deleteNA(train_x, test, NA)
    train_x, test=DeleteDuplicate(train_x, test)
    strategy='1'
    if strategy == '1':
        train_x, test=FeatureGenerate(train_x.transpose(), train_y, test.transpose())
        np.savetxt("FinalTest.csv", test, delimiter=',') 
        np.savetxt("FinalFeature.csv", train_x, delimiter=',') 
        np.savetxt("Loss.csv", train_y, delimiter=',')
    else:
        train_x, test=GoldenFeature(train_x, test)
        train_x, test=FinalFeature(train_x, test)
        np.savetxt("FinalTest.csv", test.transpose(), delimiter=',') 
        np.savetxt("FinalFeature.csv", train_x.transpose(), delimiter=',') 
        np.savetxt("Loss.csv", train_y, delimiter=',')
    
        
if __name__=='__main__':
    main()
