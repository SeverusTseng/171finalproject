import scipy.stats as stats
import random
import re
import math
import collections
import numpy as np
import time
import operator
from scipy.io import mmread, mmwrite
from random import randint
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing as pp
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import  RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import ProbabilisticPCA, KernelPCA
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
import scipy.stats as stats
from sklearn import tree
from sklearn.feature_selection import f_regression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, f1_score
from sklearn.gaussian_process import GaussianProcess

def get_distinct_feature_pairs(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for i in range(len(sorted_corr_list)):
        if sorted_corr_list[i][0] not in dis_ind and sorted_corr_list[i][1] not in dis_ind:
            dis_ind[sorted_corr_list[i][0]] = 1
            dis_ind[sorted_corr_list[i][1]] = 1
            distinct_list.append(sorted_corr_list[i])
    return distinct_list

def get_feature_pair_sub_list(train_x, train_y, eps=0.01):
 
    feature_size = len(train_x[0])
    sub_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] - train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                sub_feature_corr_list.append(feature_corr)
    
    //sub_list = find_corr_pairs_sub(train_x, train_y, eps)
    sub_list = [corr for corr in sub_feature_corr_list if abs(corr[2])>eps]
    sorted_sub_list = sorted(sub_list, key=lambda corr:abs(corr[2]), reverse=True)
  
    temp_list = get_distinct_feature_pairs(sorted_sub_list)
    dist_sub_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_sub_list = [[520, 521], [271, 521], [271, 520]]
    feature_pair_sub_list.extend(dist_sub_list[1:])
    return feature_pair_sub_list

def get_feature_pair_plus_list(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    plus_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, corr[0])
                plus_feature_corr_list.append(feature_corr)

    //plus_list = find_corr_pairs_plus(train_x, train_y, eps)
    plus_list = [corr for corr in plus_feature_corr_list if abs(corr[2])>eps]
    sorted_plus_list = sorted(plus_list, key=lambda corr:abs(corr[2]), reverse=True)

    temp_list = get_distinct_feature_pairs(sorted_plus_list)
    dist_plus_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_plus_list = dist_plus_list
    return feature_pair_plus_list

def get_feature_pair_mul_list(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    mul_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] * train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                mul_feature_corr_list.append(feature_corr)

    //mul_list = find_corr_pairs_mul(train_x, train_y, eps)
    mul_list = [corr for corr in mul_feature_corr_list if abs(corr[2])>eps]
    sorted_mul_list = sorted(mul_list, key=lambda corr:abs(corr[2]), reverse=True)

    temp_list = get_distinct_feature_pairs(sorted_mul_list)
    dist_mul_list = [[corr[0], corr[1]] for corr in temp_list]
    feature_pair_mul_list = dist_mul_list
    return feature_pair_mul_list

def get_feature_pair_divide_list(train_x, train_y, eps=0.01):
    feature_size = len(train_x[0])
    divide_feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print i
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
    
    //divide_list = find_corr_pairs_divide(train_x, train_y, eps)
    divide_list = [corr for corr in divide_feature_corr_list if abs(corr[2])>eps]
    sorted_divide_list = sorted(divide_list, key=lambda corr:abs(corr[2]), reverse=True)

    temp_list = get_distinct_feature_pairs(sorted_divide_list)
    dist_divide_list = [[corr[0],corr[1]] for corr in feature_pair_divide_list]
    feature_pair_divide_list = dist_divide_list
    return feature_pair_divide_list


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
    
    sorted_f = sorted(f_val_dict.items(), key=operator.itemgetter(1),reverse=True)
    sorted_p = sorted(p_val_dict.items(), key=operator.itemgetter(1),reverse=True)
    
    feature_indexs = []
    for i in range(0,n_features):
        feature_indexs.append(sorted_f[i][0])
    
    return feature_indexs

def get_data(train_x, feature_indexs, feature_pair_sub_list, feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list):
    sub_train_x = train_x[:,feature_indexs]

    //feature_pair_sub_list = get_feature_pair_sub_list(train_x, train_y, 0.01)
    for i in range(len(feature_pair_sub_list)):
        ind_i = feature_pair_sub_list[i][0]
        ind_j = feature_pair_sub_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i]-train_x[:,ind_j]))

    //feature_pair_plus_list = get_feature_pair_plus_list(train_x, train_y, 0.01)
    for i in range(len(feature_plus_pair_list)):
        ind_i = feature_plus_pair_list[i][0]
        ind_j = feature_plus_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] + train_x[:,ind_j]))

    //feature_pair_plus_list = get_feature_pair_mul_list(train_x, train_y, 0.01)
    for i in range(len(feature_mul_pair_list)):
        ind_i = feature_mul_pair_list[i][0]
        ind_j = feature_mul_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] * train_x[:,ind_j]))

    //feature_pair_plus_list = get_feature_pair_divide_list(train_x, train_y, 0.01)
    for i in range(len(feature_divide_pair_list)):
        ind_i = feature_divide_pair_list[i][0]
        ind_j = feature_divide_pair_list[i][1]
        sub_train_x = np.column_stack((sub_train_x, train_x[:,ind_i] / train_x[:,ind_j]))
        
    return sub_train_x

//predict
def gbc_classify(train_x, train_y):
    sub_x_Train = get_data(train_x, train_y)
    labels = np.zeros(len(train_y))
    labels[train_y>0] = 1
    gbc = GradientBoostingClassifier(n_estimators=3000, max_depth=8)
    gbc.fit(sub_x_Train, labels)
    return gbc

def gbc_svr_predict(gbc, train_x, train_y, test_x, feature_pair_sub_list, feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list):
    feature_indexs = getTopFeatures(train_x, train_y)
    sub_x_Train = get_data(train_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20], feature_pair_sub_mul_list[:20])
    sub_x_Test = get_data(test_x, feature_indexs[:16], feature_pair_sub_list
                ,feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list[:20], feature_pair_sub_mul_list[:20])
    pred_labels = gbc.predict(sub_x_Test)
    
    pred_probs = gbc.predict_proba(sub_x_Test)[:,1]
    
    ind_test = np.where(pred_probs>0.55)[0]
    
    ind_train = np.where(train_y > 0)[0]
    
    preds_all = np.zeros([len(sub_x_Test)])
    
    flag = (sub_x_Test[:,16] >= 1) 
    ind_tmp0 = np.where(flag)[0]
    
    ind_tmp = np.where(~flag)[0]
    
    sub_x_Train = get_data(train_x, feature_indexs[:100], feature_pair_sub_list, feature_pair_plus_list[:100], feature_pair_mul_list[:40], feature_pair_divide_list)
    sub_x_Test = get_data(test_x, feature_indexs[:100], feature_pair_sub_list, feature_pair_plus_list[:100], feature_pair_mul_list[:40], feature_pair_divide_list)
    sub_x_Train[:,101] = np.log(1-sub_x_Train[:,101])
    sub_x_Test[ind_tmp,101] = np.log(1-sub_x_Test[ind_tmp,101])
    scaler = pp.StandardScaler()
    scaler.fit(sub_x_Train)
    sub_x_Train = scaler.transform(sub_x_Train)
    sub_x_Test[ind_tmp] = scaler.transform(sub_x_Test[ind_tmp]) 
    
    svr = SVR(C=16, kernel='rbf', gamma =  0.000122)
    
    svr.fit(sub_x_Train[ind_train], np.log(train_y[ind_train]))
    preds = svr.predict(sub_x_Test[ind_test])
    preds_all[ind_test] = np.power(np.e, preds)
    preds_all[ind_tmp0] = 0
    return preds_all

if __name__ == '__main__':
    train_x = np.loadtxt(open("train_x.csv","rb"), delimiter=",", skiprows=0)
    train_y = np.loadtxt(open("train_y.csv","rb"), delimiter=",", skiprows=0)
    test_x = np.loadtxt(open("test.csv","rb"), delimiter=",", skiprows=0)

    feature_pair_sub_list = get_feature_pair_sub_list(train_x, train_y, 0.01)
    feature_pair_plus_list = get_feature_pair_plus_list(train_x, train_y, 0.01)
    feature_pair_plus_list = get_feature_pair_mul_list(train_x, train_y, 0.01)
    feature_pair_plus_list = get_feature_pair_divide_list(train_x, train_y, 0.01)

    gbc = gbc_classify(train_x, train_y)
    svr_preds = gbc_svr_predict(gbc, train_x, train_y, test_x, feature_pair_sub_list, feature_pair_plus_list, feature_pair_mul_list, feature_pair_divide_list)

    output_preds(svr_preds)
    
