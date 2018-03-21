# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 16:31:36 2018

@author: Zongjian Fan
"""

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
#import features

# working directory
dir = '.'
label_index = 770

eps=0.01

def output_classify(preds):
    print('writing classify data...')
    out_file = dir + '/output_classify.csv'
    fs = open(out_file,'w')
    fs.write('id,loss\n')
    for i in range(len(preds)):
        if preds[i] > 0.55:
            preds[i] = 1
        else:
            preds[i] = 0
        strs = str(i+105472) + ',' + str(np.float(preds[i]))
        fs.write(strs + '\n');
    fs.close()
    return

def output_preds(preds):
    print('writing regression data...')
    out_file = dir + '/output_regression.csv'
    fs = open(out_file,'w')
    fs.write('id,loss\n')
    for i in range(len(preds)):
        if preds[i] > 100:
            preds[i] = 100
        elif preds[i] < 0:
            preds[i] = 0
        strs = str(i+105472) + ',' + str(np.float(preds[i]))
        fs.write(strs + '\n');
    fs.close()
    return

# load train data
def load_train_fs():
    train_x = np.genfromtxt(open(dir + '/FinalFeature.csv','rb'), delimiter=',')
    train_y = np.genfromtxt(open(dir + '/loss.csv','rb'), delimiter=',')
    col_mean = np.nanmean(train_x, axis=0)
    indsx = np.where(np.isnan(train_x))
    print(indsx)
    train_x[indsx] = np.take(col_mean, indsx[1])
    train_x[np.isinf(train_x)] = 0
    print('loading training data')
    print(train_x.shape, train_y.shape)
    return train_x, train_y


# load test data
def load_test_fs():
    test_fs = np.genfromtxt(open(dir + '/FinalTest.csv','rb'), delimiter=',')
    inds = np.where(np.isnan(test_fs))
    print(inds)
    col_mean = np.nanmean(test_fs, axis=0)
    test_fs[inds] = np.take(col_mean, inds[1])
    test_fs[np.isinf(test_fs)] = 0
    print('loading testing data')
    print(test_fs.shape)
    return test_fs


# transform the loss to the binary form
def toLabels(train_y):
    labels = np.zeros(len(train_y))
    labels[train_y>0] = 1
    return labels

# generate the output file based to the predictions

# use gbm classifier to predict whether the loan defaults or not
def gbc_classify(train_x, train_y):
    labels = toLabels(train_y)
    gbc = GradientBoostingClassifier(n_estimators=3000, max_depth=8)
    gbc.fit(train_x, labels)
    return gbc

def gbc_svr_predict( train_x, train_y, test_x):
    #feature_indexs = getTopFeatures(train_x, train_y)
#    flag = (test_x >= 1) 
#    ind_tmp0 = np.where(flag)[0]
#    
#    ind_tmp = np.where(~flag)[0]
#    train_x = np.log(1-train_x)
#    test_x[ind_tmp] = np.log(1-test_x[ind_tmp])
#    col_mean = np.nanmean(train_x, axis=0)
#    inds = np.where(np.isnan(train_x))
#    train_x[inds] = np.take(col_mean, inds[1])
#    train_x[np.isinf(train_x)] = 0
#    col_mean_test = np.nanmean(test_x, axis=0)
#    inds_test = np.where(np.isnan(test_x))
#    test_x[inds_test] = np.take(col_mean_test , inds_test[1])
#    test_x[np.isinf(test_x)] = 0
#    train_x.astype(np.float)
#    test_x.astype(np.float)
    scaler = pp.StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x) 
    
    svr = SVR(C=16, kernel='rbf', gamma =  0.000122)
    
    svr.fit(train_x, np.log(train_y))
    preds = svr.predict(test_x)
    preds_all = np.zeros([len(test_x)])
    preds_all = np.power(np.e, preds)
#    preds_all[ind_tmp0] = 0
    return preds_all

# use gbm regression to predict the loss, based on the result of gbm classifier
def gbc_gbr_predict(train_x, train_y, test_x):  
    
    scaler = pp.StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x) 
    
    gbr1000 = GradientBoostingRegressor(n_estimators=1300, max_depth=4, subsample=0.5, learning_rate=0.05)
    
    gbr1000.fit(train_x, np.log(train_y))
    preds = gbr1000.predict(test_x)
    preds_all = np.power(np.e, preds)
    return preds_all

# use gbm classifier to predict whether the loan defaults or not, then invoke svr and gbr regression
def predict(train_x, train_y, test_x):
    labels = toLabels(train_y)
    print('classifying')
#    pred_probs=np.genfromtxt(open(dir + '/classify.csv','rb'), delimiter=',')
#    print(pred_probs.shape)
    gbc = GradientBoostingClassifier(n_estimators=3000, max_depth=9)
    gbc.fit(train_x, labels)
    pred_probs = gbc.predict_proba(test_x)[:,1]
    output_classify(pred_probs)
    ind_train = np.where(labels>0.55)[0]
    ind_test = np.where(pred_probs>0.55)[0]
    print('gbr regression...')
    gbr_predict = gbc_gbr_predict(train_x[ind_train], train_y[ind_train], test_x[ind_test])
    gbr= np.zeros(len(test_x))
    gbr[ind_test] = gbr_predict
    np.savetxt("gbr.csv", gbr, delimiter=',') 
    print('svr regression...')
    svr_predict= gbc_svr_predict(train_x[ind_train], train_y[ind_train], test_x[ind_test])
    svr= np.zeros(len(test_x))
    svr[ind_test] = svr_predict
    np.savetxt("svr.csv", svr, delimiter=',') 
    return 0.6*gbr + 0.4*svr




# the main function
if __name__ == '__main__':
    test_x = load_test_fs()
    train_x, train_y = load_train_fs()
    preds= predict(train_x, train_y, test_x)
    output_preds(preds)    
