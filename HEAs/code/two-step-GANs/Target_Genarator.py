# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:34:40 2022

@author: swaggy.p
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error,r2_score
from sklearn.model_selection import KFold
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'


def Inverse_transform(X,X_gen):
    '''
    Parameters
    ----------
    X : TYPE: ndarray
        true data
    X_gen : ndarray
        generated data

    Returns
    -------
    X_gen : ndarray
        return generated data with same scale as true data

    '''
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)
    
    return X_gen


def svr_opt(X_fea,y):   
    def clf_validation(epsilon,C,gamma):
       
        kf = KFold(n_splits=10,random_state=75,shuffle=True)
        i = 0
        train_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        test_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        
        for train_index, test_index in kf.split(X_fea):
            train_index_[i] = train_index
            test_index_[i] = test_index
            i += 1
        validation_error = []  
        for i in range(10):
            
            X_train = X_fea[train_index_[i]]
            y_train = y[train_index_[i]]
            X_test = X_fea[test_index_[i]]
            y_test = y[test_index_[i]]
                 
            SVR_ = SVR(epsilon=epsilon, C=C, gamma=gamma).fit(X_train, y_train)
            y_pred = SVR_.predict(X_test)  
            validation_error.append(mean_squared_error(y_test,y_pred,squared=False))
   
        test_score = -np.mean(validation_error)  
        return test_score
    
    pbounds = {'C': (1000, 8000), 'epsilon': (1, 10), 'gamma': (0.1, 1)}  # 定义边界
    optimizer = BayesianOptimization(f=clf_validation, pbounds=pbounds, verbose=0,random_state = 1)  
    optimizer.maximize(init_points=20, n_iter=50)  # 优化
    #print(optimizer.max)
    svr = SVR(epsilon=optimizer.max['params']['epsilon'], gamma=optimizer.max['params']['gamma'],
                      C=optimizer.max['params']['C'])
    
    return svr


def Target_predict(X,y,gen_data):
   
    X_fea = X[:,[2,4,7,9]]
   
    X_gen = Inverse_transform(X, gen_data)
    X_gen_fea = X_gen[:,[2,4,7,9]]
    X_fea = StandardScaler().fit_transform(X_fea)
    X_gen_fea = StandardScaler().fit_transform(X_gen_fea)
    
    svr = svr_opt(X_fea, y)
    svr.fit(X_fea, y)
    y_gen = svr.predict(X_gen_fea)
   
    return X_gen_fea, y_gen

    
    