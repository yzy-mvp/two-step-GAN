# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:59:43 2022

@author: Swappy.p
"""
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from GAN import GAN
from bayes_opt import BayesianOptimization
import multiprocessing
from functools import partial
from collections import OrderedDict
import os


def Inverse_transform(X,X_gen):
    
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)
    return X_gen  

def SVR_opt(X_true,y_true, X_gen=None, y_gen=None, use_gendata = True):       
    def clf_validation(epsilon,C,gamma): 
        kf = KFold(n_splits=10,random_state= 2*10+55,shuffle=True)
        i = 0
        train_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        test_index_ = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
        
        for train_index, test_index in kf.split(X_true):
            train_index_[i] = train_index
            test_index_[i] = test_index
            i += 1
        validation_error = []
        for i in range(10):
            
            X_train = X_true[train_index_[i]]
            y_train = y_true[train_index_[i]]
            if use_gendata:
                
                X_train = np.concatenate((X_train,X_gen),axis=0)
                y_train = np.concatenate((y_train,y_gen),axis=0)
            
            X_test = X_true[test_index_[i]]
            y_test = y_true[test_index_[i]]
            
            SVR_ = SVR(epsilon=epsilon, C=C, gamma=gamma).fit(X_train, y_train)
            y_pred = SVR_.predict(X_test)
        
            validation_error.append(mean_squared_error(y_test,y_pred,squared=False))
       
        return -np.mean(validation_error)
    
    pbounds = {'C': (1, 8000), 'epsilon': (1, 30), 'gamma': (0.0001, 2)}  # 定义边界
    optimizer = BayesianOptimization(f=clf_validation, pbounds=pbounds, verbose=0,random_state = 1)  
    optimizer.maximize(init_points=20, n_iter=50)  # 优化
    #print(optimizer.max)
    svr = SVR(epsilon=optimizer.max['params']['epsilon'], gamma=optimizer.max['params']['gamma'],
                      C=optimizer.max['params']['C'])
    return svr

def random_search(data_train,X_test,y_test,gen_number):
    RMSE2_list = []
    for i in range(10):
        
        data_train_scaled = MaxAbsScaler().fit_transform(data_train)
        gan = GAN()
        gan.train(epochs=700,X_data=data_train_scaled, batch_size=gen_number, save_interval=50)
        gen_data = gan.gen_data
        data_gen = Inverse_transform(data_train,gen_data)
    
        X_gen = data_gen[:,[2,4,7,9]]
        y_gen = data_gen[:,-1]
        X_test_scaled2 = X_test[:,[2,4,7,9]]
        scaler2 = StandardScaler().fit(X_gen)

        X_test_scaled2 = scaler2.transform(X_test_scaled2)
        X_gen_fea = scaler2.transform(X_gen)
        svr_gen = SVR_opt(X_gen_fea,y_gen, use_gendata=False)
        
        ## train in generated data, instead of training in true data and generated data 
        svr_gen.fit(X_gen_fea,y_gen)
        y_pred2 = svr_gen.predict(X_test_scaled2)
        rmse2 = mean_squared_error(y_test,y_pred2,squared=False)
        RMSE2_list.append(rmse2)
    return [int(gen_number/2), min(RMSE2_list)]

if __name__ == '__main__':

    df = pd.read_excel(r'../../datasets/hea_features.xlsx')
    X_true = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    test_size = 0.1

    gen_data_path = r'.\gen_data_file'
    
    num_list = [200,400,600,800,1000]
    rmse1_list =[]
    best_number_list = []
    best_rmse2_list = []
    all_rmse2_list = []
    random_list = list(range(0,46,5))
    for random in random_list:
        X_train, X_test,y_train, y_test = train_test_split(X_true,y, test_size=test_size, random_state=random) 
        data_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        pool = multiprocessing.Pool(5)
        results = pool.map(partial(random_search,data_train,X_test,y_test),num_list)
        pool.close()
        pool.join()

        gen_num = [x[0] for x in results]
        rmse2_list = list(map(lambda x: x[1], results))
        all_rmse2_list.append(rmse2_list)
        ind = rmse2_list.index(min(rmse2_list))
        best_number_list.append(gen_num[ind])
        best_rmse2_list.append(min(rmse2_list))
        
    num_rmse2_dict = OrderedDict()
    for i in range(len(num_list)):
        mean_rmse2 = list(map(lambda x: x[i], all_rmse2_list))
        num_rmse2_dict[i] = mean_rmse2
        print('%d  generate the average rmse2 of 10 divisions under the number: %f' % (num_list[i] / 2, np.mean(mean_rmse2)))