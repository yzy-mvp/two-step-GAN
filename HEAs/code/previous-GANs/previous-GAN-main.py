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


# Inverse transform the generated data back to the original scale based on the real data
def Inverse_transform(X,X_gen):
    
    max_scaler = MaxAbsScaler()
    max_fit = max_scaler.fit(X)
    X_gen = max_fit.inverse_transform(X_gen)
    
    return X_gen  

# Bayesian tuning hyperparameters
def SVR_opt(X_true,y_true, X_gen=None, y_gen=None, use_gendata = True):
     # objection function    
    def clf_validation(epsilon,C,gamma): 
        kf = KFold(n_splits=10, random_state= 2*10+55,shuffle=True)
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
                # Add the generated data to the training set of each fold to adjust the parameters
                X_train = np.concatenate((X_train,X_gen),axis=0)
                y_train = np.concatenate((y_train,y_gen),axis=0)
            
            X_test = X_true[test_index_[i]]
            y_test = y_true[test_index_[i]]
            
            SVR_ = SVR(epsilon=epsilon, C=C, gamma=gamma).fit(X_train, y_train)
            y_pred = SVR_.predict(X_test)
            validation_error.append(mean_squared_error(y_test,y_pred,squared=False))
          
        return -np.mean(validation_error)
    
    # the boundaries of each parameter
    pbounds = {'C': (1, 8000), 'epsilon': (1, 30), 'gamma': (0.0001, 2)}  
    optimizer = BayesianOptimization(f=clf_validation, pbounds=pbounds, verbose=0,random_state = 1)  
    optimizer.maximize(init_points=20, n_iter=50)  #
    #print(optimizer.max)
    # return the best model after tuning
    svr = SVR(epsilon=optimizer.max['params']['epsilon'], gamma=optimizer.max['params']['gamma'],
                      C=optimizer.max['params']['C'])
    
    return svr

def random_search(seed, gen_data_path, data_train, X_train_fea, y_train, X_test, y_test, gen_number):
    RMSE2_list = []
    
    for i in range(10):
        data_train_scaled = MaxAbsScaler().fit_transform(data_train)
        gan = GAN()
        gan.train(epochs=700,X_data=data_train_scaled, batch_size=gen_number, save_interval=50)
        gen_data = gan.gen_data
        data_gen = Inverse_transform(data_train,gen_data)
        # divide generated data into features and labels
        X_gen = data_gen[:,[2,4,7,9]]
        y_gen = data_gen[:,-1]

        #add the generated data to the training set in preparation for retraining
        x_all_train = np.concatenate((X_train_fea, X_gen),axis=0)
        y_all_train = np.concatenate((y_train, y_gen),axis=0)

        X_test_scaled2 = X_test[:,[2,4,7,9]]
        scaler2 = StandardScaler().fit(x_all_train)
        X_all_train = scaler2.transform(x_all_train)
        X_test_scaled2 = scaler2.transform(X_test_scaled2)
        X_fea_ = scaler2.transform(X_train_fea)
        X_gen_fea = scaler2.transform(X_gen)
        svr_all = SVR_opt(X_fea_, y_train,X_gen_fea,y_gen)
        svr_all.fit(X_all_train,y_all_train)
        y_pred2 = svr_all.predict(X_test_scaled2)
        rmse2 = mean_squared_error(y_test,y_pred2,squared=False)
        RMSE2_list.append(rmse2)
        
        # save the generated data
        #gen_data = np.concatenate((X_gen, y_gen.reshape(-1, 1)), axis=1)
        #gen_data = pd.DataFrame(gen_data)
        #gen_data.to_csv(os.path.join(gen_data_path, '%s_%d_%d.csv' % (seed, int(gen_number / 2), i)))

    return [int(gen_number/2), min(RMSE2_list)]

if __name__ == '__main__':

    #import dataset
    df = pd.read_excel(r'../../datasets/hea_features.xlsx')
    X_true = np.array(df.iloc[:,:-1])
    y = np.array(df.iloc[:,-1])
    test_size = 0.1

    gen_data_path = r'.\gen_data_file'
   
    # half batch =[100, 200, 300, 400, 500]
    num_list = [200,400,600,800,1000]
    rmse1_list =[]
    best_number_list = []
    best_rmse2_list = []
    all_rmse2_list = []
    seed_list = list(range(0,46,5))
    
    for seed in seed_list:
        # divide the dataset based on the current random seed
        X_train, X_test,y_train, y_test = train_test_split(X_true,y, test_size=test_size, random_state=seed)
        # evaluate the model without generating data on the current dataset partition
        X_train_fea = X_train[:,[2,4,7,9]]
        scaler1 = StandardScaler().fit(X_train_fea)
        X_fea = scaler1.transform(X_train_fea)
        X_test_scaled1 = X_test[:,[2,4,7,9]]
        X_test_scaled1 = scaler1.transform(X_test_scaled1)
        
        svr1 = SVR_opt(X_fea,y_train, use_gendata=False)
        svr1.fit(X_fea,y_train)
        y_pred1 = svr1.predict(X_test_scaled1)
        rmse1 = mean_squared_error(y_test, y_pred1,squared=False)
        rmse1_list.append(rmse1)
        
        # parallel computing on finding different numbers of generated data for each fold of random division  
        data_train = np.concatenate((X_train,y_train.reshape(-1,1)),axis=1)
        pool = multiprocessing.Pool(5)
        results = pool.map(partial(random_search, seed,gen_data_path, data_train, X_train_fea, y_train, X_test, y_test),num_list)
        pool.close()
        pool.join()
        
        #integrate the model prediction results of different numbers of generation under current division and compare with rmse1
        gen_num = [x[0] for x in results]
        rmse2_list = list(map(lambda x: x[1], results))
        all_rmse2_list.append(rmse2_list)
        
        
        # find the optimal number of generations under each division
        ind = rmse2_list.index(min(rmse2_list))
        best_number_list.append(gen_num[ind])
        best_rmse2_list.append(min(rmse2_list))
        
        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in rmse2_list:
            if single_rmse2 < rmse1:
                vaild_rmse2.append(single_rmse2)
                index = rmse2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])

    # integrate the RMSE2 of ten random divisions under each number
    mean_rmse1 = np.mean(rmse1_list)
    num_rmse2_dict = OrderedDict()
    print(rmse1_list)
    print('\n')
    print('The average of 10 divisions of rmse1 is:', mean_rmse1)
    for i in range(len(num_list)):
        mean_rmse2 = list(map(lambda x: x[i], all_rmse2_list))
        num_rmse2_dict[i] = mean_rmse2
        print('%d generate the average rmse of 10 divisions under the number: %f' % (num_list[i] / 2, np.mean(mean_rmse2)))