# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:59:55 2022

@author: C903
"""

import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from sklearn.model_selection import KFold
from collections import OrderedDict, defaultdict
from sklearn.preprocessing import MaxAbsScaler
import optuna
import multiprocessing
from functools import partial


###############################################################################################################################
## GAN0 to 9 respectively correspond to the dataset divided into 10 parts and use the GAN model corresponding to the first i parts
## Each time you run a GAN with a specific dataset size, you need to specify the corresponding dataset and GAN

## For example, when running the first i data sets, specify GAN as GANi, train_data_dict[i]

from GANEf0 import GAN
# from GANEf1 import GAN
# from GANEf2 import GAN
# from GANEf3 import GAN
# from GANEf4 import GAN
# from GANEf5 import GAN
# from GANEf6 import GAN
# from GANEf7 import GAN
# from GANEf8 import GAN
# from GANEf9 import GAN



###############################################################################################################################


def Optuna_opt(X,Y, gen_x=None, gen_y=None, use_gen_data=True):
    def objective(trial):
        rmse_list = []
        kf = KFold(n_splits=10,random_state=0,shuffle=True)
        for train_index, test_index in kf.split(X):
            train_x = X[train_index]
            train_y = Y[train_index]
            if use_gen_data:
                train_x = np.concatenate((train_x, gen_x),axis=0)
                train_y = np.concatenate((train_y, gen_y), axis=0)
            valid_x = X[test_index]
            valid_y = Y[test_index]           
            dtrain = lgb.Dataset(train_x, label=train_y)
            param = {
                'objective': 'regression',
                'metric': 'root_mean_squared_error',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                 "num_leaves": trial.suggest_int("num_leaves", 2, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
            }

            gbm = lgb.train(param,dtrain)
            preds = gbm.predict(valid_x)
            rmse = mean_squared_error(valid_y, preds,squared=False)
            rmse_list.append(rmse)
        return np.mean(rmse_list)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective,n_trials=200,show_progress_bar=False)
    trial = study.best_trial
    best_model = lgb.LGBMRegressor(**trial.params)
    #print(trial.value)
    return best_model


def Split_data(X_train, Y_train,seed, n_split=10):
    #Randomly divide the training set into 10 points, and take different scores to train GAN
    kf = KFold(n_splits=n_split,shuffle=True, random_state=seed)
    kf_index=[]
    for train_index, test_insex in kf.split(X_train):
        kf_index.append(test_insex)
    #Store a dictionary of training sets of different sizes, the key is the subset of the first few folds, 
    #and the value is the corresponding df data集，值是对应的df数据   
    train_data_dict = defaultdict(dict)
    train_data_indexs = defaultdict(dict)
    i = 0
    for i in range(n_split):
        if i == 0:
            train_data_indexs[i] = kf_index[i]
        else:
            train_data_indexs[i] = kf_index[i].tolist() +np.array(kf_index[:i]).reshape([-1,]).tolist()
    for i in range(n_split):
        x_data = X_train.iloc[train_data_indexs[i],:]
        y_data = Y_train.iloc[train_data_indexs[i]]
        train_data_dict[i] = [x_data,y_data]
        
    return train_data_dict


def Target_predict_lgb(model,X_train, y_train, gen_data):
    max_scaler = MaxAbsScaler().fit(X_train)
    X_gen = max_scaler.inverse_transform(gen_data)
    model.fit(X_train, y_train)
    y_gen = model.predict(X_gen)
    return X_gen,y_gen


def Generate_search(train_data_scaled,train_x,train_y, X_test, Y_test, lgb1, num):
    RMSE2_LIST = []
    for i in range(3):
        gan = GAN()
        gan.train(epochs=900,X_data=train_data_scaled,batch_size= num, save_interval=100)
        gen_data = gan.gen_data
       # assign labels to the generated data
        X_gen_fea,y_gen = Target_predict_lgb(lgb1, train_x, train_y, gen_data)
        
       #Add generated data to test the test set
        train_all_x = np.concatenate((train_x,X_gen_fea),axis=0)
        train_all_y = np.concatenate((train_y,y_gen), axis=0)
        #Adjust parameters based on the new training set
        lgb2 = Optuna_opt(train_x.values, train_y.values, X_gen_fea, y_gen)
        lgb2.fit(train_all_x, train_all_y)
        y_pred2 = lgb2.predict(X_test)
        rmse2 = mean_squared_error(Y_test, y_pred2, squared=False)
        RMSE2_LIST.append(rmse2)
        
    rmse2_min = min(RMSE2_LIST)
    return [int(num)/2, rmse2_min]

  
if __name__ == '__main__':
    data = pd.read_csv('../../datasets/final_elem+hard-feat_data.csv')

    X = data.iloc[:,1:-1]
    Y = data.iloc[:,-1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 595)
    
    best_number_list = []
    best_rmse2_list = []
    rmse1_list = []
    all_rmse2_list = []
    for seed in range(100,2101,500):
        train_data_dict = Split_data(X_train, Y_train,seed)
        
        train_x = train_data_dict[0][0]
        train_y = train_data_dict[0][1]
        
        lgb1 = Optuna_opt(train_x.values, train_y.values, use_gen_data=False)
        lgb1.fit(train_x, train_y)
        y_pred1 = lgb1.predict(X_test)
        rmse1 = mean_squared_error(Y_test, y_pred1,squared=False)
        rmse1_list.append(rmse1)
        
        num_list = [int(i*2) for i in (np.linspace(0.1,1.5,15) * len(train_data_dict[0][0]))]
        train_data_scaled = MaxAbsScaler().fit_transform(train_x)
        
        #Parallel computing to find the effect of different numbers of generated data
        pool = multiprocessing.Pool(5)
        results = pool.map(partial(Generate_search,train_data_scaled, train_x, train_y, X_test, Y_test, lgb1),num_list)
        pool.close()
        pool.join()
        
        #Integrate the model prediction results of different numbers of generation under this division and compare with rmse1
        gen_num = [x[0] for x in results]
        rmse2_list = list(map(lambda x: x[1], results))
        all_rmse2_list.append(rmse2_list)
        #Find the best generation number
        
        best_rmse2_list.append(min(rmse2_list))
        ind = rmse2_list.index(min(rmse2_list))
        best_number_list.append(gen_num[ind])     
        
        vaild_rmse2 = []
        vaild_number = []
        for single_rmse2 in rmse2_list:
            if single_rmse2 < rmse1:
                vaild_rmse2.append(single_rmse2)
                index = rmse2_list.index(single_rmse2)
                vaild_number.append(gen_num[index])
        
    mean_rmse1 = np.mean(rmse1_list)
    num_rmse2_dict = OrderedDict()
    print(rmse1_list)
    print('\n')
    print('The average of 5 divisions of rmse1:', mean_rmse1)
    for i in range(len(num_list)):
        mean_rmse2 = list(map(lambda x: x[i], all_rmse2_list)) 
        num_rmse2_dict[i] = mean_rmse2
        print('%dThe average rmse2 of the 5 divisions under the number of generation: %f'% (num_list[i]/2,np.mean(mean_rmse2)))