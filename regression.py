# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:15:10 2023

@author: l1415
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import svm

import matplotlib
matplotlib.rcParams['font.family']='Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size']=12
matplotlib.rcParams['mathtext.fontset'] ='custom'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2
from sklearn.preprocessing import StandardScaler

def y_inverse_transform(y_list, std_scalery):
    #y_orignal = np.exp(std_scalery.inverse_transform(y_list.reshape(-1,1)))
    y_orignal = std_scalery.inverse_transform(y_list.reshape(-1,1))
    return y_orignal

def data_preprocessing(X, y, seed=1):
    std_scalerXc = StandardScaler()
    std_scaleryc = StandardScaler()
    std_scalerX = [std_scalerXc]
    std_scalery = [std_scaleryc]

    Xc = np.array(X.values)
    yc = np.array(y.values)
    #yc = np.log(yc.astype('float'))
    Xc_norm =  std_scalerXc.fit_transform(Xc)
    yc_norm =  std_scaleryc.fit_transform(yc.reshape(-1,1))
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_norm, np.ravel(yc_norm), test_size = 0.2, random_state=seed)
    return Xc_train, Xc_test, yc_train, yc_test, std_scalery

from sklearn.ensemble import RandomForestRegressor
from  sklearn.model_selection import GridSearchCV

def build_model_re(X_train, X_test, y_train, y_test, model='RF', seed=1):
    if model == 'RF':
        model = RandomForestRegressor()#,criterion = 'mae')n_estimators=100, max_depth = 5, min_samples_split = 3

        param_grid = {
            # "learning_rate": np.linspace(0.001, 0.1, 5),
            # "loss": ['deviance', 'exponential'],
            "max_depth": np.arange(1, 11, step=1),
            "n_estimators": np.arange(10, 51, step=5),
        }
    elif model == 'SVR':
        model = svm.SVR()
        param_grid = {
            "C": np.arange(1, 100, step=1),
            "kernel": ['rbf', 'sigmoid'],#, 'precomputed'
            "gamma": np.arange(0.01, 10, step=0.05),
            # 'epsilon': np.arange(0.01, 10, step=0.02),
            # "degree": np.arange(1, 30, step=1),
        }
    elif model == 'GBM':
        model = GradientBoostingRegressor()
        param_grid = {
            'n_estimators': np.arange(50, 301, step=50),
            'max_depth': np.arange(1, 11, step=1),
            #'min_samples_split': np.arange(2, 11, step=1),
            #'learning_rate': [0.1, 0.75, 0.05,0.04, 0.03,0.02,0.01, 0.05]
        }
    elif model == 'NN':
        model = MLPRegressor()
        param_grid = {
            'hidden_layer_sizes': [(32,64,32), (64,64,64), (32,32,32)],
            'activation': ['relu','tanh','logistic'],
            'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'learning_rate': ['constant','adaptive'],
            'solver': ['adam'],
            'max_iter':[1000, 2000]
        }
    elif model == 'LR':
        model = LinearRegression()
        param_grid = {
            'fit_intercept': ['True','False']
        }
    elif model == 'KNN':
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': np.arange(5, 11, step=1)
        }

    #metrics.get_scorer_names() 
    random_cv = GridSearchCV(
        model, param_grid, cv=5, scoring="r2", n_jobs=-1
    )
    random_cv.fit(X_train, y_train)

    #regressor = RandomForestRegressor(**random_cv.best_params_)
    #regressor.fit(X_train, y_train)

    y_hat_train = random_cv.predict(X_train)  # Training set predictions
    y_hat_test = random_cv.predict(X_test)  # Test set predictions

    R2_train = r2(y_hat_train, y_train)
    R2_test = r2(y_hat_test, y_test)
    #print(R2_train, R2_test)
    return random_cv, y_hat_train, y_hat_test, R2_train, R2_test

def pearson_select(data_features):
    #pearson特征筛选
    cor_features = data.iloc[:, 1:].corr(method="pearson")#计算所有特征的pearson相关性
    imp_feature = pd.DataFrame(model_best.feature_importances_)[0]#特征重要性，与特征相关性的顺序一致

    best_feature = []
    feature_importance = []
    while cor_features.shape[1] > 0:
        a = list(np.abs(cor_features.iloc[0, :]) >= 0.8)# 特征之间相关性绝对值大于0.8的选择与目标性能更重要的那个特征
        b = []
        c = []
        for j in range(0, cor_features.shape[1]):
            if a[j]:
                b.append(j)
                c.append(imp_feature.iloc[j])
        best_idex = b[np.argmax(c)]
        best_feature.append(cor_features.columns[best_idex])
        feature_importance.append(np.max(c))
        cor_features.drop(cor_features.iloc[:, b], axis = 1, inplace=True)
        cor_features.drop(cor_features.index[b], axis = 0, inplace=True)
        imp_feature.drop(imp_feature.index[b], axis = 0, inplace=True)
        selection_features = data[best_feature]
        selection_features = pd.concat([data["phase"], selection_features], axis=1, join='outer')
    return selection_features

# 向前特征选择
def forward_features_selection(selection_features):
    R2_final = []
    y_all = selection_features.iloc[:, 0]
    X_all = selection_features.iloc[:, 1:]
    for i in range(0, selection_features.shape[1]-1):
        R2_train_mean = []
        R2_test_mean = []
        features = []
        if i == 0:
            for j in range(0, selection_features.shape[1]-1-i):
                R2_train_list = []
                R2_test_list = []

                for train_index, test_index in kf.split(X_all):
                    X_all_feature = X_all.iloc[:, j:j+1]
                    X_all_train, X_all_test = X_all_feature.iloc[train_index, :], X_all_feature.iloc[test_index, :]
                    y_all_train, y_all_test = y_all.iloc[train_index], y_all.iloc[test_index]
                    
                    _, _, _, R2_train, R2_test= build_model_re(X_all_train, X_all_test, y_all_train, y_all_test)#网格搜索
                    
                    R2_train_list.append(R2_train)
                    R2_test_list.append(R2_test)
                    
                features.append(X_all_feature.columns[:])
                
                R2_train_mean.append(mean(R2_train_list))
                R2_test_mean.append(mean(R2_test_list))
                
            #print(r2_train_mean, r2_test_mean)
            result_R2 = list(zip(features, R2_train_mean, R2_test_mean))
            R2_final.append(result_R2)
            print(result_R2)
            best_idex = np.argmax(R2_test_mean)
            
        else:
            if i == 1:
                best_features = X_all.iloc[:, best_idex]
            else:
                best_features = pd.concat([best_features, X_all.iloc[:, best_idex]], axis=1, join='outer')
            X_all.drop(X_all.columns[best_idex], axis = 1, inplace=True)
            
            R2_train_mean = []
            R2_test_mean = []

            features = []
            for j in range(0, selection_features.shape[1]-1-i):
                R2_train_list = []
                R2_test_list = []
                for train_index, test_index in kf.split(X_all):
                    X_all_feature = pd.concat([best_features, X_all.iloc[:, j:j+1]], axis=1, join='outer')
                    X_all_train, X_all_test = X_all_feature.iloc[train_index, :], X_all_feature.iloc[test_index, :]
                    y_all_train, y_all_test = y_all.iloc[train_index], y_all.iloc[test_index]
                    
                    _, _, _, R2_train, R2_test= build_model_re(X_all_train, X_all_test, y_all_train, y_all_test)#网格搜索
                    
                    R2_train_list.append(R2_train)
                    R2_test_list.append(R2_test)
                    
                    R2_train_list.append(R2_train)
                    R2_test_list.append(R2_test)

                    
                features.append(X_all_feature.columns[:])
                
                R2_train_mean.append(mean(R2_train_list))
                R2_test_mean.append(mean(R2_test_list))
                
            #print(r2_train_mean, r2_test_mean)
            result_R2 = list(zip(features, R2_train_mean, R2_test_mean))
            R2_final.append(result_R2)
            print(result_R2)
            best_idex = np.argmax(R2_test_mean)
            best_features = selection_features[features[0][0:4]]#选择最佳的前几个特征
            best_features = pd.concat([phase_data["phase"], best_features], axis=1, join='outer')
    return best_features

def plot(y_hat_train, y_hat_test, y_train, y_test, std_scalery, title):
    y_hat_train = y_inverse_transform(y_hat_train, std_scalery[0])
    y_hat_test = y_inverse_transform(y_hat_test, std_scalery[0])

    y_train = y_inverse_transform(y_train, std_scalery[0])
    y_test = y_inverse_transform(y_test, std_scalery[0])

    matplotlib.rcParams['font.size']=20
    x = [np.min(y_train), np.max(y_train)]
    y = [np.min(y_train), np.max(y_train)]

    R2_train = r2(y_hat_train, y_train)
    R2_test = r2(y_hat_test, y_test)
    print(R2_train, R2_test)
    # x = [800, 1300]
    # y = [800, 1300]
    #label = [700, 800, 900, 1000, 1100, 1200 ,1300, 1400]
    plt.figure(figsize=(7, 6), dpi=100)#7.4
    plt.scatter(y_train, y_hat_train, label="Train (R$^2$={:.3f})".format(R2_train),
                c="#243F99", alpha=0.7,s=100)
    plt.scatter(y_test, y_hat_test, label="Test  (R$^2$={:.3f})".format(R2_test),
                c="#E7F6B1", s=100, alpha=0.9, edgecolors="black")
    plt.plot(x, y, c='red')
    plt.xlabel("Measured value")
    plt.ylabel("Predicted value")
    plt.legend(loc='best', frameon=False)
    plt.tick_params(length=7, width=2)
    plt.title(title)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    return plt.show()

if __name__ == '__main__':
    data_pd = pd.read_csv("./Data1.csv")
    X = data_pd.iloc[:, 3:]
    y = data_pd.iloc[:, 2]# TSH_RT, YS_RT, TSN_RT, TSH_HT, YS_HT, TSN_HT
    seed = np.random.randint(0,100)
    #print(seed)
    X_train, X_test, y_train, y_test, std_scalery = data_preprocessing(X, y, seed)
    best_model, y_hat_train, y_hat_test, R2_train, R2_test = build_model_re(X_train, X_test, y_train, y_test, model='RF')
    plot(y_hat_train, y_hat_test, y_train, y_test, std_scalery, title ="RF - $T_{\gamma _ {'}}$ ($^{\circ}$C)")



