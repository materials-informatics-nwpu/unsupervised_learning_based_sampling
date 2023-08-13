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

def build_model(X_train, X_test, y_train, y_test, model='RF', seed=1):
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
            'n_estimators': np.arange(10, 51, step=5),
            'max_depth': np.arange(1, 11, step=1),
            'min_samples_split': np.arange(2, 11, step=1),
            'learning_rate': [0.1, 0.75, 0.05,0.04, 0.03,0.02,0.01, 0.05]
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
    return y_hat_train, y_hat_test, R2_train, R2_test

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
    y_hat_train, y_hat_test, R2_train, R2_test = build_model(X_train, X_test, y_train, y_test, model='RF')
    plot(y_hat_train, y_hat_test, y_train, y_test, std_scalery, title ="RF - $T_{\gamma _ {'}}$ ($^{\circ}$C)")



