# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:22:40 2023

@author: l1415
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statistics import mean
from sklearn.metrics import accuracy_score as acc
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve, auc
from sklearn import metrics



#网格搜索
def build_model(X, y, seed=1):
    model = RandomForestClassifier()#可以选择不同的模型
    X_all_train,X_all_test,y_all_train,y_all_test = train_test_split(X, y,test_size=0.2,random_state=1)#数据随机划分
    param_grid = {
        #"learning_rate": np.linspace(0.001, 0.1, 5),
        #"loss": ['deviance', 'exponential'],
        "max_depth": np.arange(1, 11, step=1),
        "n_estimators": np.arange(50, 301, step=50),
    }
    #metrics.get_scorer_names() #查看有哪些模型性能指标
    random_cv = GridSearchCV(
        model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    random_cv.fit(X_all_train, y_all_train)#在测试集中做5折交叉验证网格搜索超参数
    return random_cv


# 通过五折交叉验证绘制roc曲线
def roc(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 8))
    i = 0
    for train, test in kf.split(X):
        X_all_train, X_all_test = X.iloc[train, :], X.iloc[test, :]
        y_all_train, y_all_test = y.iloc[train], y.iloc[test]
        model = build_model(X_all_train, y_all_train)
        probas_ = model.predict(X_all_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_all[test], probas_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             alpha=.8,label='Chance')
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 Standard deviation')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Phase-classfication example')
    plt.legend(loc="lower right")
    return plt.show()

if __name__ == '__main__':
    #读取原始数据， 第一列为目标性能，后面为特征
    phase_data = pd.read_csv("./Data2.csv")
    y = phase_data.iloc[:, 2]
    X = phase_data.iloc[:, 3:]#特征选择之后可以重新选择最佳特征组合
    #X_all = MinMax(X_all)
    roc(X, y)
