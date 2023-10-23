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

import matplotlib
matplotlib.rcParams['font.family']='Arial'     #将字体设置为黑体'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.size']=15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


#网格搜索
def build_model_cl(X, y, seed=1):
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
    acc_final = []
    y_all = selection_features.iloc[:, 0]
    X_all = selection_features.iloc[:, 1:]
    for i in range(0, selection_features.shape[1]-1):
        acc_train_mean = []
        acc_test_mean = []
        features = []
        if i == 0:
            for j in range(0, selection_features.shape[1]-1-i):
                acc_train_list = []
                acc_test_list = []

                for train_index, test_index in kf.split(X_all):
                    X_all_feature = X_all.iloc[:, j:j+1]
                    X_all_train, X_all_test = X_all_feature.iloc[train_index, :], X_all_feature.iloc[test_index, :]
                    y_all_train, y_all_test = y_all.iloc[train_index], y_all.iloc[test_index]
                    
                    model_best = build_model_cl(X_all_train, y_all_train)#网格搜索
                    result_train = model_best.predict(X_all_train)
                    result_test = model_best.predict(X_all_test)
                    
                    acc_train = acc(y_all_train, result_train)
                    acc_test = acc(y_all_test, result_test)
                    
                    acc_train_list.append(acc_train)
                    acc_test_list.append(acc_test)

                    
                features.append(X_all_feature.columns[:])
                
                acc_train_mean.append(mean(acc_train_list))
                acc_test_mean.append(mean(acc_test_list))
                
            #print(r2_train_mean, r2_test_mean)
            result_acc = list(zip(features, acc_train_mean, acc_test_mean))
            acc_final.append(result_acc)
            print(result_acc)
            best_idex = np.argmax(acc_test_mean)
            
        else:
            if i == 1:
                best_features = X_all.iloc[:, best_idex]
            else:
                best_features = pd.concat([best_features, X_all.iloc[:, best_idex]], axis=1, join='outer')
            X_all.drop(X_all.columns[best_idex], axis = 1, inplace=True)
            
            acc_train_mean = []
            acc_test_mean = []

            features = []
            for j in range(0, selection_features.shape[1]-1-i):
                acc_train_list = []
                acc_test_list = []
                for train_index, test_index in kf.split(X_all):
                    X_all_feature = pd.concat([best_features, X_all.iloc[:, j:j+1]], axis=1, join='outer')
                    X_all_train, X_all_test = X_all_feature.iloc[train_index, :], X_all_feature.iloc[test_index, :]
                    y_all_train, y_all_test = y_all.iloc[train_index], y_all.iloc[test_index]
                    
                    model_best = build_model_cl(X_all_train, y_all_train)#网格搜索
                    result_train = model_best.predict(X_all_train)
                    result_test = model_best.predict(X_all_test)
                    
                    acc_train = acc(y_all_train, result_train)
                    acc_test = acc(y_all_test, result_test)
                    
                    acc_train_list.append(acc_train)
                    acc_test_list.append(acc_test)

                    
                features.append(X_all_feature.columns[:])
                
                acc_train_mean.append(mean(acc_train_list))
                acc_test_mean.append(mean(acc_test_list))
                
            #print(r2_train_mean, r2_test_mean)
            result_acc = list(zip(features, acc_train_mean, acc_test_mean))
            acc_final.append(result_acc)
            print(result_acc)
            best_idex = np.argmax(acc_test_mean)
            best_features = selection_features[features[0][0:4]]#选择最佳的前几个特征
            best_features = pd.concat([phase_data["phase"], best_features], axis=1, join='outer')
    return best_features

# 通过五折交叉验证绘制roc曲线
def roc(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(6.5, 5), dpi=300)
    i = 0
    for train, test in kf.split(X):
        X_all_train, X_all_test = X.iloc[train, :], X.iloc[test, :]
        y_all_train, y_all_test = y.iloc[train], y.iloc[test]
        model = build_model_cl(X_all_train, y_all_train)
        model.fit(X_all_train, y_all_train)
        probas_ = model.predict(X_all_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3)#,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc)
    
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='#243F99',
             label=r'Test (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 Standard deviation')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    #plt.title('Phase-classfication example')
    plt.legend(loc="lower right", frameon=False)
    return plt.show()

if __name__ == '__main__':
    #特征选择
    data_features = pd.read_csv("gama_features.csv")
    selection_features = pearson_select(data_features)
    best_features = forward_features_selection(selection_features)
    
    #读取原始数据， 第一列为目标性能，后面为特征
    phase_data = pd.read_csv("./Data2.csv")
    y = phase_data.iloc[:, 2]
    X = phase_data.iloc[:, 3:]#特征选择之后可以重新选择最佳特征组合
    #X_all = MinMax(X_all)
    roc(X, y)
