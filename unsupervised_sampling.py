# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:15:48 2023

@author: l1415
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from classfication import *
from regression import *
from sklearn.metrics import r2_score as r2
import sklearn.cluster as sc
from sklearn.manifold import TSNE

def new_alloys():
    data = pd.DataFrame(data=None,columns=["Al", "Co", "B", "Cr","Fe", "Ni", "Mn", "V", "Ti",
                                           "Mo", "Nb", "Zr", "Ta", "Hf","W", "Ru", "Re", "Ir", "C"])
    i = 0
    for al in np.linspace(7, 10, 4):
        for w in np.linspace(7, 10, 4):
            for ta in np.linspace(1, 3, 3):
                for ti in np.linspace(1, 3, 3):
                    for mo in np.linspace(1, 3, 3):
                         for nb in np.linspace(1, 3, 3):
                            for ru in np.linspace(1, 3, 3):
                                for re in np.linspace(1, 3, 3):
                                    for ir in np.linspace(1, 3, 3):
                                        one_col = []
                                        one_col.append(al) # Al
                                        one_col.append(100-al-w-ti-mo-nb-ta-ru-re-ir) # Co
                                        one_col.append(0) # B
                                        one_col.append(0) # Cr
                                        one_col.append(0) # Fe
                                        one_col.append(0) # Ni
                                        one_col.append(0) # Mn
                                        one_col.append(0) # V
                                        one_col.append(ti) # Ti
                                        one_col.append(mo) # Mo
                                        one_col.append(nb) # Nb
                                        one_col.append(0) # Zr
                                        one_col.append(ta) # Ta
                                        one_col.append(0) # Hf
                                        one_col.append(w) # W
                                        one_col.append(ru) # Ru
                                        one_col.append(re) # Re
                                        one_col.append(ir) # Ir
                                        one_col.append(0) # C
                                        data.loc[i] = one_col
                                        i = i+1
    data = data/100
    return data

def feature_mean(Alloy_composition, feature="MV"):
    Pa = Alloy_composition
    mean = 0
    for i in range(19):
        mean = mean + Pa.iloc[:, i]*ele_feature[feature][i]
    return mean
def feature_dev(Alloy_composition, mean, feature="MV"):
    Pa = Alloy_composition
    dev = 0
    for i in range(19):
        dev = dev + Pa.iloc[:, i]*np.square(1-ele_feature[feature][i]/mean)
    dev = np.sqrt(dev)
    return dev
def dev_formula(Alloy_composition, feature="MV"):
    mean = feature_mean(Alloy_composition, feature)
    d_feature = feature_dev(Alloy_composition, mean, feature)
    return d_feature

def dHm_formula(Alloy_composition):
    Pa = Alloy_composition
    dHm = Pa[Hm.iloc[0, 0]] * Pa[Hm.iloc[0, 1]] * Hm.iloc[0, 2]
    for i in range(1, Hm.shape[0]):
        dHm = dHm + Pa[Hm.iloc[i, 0]] * Pa[Hm.iloc[i, 1]] * Hm.iloc[i, 2]
    dHm = 4*dHm
    return dHm

new_Co_Al_W_data = pd.read_csv('./new_Co_Al_W_data.csv')

Hm = pd.read_csv("Hm_for_elements.csv")
ele_feature = pd.read_csv("./Features_of_Elements.csv")
ele_feature = ele_feature.fillna(0)

dMV = dev_formula(new_Co_Al_W_data, 'MV')
dDC = dev_formula(new_Co_Al_W_data, 'DC')
dCS = dev_formula(new_Co_Al_W_data, 'CS')
VEC = feature_mean(new_Co_Al_W_data, 'VEC')
YM = feature_mean(new_Co_Al_W_data, 'YM')
PR = feature_mean(new_Co_Al_W_data, 'PR')
dHm = dHm_formula(new_Co_Al_W_data)

new_Co_Al_W_re_features = pd.concat([dMV, dHm, dDC, dCS], axis=1, join='outer')
new_Co_Al_W_re_features.columns = ["δMV", "ΔHm", "δDC", "δCS"]

new_Co_Al_W_cl_features = pd.concat([VEC, YM, PR], axis=1, join='outer')
new_Co_Al_W_cl_features.columns = ["VEC", "YM", "PR"]


data_re_pd = pd.read_csv("./Data1.csv")
X_re = data_re_pd.iloc[:, 3:]
y_re = data_re_pd.iloc[:, 2]# TSH_RT, YS_RT, TSN_RT, TSH_HT, YS_HT, TSN_HT
seed = np.random.randint(0,100)
print(seed)
model_re = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=1)
X_re_train,X_re_test,y_re_train,y_re_test = train_test_split(X_re,y_re,test_size=0.2, random_state=5)
model_re.fit(X_re_train,y_re_train)
predict_new_re = model_re.predict(new_Co_Al_W_re_features)
print(predict_new_re.max(), predict_new_re.min())

data_cl_pd = pd.read_csv("./Data2.csv")
X_cl = data_cl_pd.iloc[:, 3:]
y_cl = data_cl_pd.iloc[:, 2]# TSH_RT, YS_RT, TSN_RT, TSH_HT, YS_HT, TSN_HT
seed = np.random.randint(0,100)
print(seed)
model_cl = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1)
#model_cl = GradientBoostingClassifier()
X_cl_train,X_cl_test,y_cl_train,y_cl_test = train_test_split(X_cl,y_cl,test_size=0.2, random_state=5)
model_cl.fit(X_cl,y_cl)

predict_new_re = model_re.predict(new_Co_Al_W_re_features)
predict_new_cl = model_cl.predict(new_Co_Al_W_cl_features)

predict_new_re = pd.DataFrame(predict_new_re)
predict_new_cl = pd.DataFrame(predict_new_cl)

new_Co_Al_W_re_features = pd.concat([predict_new_cl, predict_new_re, new_Co_Al_W_re_features], axis=1, join='outer')
new_Co_Al_W_re_features.columns = ["cl", "re", "δMV", "ΔHm", "δDC", "δCS"]

Co_Al_W_data = pd.read_csv('./Data1.csv').iloc[0:83, :]
Co_Al_W_feature = Co_Al_W_data.iloc[:, 3:]


all_Co_Al_W_features = pd.concat([new_Co_Al_W_re_features.iloc[:, 2:], Co_Al_W_feature], axis=0, join='outer')
all_Co_Al_W_features = all_Co_Al_W_features.reset_index(drop=True)

tsne = TSNE(perplexity=30, early_exaggeration=12, n_components=2, random_state=1)
all_Co_Al_W_tene = tsne.fit_transform(all_Co_Al_W_features)

Co_Al_W_tene = pd.DataFrame(all_Co_Al_W_tene[new_Co_Al_W_re_features.shape[0]:, ])
Co_Al_W_tene.columns = [ "TSNE_1", "TSNE_2"]

new_Co_Al_W_tene = pd.DataFrame(all_Co_Al_W_tene[0:new_Co_Al_W_re_features.shape[0], ])
new_Co_Al_W_tene.columns = [ "TSNE_1", "TSNE_2"]

new_Co_Al_W_tene = pd.concat([predict_new_cl, predict_new_re, new_Co_Al_W_tene], axis=1, join='outer')
new_Co_Al_W_tene.columns = ["cl", "es", "TSNE_1", "TSNE_2"]

new_Co_Al_W_tene = new_Co_Al_W_tene[new_Co_Al_W_tene['cl'].isin(['Y'])]

plt.figure()#dpi=300
col_1 = new_Co_Al_W_tene["es"]
plt.scatter(new_Co_Al_W_tene["TSNE_1"], new_Co_Al_W_tene["TSNE_2"], c=col_1, cmap="rainbow", s=1, vmin=950, vmax=1250)#, vmin=1230, vmax=1270
plt.colorbar()
col_2 = Co_Al_W_data.iloc[:, 2]
plt.scatter(Co_Al_W_tene["TSNE_1"],Co_Al_W_tene["TSNE_2"], c=col_2, cmap="rainbow", marker='^',
            edgecolors='black', s=50, vmin=950, vmax=1250)
plt.xlabel("TSNE_1")
plt.ylabel("TSNE_2")
plt.title("Co-Al-W Superalloys")#\
plt.show()


model_km = sc.KMeans(n_clusters=10, algorithm="elkan", random_state=10)
yhat = model_km.fit_predict(new_Co_Al_W_tene.iloc[:, 2:4])
test_good = np.array(new_Co_Al_W_tene.iloc[:, 2:4])

def KMean_plot(Data, yhat):
    clusters = np.unique(yhat)
    for cluster in clusters:
        row_ix = np.where(yhat == cluster)
        plt.scatter(Data[row_ix, 0], Data[row_ix, 1], label = cluster)#sort_test["dHm"], sort_test["dMV"]
    plt.legend(loc="center right")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("KMeans\nCo-Al-W-base Superalloys")
    return plt.show()
KMean_plot(test_good, yhat)

test_good_2 = test_good[np.where(yhat == 8), :].squeeze()
model_km = sc.KMeans(n_clusters=10, algorithm="elkan", random_state=0)
yhat = model_km.fit_predict(test_good_2)
KMean_plot(test_good_2, yhat)

test_good_3 = test_good_2[np.where(yhat == 7), :].squeeze()
model_km = sc.KMeans(n_clusters=4, algorithm="elkan", random_state=0)
yhat = model_km.fit_predict(test_good_3)
KMean_plot(test_good_3, yhat)

test_good_4 = test_good_3[np.where(yhat == 1), :].squeeze()
plt.figure()
plt.scatter(test_good_4[:, 0], test_good_4[:, 1])
plt.show()

data_candidate_index = pd.DataFrame(test_good_4)
data_candidate_index.columns = ["TSNE_1", "TSNE_2"]

model_km = sc.KMeans(n_clusters=10, algorithm="elkan", random_state=10)
yhat = model_km.fit_predict(new_Co_Al_W_tene.iloc[:, 2:4])
test_bad = np.array(new_Co_Al_W_tene.iloc[:, 2:4])
KMean_plot(test_bad, yhat)

test_bad_2 = test_bad[np.where(yhat == 4), :].squeeze()
model_km = sc.KMeans(n_clusters=10, algorithm="elkan", random_state=0)
yhat = model_km.fit_predict(test_bad_2)
KMean_plot(test_bad_2, yhat)

test_bad_3 = test_bad_2[np.where(yhat == 9), :].squeeze()
model_km = sc.KMeans(n_clusters=2, algorithm="elkan", random_state=1)
yhat = model_km.fit_predict(test_bad_3)
KMean_plot(test_bad_3, yhat)

test_bad_4 = test_bad_3[np.where(yhat == 0), :].squeeze()
plt.figure()
plt.scatter(test_bad_4[:, 0], test_bad_4[:, 1])
plt.show()

data_adjacent_index = pd.DataFrame(test_bad_4)
data_adjacent_index.columns = ["TSNE_1", "TSNE_2"]

new_Co_Al_W_data_tene = pd.concat([new_Co_Al_W_tene, new_Co_Al_W_data], axis=1, join='outer')
new_Co_Al_W_data_tene = new_Co_Al_W_data_tene[new_Co_Al_W_data_tene['cl'].isin(['Y'])]

TSNE_1 = data_candidate_index["TSNE_1"]
TSNE_2 = data_candidate_index["TSNE_2"]
index_data_1 = pd.merge(new_Co_Al_W_data_tene, TSNE_1, how='inner', on='TSNE_1')
index_data_2 = pd.merge(index_data_1, TSNE_2, how='inner', on='TSNE_2')
new_Co_Al_W_candidate = index_data_2

TSNE_1 = data_adjacent_index["TSNE_1"]
TSNE_2 = data_adjacent_index["TSNE_2"]
index_data_1 = pd.merge(new_Co_Al_W_data_tene, TSNE_1, how='inner', on='TSNE_1')
index_data_2 = pd.merge(index_data_1, TSNE_2, how='inner', on='TSNE_2')
new_Co_Al_W_adjacent = index_data_2





