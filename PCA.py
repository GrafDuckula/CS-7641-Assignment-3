# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from helpers import get_data_from_csv

out = './PCA/'
cmap = cm.get_cmap('Spectral') 


np.random.seed(42)

# import wine quality data
wineX, wineY = get_data_from_csv("./BASE/wine_trg.csv", n_features=11, sep=',', header=None)
digitX, digitY = get_data_from_csv("./BASE/digit_trg.csv", n_features=256, sep=',', header=None)

wineX = StandardScaler().fit_transform(wineX)
digitX = StandardScaler().fit_transform(digitX)

clusters = [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]
dims_wine = [i for i in range(2,12)]

# data for 1

pca = PCA(random_state=5)
pca.fit(wineX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,12))
tmp.to_csv(out+'wine scree.csv')


pca = PCA(random_state=5)
pca.fit(digitX)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,257))
tmp.to_csv(out+'digit scree.csv')


# Data for 2

grid ={'pca__n_components':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitX,digitY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digit dim red.csv')

# data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 8
pca = PCA(n_components=dim,random_state=10)

wineX2 = pca.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY))))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine_datasets.csv',index=False,header=False)

dim = 55
pca = PCA(n_components=dim,random_state=10)
digitX2 = pca.fit_transform(digitX)
digit2 = pd.DataFrame(np.hstack((digitX2,np.atleast_2d(digitY))))
cols = list(range(digit2.shape[1]))
cols[-1] = 'Class'
digit2.columns = cols
digit2.to_csv(out+'digit_datasets.csv',index=False,header=False)
