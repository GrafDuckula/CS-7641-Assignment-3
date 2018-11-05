

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
from helpers import get_data_from_csv

out = './RP/'
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

tmp = defaultdict(dict)
for i,dim in product(range(10),dims_wine):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(wineX), wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(digitX), digitX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digit scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims_wine):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)
    tmp[dim][i] = reconstructionError(rp, wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(digitX)
    tmp[dim][i] = reconstructionError(rp, digitX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'digit scree2.csv')

# Data for 2

grid ={'rp__n_components':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitX,digitY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digit dim red.csv')

# data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 6
rp = SparseRandomProjection(n_components=dim,random_state=5)

wineX2 = rp.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY))))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine_datasets.csv',index=False,header=False)

dim = 60
rp = SparseRandomProjection(n_components=dim,random_state=5)
digitX2 = rp.fit_transform(digitX)
digit2 = pd.DataFrame(np.hstack((digitX2,np.atleast_2d(digitY))))
cols = list(range(digit2.shape[1]))
cols[-1] = 'Class'
digit2.columns = cols
digit2.to_csv(out+'digit_datasets.csv',index=False,header=False)
