# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from helpers import get_data_from_csv

out = './BASE/'
np.random.seed(42)

# import wine quality data
wineX, wineY = get_data_from_csv(out+"wine_trg.csv", n_features=11, sep=',', header=None)
digitX, digitY = get_data_from_csv(out+"digit_trg.csv", n_features=256, sep=',', header=None)

wineX = StandardScaler().fit_transform(wineX)
digitX= StandardScaler().fit_transform(digitX)


# benchmarking for chart type 2
grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Wine NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitX,digitY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Digit NN bmk.csv')
