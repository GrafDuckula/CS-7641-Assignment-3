# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
import os 
import sklearn.model_selection as ms
from helpers import get_data_from_csv

RANDOM_STATE = 42

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './BASE/'


# import wine quality data
wineX, wineY = get_data_from_csv("winequality-white.csv", n_features=11, sep=';')  # instances
real_Y = []
for Y in wineY:
    for y in Y:
        real_Y.append(y)
wineY = np.array(real_Y)

# import semeion data
digitX, digitY = get_data_from_csv("semeion.data.csv", n_features=256, sep=' ', header=None)
real_Y = []
for Y in digitY:
    for i in range(10):
        if Y[i] == 1.0:
            real_Y.append(i)
digitY = np.array(real_Y)

# split data
wine_trgX, wine_tstX, wine_trgY, wine_tstY = \
    ms.train_test_split(wineX, wineY, test_size=0.2, random_state=RANDOM_STATE, stratify=wineY)
digit_trgX, digit_tstX, digit_trgY, digit_tstY = \
    ms.train_test_split(digitX, digitY, test_size=0.2, random_state=RANDOM_STATE, stratify=digitY)

wine_trgY = np.atleast_2d(wine_trgY).T
wine_tstY = np.atleast_2d(wine_tstY).T
digit_trgY = np.atleast_2d(digit_trgY).T
digit_tstY = np.atleast_2d(digit_tstY).T

wine_trg = pd.DataFrame(np.hstack((wine_trgX,wine_trgY)))
wine_tst = pd.DataFrame(np.hstack((wine_tstX,wine_tstY)))

digit_trg = pd.DataFrame(np.hstack((digit_trgX,digit_trgY)))
digit_tst = pd.DataFrame(np.hstack((digit_tstX,digit_tstY)))


wine_trg = wine_trg.dropna(axis=1,how='all')
wine_tst = wine_tst.dropna(axis=1,how='all')
digit_trg = digit_trg.dropna(axis=1,how='all')
digit_tst = digit_tst.dropna(axis=1,how='all')

# wine_cols = list(range(wine_trg.shape[1]))
# digit_cols = list(range(digit_trg.shape[1]))
# wine_cols[-1] = 'Class'
# digit_cols[-1] = 'Class'
#
# wine_trg.columns = wine_cols
# wine_tst.columns = wine_cols
# digit_trg.columns = digit_cols
# digit_tst.columns = digit_cols

wine_trg.to_csv(OUT+'wine_trg.csv',index=False,header=False)
wine_tst.to_csv(OUT+'wine_test.csv',index=False,header=False)

digit_trg.to_csv(OUT+'digit_trg.csv',index=False,header=False)
digit_tst.to_csv(OUT+'digit_test.csv',index=False,header=False)


# wine_trg.to_hdf(OUT+'datasets.hdf','wine',complib='blosc',complevel=9)
# wine_tst.to_hdf(OUT+'datasets.hdf','wine_test',complib='blosc',complevel=9)
# digit_trg.to_hdf(OUT+'datasets.hdf','digit',complib='blosc',complevel=9)
# digit_tst.to_hdf(OUT+'datasets.hdf','digit_test',complib='blosc',complevel=9)
