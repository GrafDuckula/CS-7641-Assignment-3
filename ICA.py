

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA, PCA
from helpers import get_data_from_csv

out = './ICA/'

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

ica = FastICA(random_state=5)
kurt = {}
for dim in dims_wine:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'wine scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(digitX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'digit scree.csv')

# Data for 2

grid ={'ica__n_components':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(digitX,digitY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'digit dim red.csv')


# data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 6
ica = FastICA(n_components=dim,random_state=10)

wineX2 = ica.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY))))
cols = list(range(wine2.shape[1]))
cols[-1] = 'Class'
wine2.columns = cols
wine2.to_csv(out+'wine_datasets.csv',index=False,header=False)

dim = 60
ica = FastICA(n_components=dim,random_state=10)

digitX2 = ica.fit_transform(digitX)
digit2 = pd.DataFrame(np.hstack((digitX2,np.atleast_2d(digitY))))
cols = list(range(digit2.shape[1]))
cols[-1] = 'Class'
digit2.columns = cols
digit2.to_csv(out+'digit_datasets.csv',index=False,header=False)


# # visualize the projection in new space
# import matplotlib.pyplot as plt
# n_col = 10
# n_row = 2
# dims_digit=[n_col * n_row]
# ica = FastICA(random_state=5)
# pca = PCA(random_state=5)
#
# for dim in dims_digit:
#     ica.set_params(n_components=dim)
#     tmp = ica.fit_transform(digitX)
#     tmp = pd.DataFrame(tmp)
#     tmp = tmp.kurt(axis=0)
#     # images = ica.components_
#     images = ica.mixing_.T
#     print 'dim = ', dim
#     print tmp
#
#     cmap = plt.cm.gray
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     for i, comp in enumerate(images):
#         plt.subplot(n_row, n_col, i + 1)
#         vmax = max(comp.max(), -comp.min())
#         image_shape = (16, 16)
#         plt.imshow(comp.reshape(image_shape), cmap=cmap,
#                    interpolation='nearest',
#                    vmin=-vmax, vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#         plt.suptitle(('ICA, components = %s' % dim), fontsize=18, fontweight='bold')
#     plt.subplots_adjust(0.01, 0.05, 0.99, 1.2, 0.04, 0.)
#     plt.show()
#
#     # PCA
#     # pca.set_params(n_components=dim)
#     tmp = pca.fit_transform(digitX)
#     images = pca.components_
#     print 'dim = ', dim
#     print tmp
#
#     cmap = plt.cm.gray
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     for i, comp in enumerate(images[:dim]):
#         plt.subplot(n_row, n_col, i + 1)
#         vmax = max(comp.max(), -comp.min())
#         image_shape = (16, 16)
#         plt.imshow(comp.reshape(image_shape), cmap=cmap,
#                    interpolation='nearest',
#                    vmin=-vmax, vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#     plt.suptitle(('PCA, components = %s' % dim),fontsize=18, fontweight='bold')
#     plt.subplots_adjust(0.01, 0.05, 0.99, 1.2, 0.04, 0.)
#     plt.show()
#
#     plt.figure(figsize=(2. * n_col, 2.26 * n_row))
#     for i, comp in enumerate(digitX[:dim]):
#         plt.subplot(n_row, n_col, i + 1)
#         vmax = max(comp.max(), -comp.min())
#         image_shape = (16, 16)
#         plt.imshow(comp.reshape(image_shape), cmap=cmap,
#                    interpolation='nearest',
#                    vmin=-vmax, vmax=vmax)
#         plt.xticks(())
#         plt.yticks(())
#     plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
#     plt.show()