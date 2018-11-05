

# Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from helpers import get_data_from_csv


if __name__ == '__main__':
    out = './RF/'
    
    np.random.seed(42)
    # import wine quality data
    wineX, wineY = get_data_from_csv("./BASE/wine_trg.csv", n_features=11, sep=',', header=None)
    digitX, digitY = get_data_from_csv("./BASE/digit_trg.csv", n_features=256, sep=',', header=None)

    wineX = StandardScaler().fit_transform(wineX)
    digitX = StandardScaler().fit_transform(digitX)

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    dims_wine = [i for i in range(2, 12)]

    # data for 1
    
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)

    fs_wine = rfc.fit(wineX,wineY).feature_importances_
    fs_digit = rfc.fit(digitX,digitY).feature_importances_

    tmp = pd.Series(np.sort(fs_wine)[::-1])
    tmp.to_csv(out+'wine scree.csv')

    tmp = pd.Series(np.sort(fs_digit)[::-1])
    tmp.to_csv(out+'digit scree.csv')

    # Data for 2
    filtr = ImportanceSelect(rfc)
    grid ={'filter__n':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(wineX,wineY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'wine dim red.csv')

    grid ={'filter__n':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('filter',filtr),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(digitX,digitY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'digit dim red.csv')

    # data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 6
    filtr = ImportanceSelect(rfc,dim)

    wineX2 = filtr.fit_transform(wineX,wineY)
    wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY))))
    cols = list(range(wine2.shape[1]))
    cols[-1] = 'Class'
    wine2.columns = cols
    wine2.to_csv(out + 'wine_datasets.csv', index=False, header=False)

    dim = 60
    filtr = ImportanceSelect(rfc,dim)
    digitX2 = filtr.fit_transform(digitX,digitY)
    digit2 = pd.DataFrame(np.hstack((digitX2,np.atleast_2d(digitY))))
    cols = list(range(digit2.shape[1]))
    cols[-1] = 'Class'
    digit2.columns = cols
    digit2.to_csv(out + 'digit_datasets.csv', index=False, header=False)
