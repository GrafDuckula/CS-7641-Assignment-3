# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
from helpers import get_data_from_csv, compute_bic

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(42)

rds = ['BASE','PCA','ICA','RF','RP']
rds = [sys.argv[1]]
for rd in rds:
    out = './{}/'.format(rd)

    # import wine quality data

    if rd == "BASE":
        wineX, wineY = get_data_from_csv(out+"wine_trg.csv", n_features=11, sep=',', header=None)
        digitX, digitY = get_data_from_csv(out+"digit_trg.csv", n_features=256, sep=',', header=None)
    else:
        wineX, wineY = get_data_from_csv(out+"wine_datasets.csv", sep=',', header=None)
        digitX, digitY = get_data_from_csv(out+"digit_datasets.csv", sep=',', header=None)


    wineX = StandardScaler().fit_transform(wineX)
    digitX = StandardScaler().fit_transform(digitX)

    clusters = [2,5,10,15,20,25,30,35,40]

    # Data for 1-3
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    adjRI = defaultdict(lambda: defaultdict(dict))
    bic = defaultdict(lambda: defaultdict(dict))
    silh = defaultdict(lambda: defaultdict(dict))
    km = kmeans(random_state=5)
    gmm = GMM(random_state=5)

    st = clock()
    for k in clusters:
        km.set_params(n_clusters=k)
        gmm.set_params(n_components=k)
        km.fit(wineX)
        gmm.fit(wineX)
        SSE[k]['Wine'] = km.score(wineX)
        ll[k]['Wine'] = gmm.score(wineX)
        acc[k]['Wine']['Kmeans'] = cluster_acc(wineY.ravel(),km.predict(wineX))
        acc[k]['Wine']['GMM'] = cluster_acc(wineY.ravel(),gmm.predict(wineX))
        adjMI[k]['Wine']['Kmeans'] = ami(wineY.ravel(),km.predict(wineX))
        adjMI[k]['Wine']['GMM'] = ami(wineY.ravel(),gmm.predict(wineX))
        adjRI[k]['Wine']['Kmeans'] = ari(wineY.ravel(),km.predict(wineX))
        adjRI[k]['Wine']['GMM'] = ari(wineY.ravel(),gmm.predict(wineX))
        bic[k]['Wine']['Kmeans'] = -compute_bic(km,wineX)
        bic[k]['Wine']['GMM'] = gmm.bic(wineX)
        silh[k]['Wine']['Kmeans'] = silhouette_score(wineX,km.predict(wineX))
        silh[k]['Wine']['GMM'] = silhouette_score(wineX,gmm.predict(wineX))

        km.fit(digitX)
        gmm.fit(digitX)
        SSE[k]['Digit'] = km.score(digitX)
        ll[k]['Digit'] = gmm.score(digitX)
        acc[k]['Digit']['Kmeans'] = cluster_acc(digitY.ravel(),km.predict(digitX))
        acc[k]['Digit']['GMM'] = cluster_acc(digitY.ravel(),gmm.predict(digitX))
        adjMI[k]['Digit']['Kmeans'] = ami(digitY.ravel(),km.predict(digitX))
        adjMI[k]['Digit']['GMM'] = ami(digitY.ravel(),gmm.predict(digitX))
        adjRI[k]['Digit']['Kmeans'] = ari(digitY.ravel(),km.predict(digitX))
        adjRI[k]['Digit']['GMM'] = ari(digitY.ravel(),gmm.predict(digitX))
        bic[k]['Digit']['Kmeans'] = -compute_bic(km,digitX)
        bic[k]['Digit']['GMM'] = gmm.bic(digitX)
        silh[k]['Digit']['Kmeans'] = silhouette_score(digitX,km.predict(digitX))
        silh[k]['Digit']['GMM'] = silhouette_score(digitX,gmm.predict(digitX))

        print(k, clock()-st)


    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns = lambda x: x+' SSE (left)',inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)
    adjRI = pd.Panel(adjRI)
    bic = pd.Panel(bic)
    silh = pd.Panel(silh)

    SSE.to_csv(out+'SSE.csv')
    ll.to_csv(out+'logliklihood.csv')
    acc.ix[:,:,'Digit'].to_csv(out+'Digit acc.csv')
    acc.ix[:,:,'Wine'].to_csv(out+'Wine acc.csv')
    adjMI.ix[:,:,'Digit'].to_csv(out+'Digit adjMI.csv')
    adjMI.ix[:,:,'Wine'].to_csv(out+'Wine adjMI.csv')
    adjRI.ix[:,:,'Digit'].to_csv(out+'Digit adjRI.csv')
    adjRI.ix[:,:,'Wine'].to_csv(out+'Wine adjRI.csv')
    bic.ix[:,:,'Digit'].to_csv(out+'Digit bic.csv')
    bic.ix[:,:,'Wine'].to_csv(out+'Wine bic.csv')
    silh.ix[:,:,'Digit'].to_csv(out+'Digit silh.csv')
    silh.ix[:,:,'Wine'].to_csv(out+'Wine silh.csv')


    # NN fit data (2,3)

    grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km',km),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10)

    gs.fit(wineX,wineY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Wine cluster Kmeans.csv')


    grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm',gmm),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(wineX,wineY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Wine cluster GMM.csv')


    grid ={'km__n_clusters':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    km = kmeans(random_state=5)
    pipe = Pipeline([('km',km),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(digitX,digitY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Digit cluster Kmeans.csv')


    grid ={'gmm__n_components':clusters,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    gmm = myGMM(random_state=5)
    pipe = Pipeline([('gmm',gmm),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(digitX,digitY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'Digit cluster GMM.csv')

    # For chart 4/5
    wineX2D = TSNE(verbose=10,random_state=5).fit_transform(wineX)
    digitX2D = TSNE(verbose=10,random_state=5).fit_transform(digitX)

    print wineX
    print wineX2D

    if rd == "BASE":
        wine2D = pd.DataFrame(np.hstack((wineX2D,np.atleast_2d(wineY))),columns=['x','y','target'])
        digit2D = pd.DataFrame(np.hstack((digitX2D,np.atleast_2d(digitY))),columns=['x','y','target'])
    else:
        wine2D = pd.DataFrame(np.hstack((wineX2D,np.atleast_2d(wineY).T)),columns=['x','y','target'])
        digit2D = pd.DataFrame(np.hstack((digitX2D,np.atleast_2d(digitY).T)),columns=['x','y','target'])
    wine2D.to_csv(out+'wine2D.csv')
    digit2D.to_csv(out+'digit2D.csv')

    # silhouette analysis
    # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
    datasets = ['wine','digit']
    datasets = ['wine']
    infile = './@EG@/@DATA@2D.csv'
    columns = ['num','x','y','target']
    algo = rd

    for x in datasets:
        if x == 'wine':
            X = wineX
            clusters = [2,5,10]
            clusters = [5]
            m = 1.1
        else:
            X = digitX
            clusters = [5,10,20]
            clusters = [10]
            m = 1.0

        # import t-sne data
        data_list = pd.DataFrame(columns=columns)
        fname = infile.replace('@EG@', algo).replace('@DATA@', x)
        data = pd.read_csv(fname, sep=',', header='infer')

        for k in clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            fig.set_size_inches(24, 9)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 0.6])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (k + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=k, random_state=10)
            # clusterer = GMM(n_components=k, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", k,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / k *m)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / k *m)
            ax2.scatter(data['x'],
                        data['y'], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("T-SNE 1st dimension")
            ax2.set_ylabel("T-SNE 2nd dimension")

            # 3rd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(data['target'].astype(float) / 10*m)
            ax3.scatter(data['x'],
                        data['y'], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors, edgecolor='k')
            # ax3.legend(loc="best")
            ax3.set_title("The visualization of data with real label.")
            ax3.set_xlabel("T-SNE 1st dimension")
            ax3.set_ylabel("T-SNE 2nd dimension")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % k,
                          " %s" % rd,
                          " %s" % x),
                         fontsize=14, fontweight='bold')
            # plt.suptitle(("Silhouette analysis for GMM clustering on sample data "
            #               "with n_clusters = %d" % k,
            #               " %s" % rd,
            #               " %s" % x),
            #              fontsize=14, fontweight='bold')
            plt.subplots_adjust(0.12, 0.11, 0.90, 0.88, 0.2, 0.2)
            plt.show()
