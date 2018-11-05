"""
@author: Aiping
"""

import pandas as pd
from itertools import product
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#000000']


# only BASE
plt.rcParams.update({'font.size': 12})

infile = './@EG@/@DATA@ @met@.csv'
rds = ['BASE']
datasets = ['Wine','Digit']
metrics = ['acc', 'adjMI', 'adjRI', 'silh']
columns = [0, 2,5,10,15,20,25,30,35,40]

for dataset in datasets:
    for m in metrics:
        for rd in rds:
            data_list = pd.DataFrame()
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
            data['cluster'] = [int(i) for i in data.index]

            plt.plot(data['cluster'], data['Kmeans'], 'o-', label=rd+' Kmeans')
            plt.plot(data['cluster'], data['GMM'], '.-', label=rd + ' GMM')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('clusters')
        plt.ylabel(m)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}'.format(m, dataset))
        plt.show()

# only BASE/bic
plt.rcParams.update({'font.size': 12})

infile = './@EG@/@DATA@ @met@.csv'
rds = ['BASE']
datasets = ['Wine','Digit']
metrics = ['bic']
columns = [0, 2,5,10,15,20,25,30,35,40]

for dataset in datasets:
    for m in metrics:
        for rd in rds:
            data_list = pd.DataFrame()
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
            data['cluster'] = [int(i) for i in data.index]

            plt.plot(data['cluster'], data['Kmeans'], 'o-', label=rd+' Kmeans')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('clusters')
        plt.ylabel(m)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}'.format(m, dataset))
        plt.show()

        for rd in rds:
            data_list = pd.DataFrame()
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
            data['cluster'] = [int(i) for i in data.index]

            plt.plot(data['cluster'], data['GMM'], '.-', label=rd + ' GMM')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('clusters')
        plt.ylabel(m)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}'.format(m, dataset))
        plt.show()


# acc and adjMI comparison together, grouped by different clustering methods
plt.rcParams.update({'font.size': 12})

infile = './@EG@/@DATA@ @met@.csv'
rds = ['BASE','PCA','ICA','RF','RP']
datasets = ['Wine','Digit']
metrics = ['acc', 'adjMI', 'adjRI', 'bic', 'silh']
columns = [0, 2,5,10,15,20,25,30,35,40]

for dataset in datasets:
    for m in metrics:
        for rd in rds:
            data_list = pd.DataFrame()
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
            data['cluster'] = [int(i) for i in data.index]

            plt.plot(data['cluster'], data['Kmeans'], 'o-', label=rd+' Kmeans')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('clusters')
        plt.ylabel(m)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}'.format(m, dataset))
        plt.show()

        for rd in rds:
            data_list = pd.DataFrame()
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
            data['cluster'] = [int(i) for i in data.index]

            plt.plot(data['cluster'], data['GMM'], '.-', label=rd+' GMM')

        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        plt.xlabel('clusters')
        plt.ylabel(m)
        plt.grid(True)
        plt.legend(loc="best")
        plt.title('{}--{}'.format(m, dataset))
        plt.show()


# SSE and log likely hood
rds = ['BASE','PCA','ICA','RF','RP']
metrics = ['SSE', 'logliklihood']
columns = ['Digit','Wine']

for m in metrics:
    for rd in rds:
        data_list = pd.DataFrame(columns=columns)
        fname = infile.replace('@EG@', rd).replace('@met@', m)
        data = pd.read_csv(fname, sep=',', header='infer', index_col=0)
        data.columns = columns
        data['cluster'] = [int(i) for i in data.index]
        print data

        plt.plot(data['cluster'], data['Digit'], 'o-', label='{}  {}  {}'.format(rd, m, 'Digit'))

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('clusters')
    plt.ylabel(m)
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}'.format(m, 'Digit'))
    plt.show()

    for rd in rds:
        data_list = pd.DataFrame(columns=columns)
        fname = infile.replace('@EG@', rd).replace('@met@', m)
        data = pd.read_csv(fname, sep=',', header='infer', index_col=0)
        data.columns = columns
        data['cluster'] = [int(i) for i in data.index]

        plt.plot(data['cluster'], data['Wine'], '.-', label='{}  {}  {}'.format(rd, m, 'Wine'))

    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    plt.xlabel('clusters')
    plt.ylabel(m)
    plt.grid(True)
    plt.legend(loc="best")
    plt.title('{}--{}'.format(m, 'Wine'))
    plt.show()


# scree for PCA, ICA and RF
plt.rcParams.update({'font.size': 12})
infile = './@EG@/@DATA@ scree.csv'
rds = ['PCA','ICA','RF']
datasets = ['wine','digit']

for dataset in datasets:
    for rd in rds:
        data_list = pd.DataFrame()
        fname = infile.replace('@EG@', rd).replace('@DATA@', dataset)

        data = pd.read_csv(fname, sep=',', header='infer', index_col=0)
        data.columns = ['scree']
        data['cluster'] = [int(i) for i in data.index]

        plt.rc('axes', prop_cycle=(cycler('color', colors)))

        if rd == 'PCA':
            data['explained_variance_ratio_'] = data['scree']/np.sum(data['scree'])
            data['cumsum'] = np.cumsum(data['explained_variance_ratio_'])
            plt.xlabel('Principle component')
            plt.plot(data['cluster'], data['cumsum'], '.-', label='cumulative explained variance ratio')
            plt.plot(data['cluster'], data['explained_variance_ratio_'], '.-', label=rd + ' explained variance ratio')
            plt.ylabel('explained_variance_')
            plt.title('{}--{}'.format('explained_variance_', dataset))
        elif rd == 'ICA':
            plt.plot(data['cluster'], data['scree'], '.-', label=rd)
            plt.xlabel('n_components')
            plt.ylabel('kurt')
            plt.title('{}--{}'.format('kurt', dataset))
        elif rd == 'RF':
            plt.plot(data['cluster'], data['scree'], '.-', label=rd)
            plt.xlabel('n_components')
            plt.ylabel('feature_importances')
            plt.title('{}--{}'.format('feature_importances', dataset))
        plt.grid(True)
        plt.legend(loc="best")

        plt.show()

# scree 1&2 of RP

infile = './@EG@/@DATA@ scree@n@.csv'
rds = ['RP']
datasets = ['wine','digit']

for dataset in datasets:
    for rd in rds:
        for n in [1,2]:
            fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@n@', str(n))
            # data_list = pd.DataFrame()
            data = pd.read_csv(fname, sep=',', header='infer', index_col=0)
            data.columns = [str(i) for i in range(10)]

            data_new = pd.DataFrame()
            data_new['mean'] = data.mean(axis=1)
            data_new['std'] = data.std(axis=1)
            data_new['cluster'] = [int(i) for i in data.index]

            plt.rc('axes', prop_cycle=(cycler('color', colors)))
            plt.xlabel('n_components')

            plt.fill_between(data_new['cluster'], data_new['mean'] - data_new['std'],
                             data_new['mean'] + data_new['std'], alpha=0.1)
            plt.plot(data_new['cluster'], data_new['mean'], '.-')

            if n==1:
                y = 'pairwise Dist Corr'
            elif n==2:
                y = 'reconstruction Error'
            plt.ylabel(y)
            plt.title('{}--{}'.format(y, dataset))

            plt.grid(True)
            plt.legend(loc="best")

            plt.show()


# acc and adjMI comparison individually
# GMM and Kmeans in one plot
# not used in the reports

# infile = './@EG@/@DATA@ @met@.csv'
# rds = ['BASE','PCA','ICA','RF','RP']
# datasets = ['Wine','Digit']
# metrics = ['acc', 'adjMI']
# columns = [0, 2,5,10,15,20,25,30,35,40]
#
# for dataset in datasets:
#     for rd in rds:
#         for m in metrics:
#             data_list = pd.DataFrame()
#             fname = infile.replace('@EG@', rd).replace('@DATA@', dataset).replace('@met@', m)
#             data = pd.read_csv(fname, sep=',', header='infer', index_col=0).T
#             data['cluster'] = [int(i) for i in data.index]
#
#             plt.plot(data['cluster'], data['GMM'], '.-', label='GMM')
#             plt.plot(data['cluster'], data['Kmeans'], '.-', label='Kmeans')
#
#             plt.rc('axes', prop_cycle=(cycler('color', colors)))
#             plt.xlabel('clusters')
#             plt.ylabel(m)
#             plt.grid(True)
#             plt.legend(loc="best")
#             plt.title('{}--{}--{}'.format(m, rd, dataset))
#             plt.show()
