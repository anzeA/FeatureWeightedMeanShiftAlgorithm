import os
from time import time
from itertools import chain

from collections import defaultdict
import numpy as np
import seaborn as sns
from WBMS import WBMS, create_dataset_simulation

# sns.set_theme(style="whitegrid", palette="Set2")
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.io import loadmat
from matplotlib import pyplot as plt
from gap_statistic import OptimalK
from dataset import *
from clustering import *

def create_defoult_parms(k):
    parms = dict()  # defaultdict(dict)
    parms['wgmeans'] = {'beta': 2, 'alpha': 0.01}
    parms['dp_means'] = {'lambda_': 1}
    parms['kmeans'] = {'k': k}
    parms['gmeans'] = {'k': k}
    parms['wbms'] = {'h': 0.35, 'lambda_': 0.01}
    parms['wbms_authors'] = {'h': 0.1, 'lambda_': 10}
    return parms


def run_expirement(df, k, parms=None,verbose=True,run=None):
    assert isinstance(df, pd.DataFrame)
    assert isinstance(k, int)
    if run is None:
        run = ['wgmeans','kmeans','gmeans','rcc','mean-shift','db-means' ,'wbms','wbms_authors']
    else:
        for alg in run:
            assert alg in  ['wgmeans','kmeans','gmeans','rcc','mean-shift','db-means' ,'wbms','wbms_authors']

    if parms is None:
        parms = create_defoult_parms(k)
    else:
        assert isinstance(parms, dict)
        # dparms = create_defoult_parms(k)
        # for k,v in dparms.items():
        #    if k not in parms:
        #        parms[k] = v

    labels_true = df['label'].to_numpy()
    assert min(labels_true) == 0
    df.drop('label', axis=1, inplace=True)
    # print(df.columns)

    X = df.to_numpy()
    X = StandardScaler().fit_transform(X)
    df = pd.DataFrame(X, columns=df.columns)
    tmp_file_path = os.path.join('data', 'tmp_data.csv')
    df.to_csv(tmp_file_path, index=False)

    #wgmeans
    if 'wgmeans' in run:
        start = time()
        label_wgmeans = get_clusters_wgmeans(tmp_file_path, **(parms['wgmeans']))
        end = time()-start
        if verbose: print(f'Time needed wgmeans: {end:.2f}s')
        df['label_wgmeans'] = label_wgmeans



    # label_ewdp = get_clusters_ewdp(tmp_file_path, lambda_w=1, lambda_k=k) # k je nao stevilo clustrov
    #k-means
    if 'kmeans' in run:
        start = time()
        label_kmeans = get_clusters_kmeans(X, **(parms['kmeans']))
        end = time()-start
        if verbose: print(f'Time needed kmeans: {end:.2f}s')
        df['label_kmeans'] = label_kmeans

    #Gmeans
    if 'gmeans' in run:
        start = time()
        label_gmeans = get_clusters_gmeans(X)
        end = time()-start
        if verbose: print(f'Time needed gmeans: {end:.2f}s')
        df['label_gmeans'] = label_gmeans

    

    #RCC
    if 'rcc' in run:
        start = time()
        label_rcc = get_clusters_rcc(X)
        end = time()-start
        if verbose: print(f'Time needed rcc: {end:.2f}s')
        df['label_rcc'] = label_rcc


    #mean shift
    if 'mean-shift'in run:
        start = time()
        label_mean_shift = get_clusters_mean_shift(X)
        end = time()-start
        if verbose: print(f'Time needed mean_shift: {end:.2f}s')
        df['label_mean_shift'] = label_mean_shift

    #DB-means
    if 'db-means' in run:
        start = time()
        label_dp_means = get_clusters_dp_means(tmp_file_path,verbose=verbose, **(parms['dp_means']))
        end = time()-start
        df['label_dp_means'] = label_dp_means
        if verbose: print(f'Time needed dp_means: {end:.2f}s')

    # WBMS
    if 'wbms' in run:
        start = time()
        label_wbms = get_clusters_wbms(X, **parms['wbms'])
        end = time() - start
        if verbose: print(f'Time needed WBMS: {end:.2f}s')
        df['label_wbms'] = label_wbms

    # wbms_authors
    if 'wbms_authors' in run:
        start = time()
        label_wbms_authors = get_clusters_wbms_R(tmp_file_path,verbose=verbose,**parms['wbms_authors'])
        end = time() - start
        if verbose: print(f'Time needed WBMS_authors: {end:.2f}s')
        df['label_wbms_authors'] = label_wbms_authors

    col_labels = [c for c in df.columns if 'label' == str(c)[:5]]
    nmi = {c[6:]: normalized_mutual_info_score(labels_true, df[c].to_numpy()) for c in col_labels}
    ari = {c[6:]: adjusted_rand_score(labels_true, df[c].to_numpy()) for c in col_labels}
    n_clusters = {c[6:]: len(set(df[c])) for c in col_labels}

    return df, nmi, ari, n_clusters


if __name__ == '__main__':
    # df,nmi,ari = run_expirement(create_data(100))
    t = time()
    # TODO poprav DB-means
    # df = run_real_data()
    df = run_simulatiom_effect_of_increasing_k()
    t = time() - t
    print('Time:', t, 's')
    df.to_csv('results3.csv')
