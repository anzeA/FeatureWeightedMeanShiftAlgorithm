import os
import pandas as pd

from WBMS import WBMS
import numpy as np

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans
from pyrcc import RccCluster
from sklearn.cluster import MeanShift


def get_clusters_wbms(data, h, lambda_, max_iter=100):
    wbms = WBMS(h, lambda_, max_iter)
    wbms.fit(data)
    return wbms.labels_


def get_clsuters_pyclustering(data, method, **kwargs):
    method_instance = method(data, **kwargs)
    method_instance.process()
    label_list = method_instance.get_clusters()  # returns list of lsits
    label = np.zeros(data.shape[0], dtype=int)
    for i, lab in enumerate(label_list):
        label[lab] = i
    return label


def get_clusters_kmeans(data, k):
    return get_clsuters_pyclustering(data, kmeans, initial_centers=np.random.normal(size=(k, data.shape[1])))


def get_clusters_gmeans(data):
    return get_clsuters_pyclustering(data, gmeans)


def get_clusters_xmeans(data):
    return get_clsuters_pyclustering(data, xmeans)


def get_clusters_rcc(data):
    rcc = RccCluster(10, verbose=False)
    label = rcc.fit(data)
    return label
    # print(max(label),len(set(label)))
    # print(sorted(set(label)))
    # label2new_label = {n:i for i,n in enumerate(set(label))}
    # print('dif labels',len(set(label)))
    # return np.array([label2new_label[i] for i in label])


def get_clusters_mean_shift(data):
    return MeanShift().fit_predict(data)


def get_clusters_ewdp(data_path, lambda_w, lambda_k):
    output_file = os.path.join('data', 'tmp_ewdp.csv')
    command = f"Rscript.exe --vanilla .\\EWDP.R {data_path} {lambda_w} {lambda_k} {output_file}"
    print(command)
    os.system(command)
    df = pd.read_csv(output_file)
    os.remove(output_file)
    return df['clusters'].to_numpy()


def get_clusters_dp_means(data_path, lambda_,verbose=True):
    output_file = os.path.join('data', 'tmp_dp_means.csv')
    command = f"Rscript.exe --vanilla .\DP-means.R  {data_path} {lambda_} {output_file}"
    if verbose: print(command)
    os.system(command)
    df = pd.read_csv(output_file)
    os.remove(output_file)
    return df['clusters'].to_numpy()


def get_clusters_wbms_R(data_path,h, lambda_,return_weigths=False,verbose=True):
    output_file = os.path.join('data', 'tmp_wbms_R.csv')
    output_file_weights = os.path.join('data', 'tmp_wbms_w_R.csv')
    command = f"Rscript.exe --vanilla .\WBMS_run.R  {data_path} {h} {lambda_} {output_file} {output_file_weights}"
    if verbose:
        print(command)
    os.system(command)
    df = pd.read_csv(output_file)

    if return_weigths:
        df_w = pd.read_csv(output_file_weights)
        return df['clusters'].to_numpy(),df_w['x'].to_numpy()
    return df['clusters'].to_numpy()


def get_clusters_wgmeans(data_path, beta=2, alpha=0.0001):  # beta je lah tud 5
    output_file = os.path.join('data', 'tmp_wgmeans.csv')
    command = f"Rscript.exe --vanilla .\wgmeans_run.R  {data_path} {beta} {alpha} {output_file}"
    print(command)
    os.system(command)
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        print('Wgmeans error. Did not complete. Returning only one cluster')
        return np.zeros(pd.read_csv(data_path).shape[0])
    os.remove(output_file)
    return df['clusters'].to_numpy()
