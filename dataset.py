import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder


def read_appendicitis():
    df = pd.read_csv(os.path.join('real_data', 'appendicitis.csv'), header=None)
    df.rename(columns={7: "label"}, inplace=True)
    return df


def read_gcm():
    df = pd.read_csv(os.path.join('real_data', 'GCM.csv'))
    df.rename(columns={'class': "label"}, inplace=True)
    df['label'] = LabelEncoder().fit_transform(df['label'])
    return df


def read_mammographic():
    df = pd.read_csv(os.path.join('real_data', 'mammographic.csv'), header=None)
    df.rename(columns={5: "label"}, inplace=True)
    return df


def read_movement_libras():
    df = pd.read_csv(os.path.join('real_data', 'movement_libras.csv'), header=None)
    df.rename(columns={90: "label"}, inplace=True)
    df['label'] = df['label'] - 1
    return df


def read_zoo():
    df = pd.read_csv(os.path.join('real_data', 'zoo.csv'), header=None)
    df.rename(columns={16: "label"}, inplace=True)
    df['label'] = df['label'] - 1
    return df


def read_nci9():
    mat = loadmat(os.path.join('real_data', 'nci9.mat'))
    df = pd.DataFrame(data=mat['X'], index=np.arange(mat['X'].shape[0]), columns=np.arange(mat['X'].shape[1]))
    # df.rename(columns={16: "label"},inplace=True)
    df['label'] = mat['Y'] - 1
    return df


def read_yale():
    mat = loadmat(os.path.join('real_data', 'yale.mat'))
    df = pd.DataFrame(data=mat['X'], index=np.arange(mat['X'].shape[0]), columns=np.arange(mat['X'].shape[1]))
    # df.rename(columns={16: "label"},inplace=True)
    df['label'] = mat['Y'] - 1
    return df




def create_dataset_simulation(k, n_informative_features, n_samples, n_features, scale, seed=None):
    # k is number of clusters
    # p number of informative features
    assert n_informative_features <= n_features
    assert n_samples > 0
    rand = np.random.RandomState(seed)
    X = rand.random(size=(k, n_informative_features))
    X[:, 5:] = 0
    data = []
    y = []
    for i in range(n_samples):
        cluster = rand.randint(k)
        assert 0 <= cluster < k
        center = X[cluster, :]
        sample = np.concatenate(
            [rand.normal(center, scale), rand.normal(0, 1, size=n_features - n_informative_features)])
        data.append(sample)
        y.append(cluster)
    return np.array(data), np.array(y)

