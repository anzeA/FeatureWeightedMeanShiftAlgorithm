import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import euclidean
from sklearn import config_context
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.cluster import estimate_bandwidth
from sklearn.datasets import make_biclusters, make_blobs
from sklearn.metrics import pairwise_distances, euclidean_distances, normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pyclustering.cluster import cluster_visualizer

from sklearn.utils.validation import check_is_fitted

from dataset import create_dataset_simulation


class WBMS(ClusterMixin, BaseEstimator):

    def __init__(
            self,
            h=0.3,
            lambda_=0.01,
            eps=1e-5,
            max_iter=300
    ):
        self.h = h
        self.eps = eps
        self.lambda_ = lambda_
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to cluster.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
               Fitted instance.
        """
        X = self._validate_data(X)
        h = self.h
        #if h is None:
        #    h = estimate_bandwidth(X)
        #    print(f'Estimated bandwith is {h}')
        if h <= 0:
            raise ValueError(
                "bandwidth needs to be greater than zero or None, got %f" % h
            )

        n_samples, n_features = X.shape

        y = X.copy()
        w = np.full(shape=(n_features), fill_value=1 / n_features, dtype=np.float64)
        max_dist_prev = 1000
        for i in range(self.max_iter):
            try:
                y, w, max_dist = self._step(y, X, w, h, self.lambda_)
            except ValueError as v:
                print(f'Exception was raised: {v}')
                print('Return all data points in the same cluster')
                self._n_components, self.labels_ = 1, np.zeros(n_samples)
                self.w_ = np.full(shape=(n_features), fill_value=1 / n_features, dtype=np.float64)
                return self
            # if i < 5 or i % 10==0:
            #    plt.scatter(y[:,0],y[:,1])
            #    plt.show()
            converge_con = abs(max_dist - max_dist_prev)
            # print(converge_con)
            self.n_iter_ = i+1
            if converge_con < 1e-4 and i > 5:
                break
            max_dist_prev = max_dist
            if i == self.max_iter - 1:
                warnings.warn(f'Did not covrge. Convege_con is {converge_con}')
        self.w_ = w
        # postprocess
        self._n_components, self.labels_ = connected_components(pairwise_distances(y, metric='euclidean') < self.eps,
                                                                directed=False, return_labels=True)
        # self.cluster_centers_, self.labels_ = cluster_centers, labels
        return self

    @staticmethod
    def _step(y, X, w, h, lambda_):
        n_samples, n_features = X.shape
        dist = pairwise_distances(y, metric=euclidean, w=w)
        dist = dist / h
        dist = dist * dist

        # kernal
        # dist = dist * dist  # squere
        K = np.exp(-dist)

        #diag_el = np.arange(n_samples)
        #K[diag_el, diag_el] = 0

        y_new = np.zeros(y.shape)
        sum_sample_weights = np.sum(K, axis=1)

        for i in range(n_samples):
            weigted = y * K[i, :].reshape((n_samples, 1))
            y_new[i, :] = np.sum(weigted, axis=0) / sum_sample_weights[i]

        max_dist = pairwise_distances(y_new,
                                      metric=euclidean).max()

        w = (X - y_new)
        w = w * w
        w = np.mean(w, axis=0)
        w = -w / (lambda_)  #
        w = np.exp(w)
        w = w / (np.sum(w) + 1e-10)

        return y_new, w, max_dist



def multiple_runs():
    df = pd.DataFrame()
    df_score = pd.DataFrame()
    for i in range(20):
        X, y = create_dataset_simulation(4, 5, 80, 20, 0.02, i)

        df_plot = pd.DataFrame({'x':X})

        X = StandardScaler().fit_transform(X)
        wbms = WBMS(h=0.1, lambda_=10, max_iter=100)
        wbms.fit(X)
        df = df.append(pd.Series(wbms.w_), ignore_index=True)
        nmi = normalized_mutual_info_score(y, wbms.labels_)
        ari = adjusted_rand_score(y, wbms.labels_)
        df_score = df_score.append(pd.Series({'NMI': nmi, 'ARI': ari}), ignore_index=True)
    sns.boxplot(data=df)
    plt.show()
    sns.boxplot(data=df_score)
    plt.show()


def test_diff_parms():
    df_score = pd.DataFrame()
    #lambdas = [i * 10 if i > 0 else 5 for i in range(8)]
    h_vals = [0.05 * i for i in range(6, 10)]
    for l in [ 1e-3, 0.005, 1e-2, 0.025]:  # lambdas:
        for h in h_vals:  # h_vals:
            for i in range(10):
                X, y = create_dataset_simulation(2, 2, 200, 32, 0.02, i)

                X = StandardScaler().fit_transform(X)
                wbms = WBMS(h=h, lambda_=l, max_iter=100)
                wbms.fit(X)
                y_pred = wbms.labels_
                nmi = normalized_mutual_info_score(y, y_pred)
                ari = adjusted_rand_score(y, y_pred)
                df_score = df_score.append(pd.Series({'NMI': nmi, 'ARI': ari, 'lambda': l, 'h': h}), ignore_index=True)
                print(df_score)
                #data_plot = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'c': y_pred})
                # sns.scatterplot(data=data_plot, x='x', y='y', hue='c')
                # plt.title(f'h {h},lambda {l}')
                # plt.show()

        df_score.to_csv('test_diff_parms_df_score2.csv', index=False)
        plt.clf()
        plt.close()
        try:
            for k in ['NMI', 'ARI']:
                df_tmp = df_score.copy()
                df_tmp = df_tmp.groupby(['lambda', 'h']).mean().reset_index()
                df_tmp = df_tmp.pivot('lambda', 'h', k)
                sns.heatmap(df_tmp)
                plt.title(k)
                plt.show()
        except:
            pass



def test():
    df_score = pd.DataFrame()#columns=['kmeans', 'gmeans', 'mean_shift', 'rcc', 'dp_means', 'wgmeans', 'wbms', 'type', '# Clusters'])

    for c in [5, 10,20]:# #20, 30, 40, 50]:
        for h in [0.25,0.3,0.35,0.4]:
            for i in range(10):
                print(f'Data: c {c}, i {i}')
                X, y = create_dataset_simulation(c, 5, 20 * c, 20, 0.02, i)

                X = StandardScaler().fit_transform(X)
                wbms = WBMS(h=h,lambda_= 0.005, max_iter=100)
                wbms.fit(X)
                y_pred = wbms.labels_
                nmi = normalized_mutual_info_score(y, y_pred)
                ari = adjusted_rand_score(y, y_pred)
                df_score = df_score.append(pd.Series({'NMI': nmi, 'ARI': ari,'# Clusters':c,'n_clusters':y_pred.max()+1,'h':h}), ignore_index=True)
                #data_plot = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'c': y_pred})
                #sns.scatterplot(data=data_plot, x='x', y='y', hue='c')
                #plt.title(f'True clusters {c}')
                #plt.show()
            try:
                for t in ['NMI', 'ARI', 'n_clusters']:
                    sns.lineplot(data=df_score,x='# Clusters',y=t,hue='h')
                    #plt.ylabel(t)
                    plt.show()
            except:
                pass



if __name__ == '__main__':

    #test()
    #test_feature_weights()
    #test_diff_parms()
    # multiple_runs()
    exit()

    # X, y = make_blobs(n_samples=100, n_features=20, random_state=13, centers=2,cluster_std=0.02)
    # X = pd.read_csv('debug_data.csv').to_numpy(dtype=np.float64)
    # X = np.array([[-1,0,1],[0,0,0],[1,1,-1],[1,1,1]],dtype=np.float64)
    # y = np.zeros(X.shape[0])
    X, y = create_dataset_simulation(4, 5, 80, 10, 0.02, 16)
    X = StandardScaler().fit_transform(X)
    df = pd.DataFrame()
    df['x'] = X[:, 0]
    df['y'] = X[:, 1]
    df['c'] = y
    # sns.scatterplot(data=df, x='x', y='y', hue='c')
    # plt.show()
    # for h in [0.1]:#0.15,0.2,0.25,0.3,0.5,0.6,0.7,0.8
    wbms = WBMS(h=0.3, lambda_=0.01, max_iter=100)
    wbms.fit(X)
    df['l'] = wbms.labels_
    sns.scatterplot(data=df, x='x', y='y', hue='l')
    plt.show()
    print('Number of clusters is:', len(set(wbms.labels_)))
    print(wbms.w_)
