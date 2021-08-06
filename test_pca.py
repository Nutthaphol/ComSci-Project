from function.bestEps import bestEps
from function.bestKmeans import best_k
from function.bestPCA import bestPCA
from function.tf_idf import TF_IDF
from function.sse import SSE
from function.purityCluster import purity
from function.kmean import Kmean
from function.cleanTextTH import CleanText
from function.dbscan import DBSCAN


import pandas as pd
import time


if __name__ == '__main__':
    df = pd.read_csv('lineData/comsci_data.csv')

    text = df.text

    fe = TF_IDF(text=text, format='thai')

    n_component = min(fe.shape[0], fe.shape[1])

    fe_pca = bestPCA(feature=fe, n_component=n_component)

    k_value = best_k(feature=fe, max_=int(len(text)*0.1))
    k_value_pca = best_k(feature=fe_pca, max_=int(len(fe)*0.1))

    eps_3 = bestEps(feature=fe)
    eps_5 = bestEps(feature=fe)
    eps_3_pca = bestEps(feature=fe_pca)
    eps_5_pca = bestEps(feature=fe_pca)
    
    #kmean 
    print('kmean non pca')
    kmeans = Kmean(fe=fe, data_=df.copy(), k_value=k_value)
    sse = sum(SSE(kmeans.copy()).sse)
    print('k value = ', k_value)
    print('sse = ', sse)
    print('shape = ', fe.shape)
    kmeans = kmeans.sort_values(by=['centroids_id'])
    print(kmeans.head(20))
    print('----------------------------')
    #kmean pca
    print('kmean pca')
    kmeans = Kmean(fe=fe_pca, data_=df.copy(), k_value=k_value_pca)
    sse = sum(SSE(kmeans.copy()).sse)
    print('k value = ', k_value_pca)
    print('sse = ', sse)
    print('shape = ', fe_pca.shape)
    kmeans = kmeans.sort_values(by=['centroids_id'])
    print(kmeans.head(20))

