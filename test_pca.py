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

    pca_t = time.time()
    fe_pca = bestPCA(feature=fe, n_component=n_component)
    print('pca time = ', time.time() - pca_t)

    k_value = best_k(feature=fe, max_=int(len(text)*0.1))
    k_value_pca = best_k(feature=fe_pca, max_=int(len(fe)*0.1))

    eps = bestEps(feature=fe)
    eps_pca = bestEps(feature=fe_pca)
    
    #kmean 
    print('kmean non pca')
    kmeans_t = time.time()
    kmeans = Kmean(fe=fe, data_=df.copy(), k_value=k_value)
    print('kmeans time = ', time.time() - kmeans_t)
    sse = sum(SSE(kmeans.copy()).sse)
    print('k value = ', k_value)
    print('sse = ', sse)
    print('shape = ', fe.shape)
    print('----------------------------')

    #kmean pca
    print('kmean pca')
    kmeans_pca_t = time.time()
    kmeans = Kmean(fe=fe_pca, data_=df.copy(), k_value=k_value_pca)
    print('kmeans with pca time = ', time.time() - kmeans_pca_t)
    sse = sum(SSE(kmeans.copy()).sse)
    print('k value = ', k_value_pca)
    print('sse = ', sse)
    print('shape = ', fe_pca.shape)
    print('----------------------------')

    #dbscan 3
    print('dbscan minPts = 3')
    db3_t = time.time()
    dbscan_3 = DBSCAN(fe=fe, data_=df.copy(), eps_value=eps, min_pts=3)
    print('DBSCAN minPts 3 time = ', time.time() - db3_t)
    sse = sum(SSE(dbscan_3.copy()).sse)
    print('eps value = ', eps)
    print('sse = ', sse)
    print('shape = ', fe.shape)
    print('----------------------------')

    #dbscan 5
    print('dbscan minPts = 5')
    db5_t = time.time()
    dbscan_5 = DBSCAN(fe=fe, data_=df.copy(), eps_value=eps, min_pts=5)
    print('dbscan minPts 5 time = ', time.time() - db5_t)
    sse = sum(SSE(dbscan_5.copy()).sse)
    print('eps value = ', eps)
    print('sse = ', sse)
    print('shape = ', fe.shape)
    print('----------------------------')

    #dbscan 3
    print('dbscan minPts = 3 PCA')
    db3_pca_t = time.time()
    dbscan_3_pca = DBSCAN(fe=fe_pca, data_=df.copy(), eps_value=eps_pca, min_pts=3)
    print('dbscan minPts 3 PCA time = ', time.time() - db3_pca_t)
    sse = sum(SSE(dbscan_3.copy()).sse)
    print('eps value = ', eps_pca)
    print('sse = ', sse)
    print('shape = ', fe_pca.shape)
    print('----------------------------')

    #dbscan 3
    print('dbscan minPts = 5 PCA')
    db5_pca_t = time.time()
    dbscan_5_pca = DBSCAN(fe=fe_pca, data_=df.copy(), eps_value=eps_pca, min_pts=5)
    print('dbscan minPts 5 PCA time = ', time.time() - db5_pca_t)
    sse = sum(SSE(dbscan_5_pca.copy()).sse)
    print('eps value = ', eps_pca)
    print('sse = ', sse)
    print('shape = ', fe_pca.shape)
    print('----------------------------')

    
    

