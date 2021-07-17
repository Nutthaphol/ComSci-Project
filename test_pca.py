from function.purityCluster import purity
from function.dbscan import DBSCAN
from function.bestEps import bestEps
from function.kmean import Kmean
from function.bestKmeans import best_k
from function.cleanTextEng import CleanText
from function.tf_idf import TF_IDF
from function.bestPCA import bestPCA

import numpy as np
import pandas as pd
import time

if __name__ == '__main__':
    df = pd.read_csv('dataset/case_routing_intent.csv')

    text = df.text.tolist()

    clean_text = CleanText(text)

    fe = TF_IDF(text=clean_text, format='english')

    # pca feature
    n_component = min(fe.shape[0], fe.shape[1])
    pca_time = time.time()
    fe_pca = bestPCA(feature=fe, n_component=n_component)
    pca_time = time.time() - pca_time
    
    print('time use for make pca = ', pca_time)

    # format csv
    data_set = []
    pca_use = []
    static_value = []
    dynamic_value = []
    purity_score = []
    time_cluster = []
    

    # cluster with kmean normal feature
    k_value = best_k(feature=fe, max_= int(len(fe) *0.3))
    kmean_time = time.time()
    kmean = Kmean(fe=fe, data_=df.copy(), k_value= k_value)
    kmean_time = time.time() - kmean_time
    data_set.append('case_routing_intent')
    pca_use.append('no')
    static_value.append(np.NaN)
    dynamic_value.append(k_value)
    purity_score.append(purity(crosstab_=pd.crosstab(kmean.intent,kmean.centroids_id),size_data=len(clean_text)))
    time_cluster.append(kmean_time)   

    # cluster with kmean pca feature
    k_value_pca = best_k(feature=fe_pca, max_= int(len(fe_pca) *0.3))
    kmean_pca_time = time.time()
    kmean_pca = Kmean(fe=fe_pca, data_=df.copy(), k_value=k_value_pca)
    kmean_pca_time = time.time() - kmean_pca_time
    data_set.append('case_routing_intent')
    pca_use.append('yes')
    static_value.append(np.NaN)
    dynamic_value.append(k_value_pca)
    purity_score.append(purity(crosstab_=pd.crosstab(kmean_pca.intent,kmean_pca.centroids_id),size_data=len(clean_text)))
    time_cluster.append(kmean_pca_time)   

    # cluster with dbscan normal feature and minpts = 3
    eps = bestEps(fe)
    dbscan_3_n_t = time.time()
    dbscan = DBSCAN(fe=fe ,data_=df.copy(), eps_value=eps, min_pts=3)
    dbscan_3_n_t = time.time() - dbscan_3_n_t
    data_set.append('case_routing_intent')
    pca_use.append('no')
    static_value.append(3)
    dynamic_value.append(eps)
    purity_score.append(purity(crosstab_=pd.crosstab(dbscan.intent,dbscan.centroids_id),size_data=len(clean_text)))
    time_cluster.append(dbscan_3_n_t)   

    # cluster with dbscan pca feature and minpts = 3
    eps_pca = bestEps(fe_pca)
    dbscan_3_pca_t = time.time()
    dbscan_pca = DBSCAN(fe=fe_pca, data_=df.copy(), eps_value=eps_pca, min_pts=3)
    dbscan_3_pca_t = time.time() - dbscan_3_pca_t
    data_set.append('case_routing_intent')
    pca_use.append('yes')
    static_value.append(3)
    dynamic_value.append(eps_pca)
    purity_score.append(purity(crosstab_=pd.crosstab(dbscan_pca.intent,dbscan_pca.centroids_id),size_data=len(clean_text)))
    time_cluster.append(dbscan_3_pca_t)   

    # cluster with dbscan normal feature and minpts = 5
    eps = bestEps(fe)
    dbscan_5_n_t = time.time()
    dbscan = DBSCAN(fe=fe, data_=df.copy(), eps_value=eps, min_pts=5)
    dbscan_5_n_t = time.time() - dbscan_5_n_t
    data_set.append('case_routing_intent')
    pca_use.append('no')
    static_value.append(5)
    dynamic_value.append(eps)
    purity_score.append(purity(crosstab_=pd.crosstab(dbscan.intent,dbscan.centroids_id),size_data=len(clean_text)))
    time_cluster.append(dbscan_5_n_t)   
    
    # cluster with dbscan pca feature and minpts = 5
    eps_pca = bestEps(fe_pca)
    dbscan_5_pca_t = time.time()
    dbscan_pca = DBSCAN(fe=fe_pca, data_=df.copy(), eps_value=eps_pca, min_pts=5)
    dbscan_5_pca_t = time.time() - dbscan_5_pca_t
    data_set.append('case_routing_intent')
    pca_use.append('yes')
    static_value.append(5)
    dynamic_value.append(eps_pca)
    purity_score.append(purity(crosstab_=pd.crosstab(dbscan_pca.intent,dbscan_pca.centroids_id),size_data=len(clean_text)))
    time_cluster.append(dbscan_5_n_t)   
    
    final_df = pd.DataFrame({'data':data_set, 'pca':pca_use, 'static value':static_value,\
                                                'dynamic value':dynamic_value, 'purity':purity_score,\
                                                'time cluster':time_cluster })

    final_df.to_excel("stat/test_pca.xlsx", encoding='utf-8-sig')