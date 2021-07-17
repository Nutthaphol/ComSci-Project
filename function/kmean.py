from sklearn.cluster import KMeans
from function.bestKmeans import best_k

import numpy as np


def Kmean(fe,data_,k_value = None):
        if k_value == None:
                k_value = best_k(feature=fe, max_= int(len(fe) *0.3))
        
        cluster = KMeans(n_clusters=k_value).fit(fe)

        centroids_id = cluster.labels_
        
        data_["centroids_id"] = centroids_id
        data_["dist_score"] = -1
        data_["avg_dist_score"] = -1
        
        u = cluster.cluster_centers_

        for index_u in range(len(u)):
                mix = np.where(centroids_id == index_u)
                mix = mix[0].tolist()
                tmp_value = []
                for i in mix:
                        value_dist = np.linalg.norm(u[index_u] - fe[i])
                        tmp_value.append(value_dist)
                        data_.at[data_.index == i, "dist_score"] = value_dist
                avg_dist = np.mean(tmp_value)
                data_.at[data_["centroids_id"] == index_u, "avg_dist_score"] = avg_dist
        
        # tm = data_
        # tm = tm.sort_values(by=['centroids_id'])
        # tm.to_csv("tmp_show.csv",encoding='utf-8-sig')
        return data_
