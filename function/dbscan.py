import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN as dbsacn
from function.bestEps import bestEps

def DBSCAN(fe,data_):
        eps_value = bestEps(fe)

        cluster = dbsacn(eps=eps_value,min_samples=3).fit(fe)

        centroids_id = cluster.labels_

        data_["centroids_id"] = centroids_id
        data_["dist_score"] = -1

        set_centroids_id = list(set(centroids_id))

        for i in set_centroids_id:
                if i == -1:
                        continue
                pick_fe = fe[centroids_id == i]
                mean_of_cluster = np.mean(pick_fe, axis=0)
                index_ = np.where(centroids_id == i)
                index_ = index_[0].tolist()
                for index in index_:
                        distance_ = np.linalg.norm(fe[index] - mean_of_cluster)
                        data_.at[data_.index == index,"dist_score"] = distance_
                
               
        return data_

        