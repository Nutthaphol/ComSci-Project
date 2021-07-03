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

        return data_

        