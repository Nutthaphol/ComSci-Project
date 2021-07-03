from function.kmean_three_level import avg_dist
import numpy as np
import pandas as pd
import math

def SSE(data_):
        centroids_id = np.array(data_["centroids_id"].tolist())
        set_centroid_id = list(set(centroids_id))
        num_data_centroids_id = []
        sse = []

        for i in set_centroid_id:
                tmp = np.where(centroids_id == i)
                tmp = tmp[0].tolist()
                num_data_centroids_id.append(len(tmp))

                avg_dist = data_[data_["centroids_id"] == i]["avg_dist"].tolist()
                avg_dist = avg_dist[0]
                dist_score = data_[data_["centroids_id"] == i]["dist_score"].tolist()

                tmp_sse = []
                
                for i in dist_score:
                        pow_ = math.pow(i-avg_dist,2)
                        tmp_sse.append(pow_)
                
                sse.append(sum(tmp_sse))

        data_set = pd.DataFrame({"centroids_id":set_centroid_id,"number":num_data_centroids_id,"sse":sse})
