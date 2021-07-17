import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from function.bestKmeans import best_k

def identity_fun(text):
    return text

def cluster_kmean(feature):
        k_value = 2
        #  feature > 2 for find k value with beat_k
        if len(feature) >= 2:
                k_value = best_k(feature=feature, max_=max(2,int(len(feature)*0.5)))
        cluster = KMeans(n_clusters=k_value).fit(feature)
        centroids_id = cluster.labels_
        u = cluster.cluster_centers_
        data_f = pd.DataFrame({"label":centroids_id})
        data_f["dist_score"] = -1
        for index_u in range(len(u)):
                for index_label in range(len(centroids_id)):
                        if centroids_id[index_label] == index_u:
                                data_score = np.linalg.norm(u[index_u] - feature[index_label])
                                data_f.at[data_f.index == index_label,"dist_score"] = data_score
        return data_f


def avg_dist(score):
        mean_score = np.mean(score)
        next_ = []
        for i in range(len(score)):
                if score[i] > mean_score:
                        next_.append(i)
        return next_

def Kmean_three_level(fe,data_):
        k_value = best_k(feature=fe, max_= int(len(fe) *0.2))

        cluster = KMeans(n_clusters=k_value).fit(fe)

        centroids_id = cluster.labels_

        # create col in datafream
        data_["centroids_id_level_1"] = centroids_id
        data_["centroids_id_level_2"] = -1
        data_["centroids_id_level_3"] = -1
        data_["dist_score"] = -1
        data_["avg_dist_score"] = -1
        data_["avg_dist_score_from_level"] = "level 1"

        u = cluster.cluster_centers_ #point of center

        score = []

        #find average distance of each data to its center
        for index_u in range(len(u)):
                tmp_value = []
                for index_label in range(len(centroids_id)):
                        if centroids_id[index_label] == index_u:
                                value_dist = np.linalg.norm(u[index_u] - fe[index_label])
                                tmp_value.append(value_dist) 
                                data_.at[data_.index == index_label, "dist_score"] = value_dist
                avg_dist = np.mean(tmp_value)
                score.append(avg_dist)
                data_.at[data_["centroids_id_level_1"] == index_u,"avg_dist_score"] = avg_dist

        avg_score = np.mean(score)

        # next level
        level_two = []
        level_three = []

        # if the score distance of any centroids id > avg_score to save id to cluster next level
        for center_id in range(len(u)):
                if score[center_id] > avg_score:
                        level_two.append(center_id)

        ''' start cluster level 2 '''
        # **optimized geting feature of each label (fe_next = fe[centroids_id == next_])
        for next_ in level_two:
                index_fe = np.where(centroids_id == next_) #select all index  with same centroids id
                index_fe = list(index_fe[0]) #tranform tuple to list 

                fe_next = fe[centroids_id == next_] # get each feature with centroids_id = next_
                 
                if len(fe_next) <= 2:     #  feature > 2 for find k value with silhouette_score in function beat_k
                        continue

                data_f = cluster_kmean(fe_next) # cluster level 2 return datafream label and score distance of each feature
                centroids_id_next = data_f["label"] 
                dist_score = data_f["dist_score"]

                # compute average of each label
                set_label = list(set(list(centroids_id_next)))
                score = []
                for i in set_label:
                        tmp = dist_score[centroids_id_next == i].tolist()
                        score.append(np.mean(tmp))
                avg_score = np.mean(score)

                # compute index for cluster in level three 
                for i in set_label:
                        if score[i] > avg_score:
                                tmp_index = np.array(list(centroids_id_next))
                                tmp_index = np.where(tmp_index == i)    # get index of centroid id level 2 than have score > avg 
                                tmp_index = list(tmp_index[0])
                                to_level_three = []
                                for i in tmp_index:
                                        to_level_three.append(index_fe[i]) # compare between index of centroid id level 2 with index of data 
                                if len(to_level_three) > 1:
                                        level_three.append(to_level_three)

                for i in range(len(index_fe)): # set centroids id level 2 at index from line 113 
                        data_.at[data_.index == index_fe[i],"centroids_id_level_2"] = centroids_id_next[i]
                        data_.at[data_.index == index_fe[i],"avg_dist_score_from_level"] = "level 2"
                        data_.at[data_.index == index_fe[i],"avg_dist_score"] = score[centroids_id_next[i]]
                        data_.at[data_.index == index_fe[i],"dist_score"] = data_f["dist_score"][i]


        ''' start cluster level 3'''
        for index_fe in level_three:
                if len(index_fe) <= 2:     #  feature > 2 for find k value with silhouette_score in function beat_k
                        continue
                
                tmp_fe = fe.tolist()  # select feature 
                fe_next = []
                for i in index_fe:      # select feature every feature used in cluster level 2 from matching index
                        fe_next.append(tmp_fe[i])
                fe_next = np.matrix(fe_next)

                data_f = cluster_kmean(fe_next) # cluster level 2 return datafream label and score distance of each feature
                centroids_id_next = data_f["label"] 
                dist_score = data_f["dist_score"]

                # compute average of each label
                set_label = list(set(list(centroids_id_next)))
                score = []
                for i in set_label:
                        tmp = dist_score[centroids_id_next == i].tolist()
                        score.append(np.mean(tmp))

                for i in range(len(index_fe)): # set centroids id level 2 at index from line 113 
                        data_.at[data_.index == index_fe[i],"centroids_id_level_3"] = centroids_id_next[i]
                        data_.at[data_.index == index_fe[i],"avg_dist_score_from_level"] = "level 3"
                        data_.at[data_.index == index_fe[i],"avg_dist_score"] = score[centroids_id_next[i]]
                        data_.at[data_.index == index_fe[i],"dist_score"] = data_f["dist_score"][i]


        data_["centroids_id"] = -1
        pattern = []
        for i in range(len(data_)):
                tmp = []                
                tmp.append(data_["centroids_id_level_1"].loc[i])
                tmp.append(data_["centroids_id_level_2"].loc[i])
                tmp.append(data_["centroids_id_level_3"].loc[i])
                if tmp in pattern:
                        continue
                pattern.append(tmp)

        for i in range(len(pattern)):
                set_pattern = pattern[i]
                data_.at[(data_.centroids_id_level_1 == set_pattern[0] ) & \
                                (data_.centroids_id_level_2 == set_pattern[1]) & \
                                (data_.centroids_id_level_3 == set_pattern[2]), "centroids_id"] = i

        return data_
