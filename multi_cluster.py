from re import L
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from function.cleanTextTH import CleanText
from function.bestKmeans import best_k

def identity_fun(text):
    return text

def cluster_kmean(feature):
        #  feature > 2 for find k value with beat_k
        k_value = best_k(feature=feature, max_=max(2,int(len(feature)*0.5)))
        cluster = KMeans(n_clusters=k_value).fit(feature)
        centroids_id = cluster.labels_
        u = cluster.cluster_centers_
        score = []
        for index_u in range(len(u)):
                tmp_value = 0
                counter = 0
                for index_label in range(len(centroids_id)):
                        if centroids_id[index_label] == index_u:
                                tmp_value += np.linalg.norm(u[index_u] - feature[index_label])
                                counter += 1
                                score.append(tmp_value)

        col_ = {"label":centroids_id,"dist score":score}
        data_f = pd.DataFrame(data=col_)
        return data_f


def avg_dist(score):
        mean_score = np.mean(score)
        next_ = []
        for i in range(len(score)):
                if score[i] > mean_score:
                        next_.append(i)
        return next_
        

df = pd.read_csv("lineData/comsci_data.csv") 

text =df.text

clean_text = CleanText(text)

tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(clean_text).todense()

k_value = best_k(feature=fe, max_= int(len(fe) *0.2))

# k_value = int(len(fe) * 0.2)

cluster = KMeans(n_clusters=k_value).fit(fe)

centroids_id = cluster.labels_

# create col in datafream
df["centroids_id_level_1"] = centroids_id
df["centroids_id_level_2"] = -1
df["centroids_id_level_3"] = -1
df["avg_dist_score"] = -1
df["avg_dist_score_from_level"] = "level 1"

u = cluster.cluster_centers_ #point of center

score = []

#find average distance of each data to its center
for index_u in range(len(u)):
        tmp_value = 0
        counter = 0
        for index_label in range(len(centroids_id)):
                if centroids_id[index_label] == index_u:
                        tmp_value += np.linalg.norm(u[index_u] - fe[index_label])
                        counter += 1
        av_dis = tmp_value/counter
        score.append(av_dis)

sum_score = sum(score)
avg_score = np.mean(score)

for i in range(len(score)):
        df.at[df["centroids_id_level_1"] == i,"avg_dist_score"] = score[i] 

# next level
level_two = []
level_three = []

# if the score distance of any centroids id > avg_score to save id to cluster next level
for center_id in range(len(u)):
        if score[center_id] > avg_score:
                level_two.append(center_id)

''' start cluster level 2 '''
for next_ in level_two:
        index_fe = np.where(centroids_id == next_) #select all index  with same centroids id
        index_fe = list(index_fe[0]) #tranform tuple to list 
        tmp_fe = fe.tolist() # select feature 
        fe_next = []
        for i in index_fe: #select feature every feature used in cluster level 2 from matching index
                fe_next.append(tmp_fe[i])
        if len(fe_next) <= 2:     #  feature > 2 for find k value with silhouette_score in function beat_k
                continue
        fe_next = np.matrix(fe_next)

        data_f = cluster_kmean(fe_next) # cluster level 2 return datafream label and score distance of each feature
        centroids_id_next = data_f["label"] 
        dist_score = data_f["dist score"]

        # compute average of each label
        set_label = list(set(list(centroids_id_next)))
        score = []
        for i in set_label:
                tmp = dist_score[centroids_id_next == i].tolist()
                score.append((sum(tmp))/len(tmp))
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
                df.at[index_fe[i],"centroids_id_level_2"] = centroids_id_next[i]
                df.at[index_fe[i],"avg_dist_score_from_level"] = "level 2"
                df.at[index_fe[i],"avg_dist_score"] = score[centroids_id_next[i]]

''' start cluster level 3'''
for index_fe in level_three:
        tmp_fe = fe.tolist()  # select feature 
        fe_next = []
        for i in index_fe:      # select feature every feature used in cluster level 2 from matching index
                fe_next.append(tmp_fe[i])
        if len(fe_next) <= 2:     #  feature > 2 for find k value with silhouette_score in function beat_k
                continue
        fe_next = np.matrix(fe_next)

        data_f = cluster_kmean(fe_next) # cluster level 2 return datafream label and score distance of each feature
        centroids_id_next = data_f["label"] 
        dist_score = data_f["dist score"]

        # compute average of each label
        set_label = list(set(list(centroids_id_next)))
        score = []
        for i in set_label:
                tmp = dist_score[centroids_id_next == i].tolist()
                score.append((sum(tmp))/len(tmp))

        for i in range(len(index_fe)): # set centroids id level 2 at index from line 113 
                df.at[index_fe[i],"centroids_id_level_3"] = centroids_id_next[i]
                df.at[index_fe[i],"avg_dist_score_from_level"] = "level 3"
                df.at[index_fe[i],"avg_dist_score"] = score[centroids_id_next[i]]

tmp_df = df.sort_values(by=['centroids_id_level_1'])    
tmp_df.to_csv("cluster_comsci_multi_kmean.csv",encoding='utf-8-sig')


