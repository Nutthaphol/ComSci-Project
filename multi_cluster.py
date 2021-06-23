import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pythainlp import  word_tokenize
import math
from function.cleanTextTH import CleanText
from function.findNumberPCA import featurePCA
from function.bestKmeans import best_k

def identity_fun(text):
    return text

def cluster_kmean(feature):
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
                                tmp_value += np.linalg.norm(u[index_u] - fe[index_label])
                                counter += 1
                                score.append(tmp_value)
                # av_dis = tmp_value/counter
                # score.append(av_dis)

        col_ = {"label":centroids_id,"dist score":score}
        data_f = pd.DataFrame(data=col_)
        # print("label = ",centroids_id)
        # print("score = ",score)
        return data_f




        

df = pd.read_csv("lineData/comsci_data.csv") 

text =df.text

clean_text = CleanText(text)

tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(clean_text).todense()

# k_value = best_k(feature=fe, max_= int(len(clean_text) *0.2))

k_value = int(len(fe) * 0.2)

cluster = KMeans(n_clusters=k_value).fit(fe)

centroids_id = cluster.labels_

df["centroids_id_level_1"] = centroids_id

for i in range(len(fe)):
        if centroids_id[i] == 0:
                print(i,",")

u = cluster.cluster_centers_

score = []

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
av_score = np.mean(score)

next_level = []

for center_id in range(len(u)):
        if score[center_id] > av_score:
                next_level.append(center_id)

df["centroids_id_level_2"] = -1

for next_ in next_level:
        index_fe = np.where(centroids_id == next_)
        index_fe = list(index_fe[0])
        tmp_fe = fe.tolist()
        fe_next = []
        for i in index_fe:
                fe_next.append(tmp_fe[i])
        fe_next = np.matrix(fe_next)
        data_f = cluster_kmean(fe_next)
        # cluster_kmean(fe_next)
        