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

df = pd.read_csv("lineData/comsci_data.csv")

loop = 1

raw_data = list(df['text'])

clean_text = CleanText(raw_data)

avg_dist = []

comput_next = []

for level in range(0,2):
        for i in clean_text:
                centroids_id.append(-1)
        for num_cluster in range(0,loop):
                data = []
                if level == 0:
                        data = clean_text
                        tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

                        fe = tf_idf.fit_transform(data).todense()
                        maxK = (len(data) *0.2)
                        maxK = math.ceil(maxK)
                        if maxK < 2:
                                continue
                        k_value = best_k(feature=fe, max_= maxK)
                        print("K value",k_value)
                        cluster = KMeans(n_clusters=k_value).fit(fe)
                        centroids_id = cluster.labels_
                        df["centroids_id_level_1"] = centroids_id
                                
                        u = cluster.cluster_centers_
                        score = []
                        tmp_label = set(centroids_id)

                        for index_u in range(len(u)):
                                tmp_value = 0
                                counter = 0
                                for index_label in range(len(centroids_id)):
                                        if centroids_id[index_label] == index_u:
                                                tmp_value += np.linalg.norm(u[index_u] - fe[index_label])
                                                counter += 1
                                av_dis = tmp_value/counter
                                score.append(av_dis)
                        
                        # if len(avg_dist) == 0:
                        #         for i in centroids_id:
                        #                 avg_dist.append(score[i])
                        #         df["avg_dist"] = avg_dist
                        
                        sum_score = sum(score)
                        avg_score = np.mean(score)
                        print("sum_score", sum_score)
                        print("average score", avg_score)
                        for i in range(len(score)):
                                print("score[{}] = {}".format(i,score[i]))
                                if score[i] > avg_score:
                                        comput_next.append(i)
                        print(comput_next)
                        loop = len(comput_next)
                        
                elif level == 1:
                        for i in range(len(clean_text)):
                                if int(df.iloc[i]["centroids_id_level_1"] ) == comput_next[num_cluster]:
                                        data.append(clean_text[i])
                        print(comput_next[num_cluster])
                        print(data)
                        

                # print(clean_text)
                
                print("--------------")
        print(centroids_id)

# print(df.iloc[1])
# comput_next.pop()
# print(comput_next)