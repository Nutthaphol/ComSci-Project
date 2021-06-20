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

def nextLevel (data):
        text = data['text']
        # print("in data = ",data["centroids_id_level_1"].head(1))
        clean_data = CleanText(text)

        tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

        fe = tf_idf.fit_transform(clean_data).todense()

        k_value = best_k(feature=fe, max_= int(math.ceil((len(clean_data) *0.2))))

        if k_value < 2:
                k_value = 2

        # print("k value", k_value)

        cluster = KMeans(n_clusters=k_value).fit(fe)

        centroids_id = cluster.labels_
        # print(centroids_id)
        col = ["centroids_id"]
        df = pd.DataFrame(centroids_id,columns=col)

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
                # print(av_dis)

        sum_score = sum(score)
        av_score = np.mean(score)

        dist = []

        for i in centroids_id:
                dist.append(score[i])
        
        df["distance"] = dist

        # print("sum_score", sum_score)
        # print("average score", av_score)
        # print("________________________________")

        return df


# import data
df = pd.read_csv("lineData/comsci_data.csv")
# df["id"] = range(len(df))
raw_data = list(df['text'])

clean_text = CleanText(raw_data)

tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(clean_text).todense()

k_value = best_k(feature=fe, max_= int(len(clean_text) *0.2))

cluster = KMeans(n_clusters=k_value).fit(fe)

centroids_id = cluster.labels_

df["centroids_id_level_1"] = centroids_id

df = df.sort_values(by=['centroids_id_level_1'])    

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

sum_score = sum(score)
av_score = np.mean(score)
print("sum_score", sum_score)
print("average score", av_score)

next_ = []

for i in range(len(score)):
        print(i,",",score[i])
        if (score[i] > av_score):
                next_.append(i)

print("next",next_)

label = set(centroids_id)
label = list(label)

label = set(centroids_id)
label = list(label)

level_2 = []

for index in label:
        if index not in next_:
                tmp = df[df.centroids_id_level_1 == index]
                for i in range(len(tmp)):
                        level_2.append(-1)
        else:
                data = df[df.centroids_id_level_1 == index]
                dataF = nextLevel(data)
                tmp = list(dataF['centroids_id'])
                for i in tmp:
                        level_2.append(i)

# print(level_2)
df["centroids_id_level_2"] = level_2
print(df)

df.to_csv("cluster_comsci_multi_kmean.csv",encoding='utf-8-sig')
# for i in range(len(next_)):
#         data = df[df.centroids_id_level_1 == next_[i]]
#         dataF = nextLevel(data)
#         print("id = {} , data = \n{}".format(next_[i], dataF))



