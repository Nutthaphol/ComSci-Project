import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from pythainlp import  word_tokenize


from function.cleanTextTH import CleanText
from function.findNumberPCA import featurePCA
from function.bestKmeans import best_k

def identity_fun(text):
    return text


# import data
df = pd.read_csv("lineData/comsci_data.csv")
raw_data = list(df['text'])

clean_text = CleanText(raw_data)

# for i in range(len(clean_text)):
#     print("No",i," ",clean_text[i])

# token_text = []
# for sen in clean_text:
#         token_text.append(word_tokenize(sen,engine='newmm'))


tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(clean_text).todense()

# n_component = min(len(fe), len(tf_idf.get_feature_names())) - 1

k_value = best_k(feature=fe, max_= int(len(clean_text) *0.2))

cluster = KMeans(n_clusters=k_value).fit(fe)
distence_point = cluster.transform(fe)

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

label = set(centroids_id)
label = list(label)

df["avg_dist_score"] = -1

for i in range(len(score)):
    df.at[df["centroids_id_level_1"] == i,"avg_dist_score"] = score[i] 

sum_score = sum(score)
av_score = np.mean(score)
print("sum_score", sum_score)
print("average score", av_score)



df = df.sort_values(by=['centroids_id_level_1'])    

# # print(df)

df.to_csv("cluster_comsci_kmean.csv",encoding='utf-8-sig')