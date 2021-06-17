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

for i in range(len(clean_text)):
    print("No",i," ",clean_text[i])

# token_text = []
# for sen in clean_text:
#         token_text.append(word_tokenize(sen,engine='newmm'))


tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(clean_text).todense()

# print(fe.shape)

# n_component = min(len(fe), len(tf_idf.get_feature_names()))

k_value = best_k(feature=fe, max_=30)

cluster = KMeans(n_clusters=k_value).fit(fe)
distence_point = cluster.transform(fe)

df["centroids_id"] = cluster.labels_

label = cluster.labels_
u = cluster.cluster_centers_

num_label = set(label)

# print("center shape = ", u.shape)
# print("feature shape = ", fe.shape)
# print("label shape = ", len(num_label))
# print("size of data = ", len(clean_text))

score = []

for center in range(len(u)):
    number = 0
    dis = 0
    text_ = list(df["text"])
    label_ = list(df["centroids_id"])
    for point in range(len(label_)):
        if label_[point] == center:
            number += 1
            dis += np.linalg.norm(u[center] - fe[point])
    # print("dis = ", dis, ", number = ", number)
    dis = dis/number
    score.append(dis)

for i in range(len(score)):
    print("label = ", i ,", average distance of all data point in label = ", score[i])

sum_score = 0
for i in score:
    sum_score += i
print("sum_score", sum_score)
print("average score", sum_score/len(score))
        
# dist = np.linalg.norm(u[1] - fe[1])

# print("dist", dist)

# df["distances"] = cluster.inertia_

df = df.sort_values(by=['centroids_id'])

# print(df)

df.to_csv("cluster_comsci_kmean.csv",encoding='utf-8-sig')