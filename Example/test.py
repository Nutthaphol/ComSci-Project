import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

from function.cleanText import CleanText
from function.word_embedding import averaged_word_vectorizer
from function.findNumberPCA import featurePCA
from function.purityCluster import purity
from function.findEPS import best_eps
from function.bestKmeans import best_k
from gensim.models import Word2Vec  # new feature extraction


df = pd.read_csv("dataset/corona.csv")
data = list(df['data'])

data = np.array(data)

data = CleanText(data)
# print(data)
tmp = []
for i in range(len(data)):
    text = word_tokenize(data[i])
    # print(text)
    tmp.append(text)

# str = ""
# for i in tmp:
#     for j in i:
#         str += j+" "

# num = word_tokenize(str)
# print(len(set(num)))

model = Word2Vec(tmp, min_count=1,size=600)

model[model.wv.word_vec(tmp, use_norm=True)]
feature = model[model.wv.vocab]

print(feature)
print(np.max(feature))
print(np.min(feature))

# print(feature.shape)
# word = model.wv.index2word
# result = model.wv[word]
# print(result.shape)

# feature = averaged_word_vectorizer(tmp, model, model.vector_size)

# eps_value = best_eps(feature)

# print(eps_value)

# # start clustering
# cluster = DBSCAN(eps=0.1, min_samples=100).fit(feature)

# labels_ = cluster.labels_

# set_ = set(labels_)

# print(set_)
# start_time = time.time()
# print(feature.shape)
print(np.max(feature))
print(np.min(feature))


# feature_pca = featurePCA(feature=feature, n_component=600)

# k_value = best_k(feature=feature_pca, max_=int(0.1*len(data)))


# cluster = KMeans(n_clusters=k_value).fit(feature_pca)
# dis_point = cluster.transform(feature_pca)

# label_ = cluster.labels_

# print(len(label_))

# df["centroids_id"] = cluster.labels_
# comp = pd.crosstab(df['intent'], df['centroids_id'])
# purity_ = purity(crosstab_=comp, size_data=len(data))
# print("best k value = ", 20)
# print("purity = ", purity_)
# timer_ = time.time() - start_time
# print(timer_)