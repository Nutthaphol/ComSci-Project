import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize

from function.cleanText import CleanText
from function.word_embedding import averaged_word_vectorizer
from function.findNumberPCA import featurePCA
from function.purityCluster import purity
from function.findEPS import best_eps
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

model = Word2Vec(tmp, min_count=1)

word = model.wv.index2word
result = model.wv[word]

feature = averaged_word_vectorizer(tmp, model, model.vector_size)

# eps_value = best_eps(feature)


# start clustering
cluster = DBSCAN(eps=0.1, min_samples=100).fit(feature)

labels_ = cluster.labels_

set_ = set(labels_)

print(set_)