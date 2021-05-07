import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from nltk.tokenize import word_tokenize

from function.cleanText import CleanText
from function.findNumberPCA import featurePCA
from function.purityCluster import purity
from function.findEPS import best_eps
from gensim.models import Word2Vec  # new feature extraction


df = pd.read_csv("dataset/corona.csv")
data = list(df['data'])

data = np.array(data)
# print(data)
tmp = []
for i in range(len(data)):
    text = word_tokenize(data[i])
    print(text)
    tmp.append(text)

# print(data)
# model = Word2Vec(tmp, min_count=1)
# # model = Word2Vec(sentences=data, min_count=1)

# result = model[model.wv.vocab]

# print(result)

# cluster = DBSCAN(eps=0.9, min_samples=5).fit(result)

# labels_ = cluster.labels_

# print(labels_)
# df["labels"] = labels_

# df = df[df.labels != -1]

# comp = pd.crosstab(df['intent'], df['labels'])
# print("comp\n", comp)
# purity_ = purity(crosstab_=comp, size_data=len(df))
# print("best eps = ", eps_value)
# print("purity = ", purity_)
# # print("timer = ", timer_)