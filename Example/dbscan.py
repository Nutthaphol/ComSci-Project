import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

from function.cleanTextEng import CleanText
from function.findNumberPCA import featurePCA
from function.purityCluster import purity
from function.findEPS import best_eps
from gensim.models import Word2Vec  # new feature extraction

# import data
# df = pd.read_csv('dataset/atis_intents.csv')
df = pd.read_csv('dataset/corona.csv')
# df = pd.read_csv('dataset/case_routing_intent.csv')
data = list(df['data'])

# cleaning data
clean_data = CleanText(data)

# create tf-idf  
tf_idf = TfidfVectorizer() 
feature_extraction = tf_idf.fit_transform(clean_data).todense()

# print(feature_extraction)

for i in (feature_extraction):
        print(i)



# n_component = min(len(feature_extraction), len(tf_idf.get_feature_names()))

# feature_pca = featurePCA(feature=feature_extraction, n_component=n_component)

# eps_value = best_eps(feature_pca)

# # start timing for clustering
# start_time = time.time()

# # start clustering
# cluster = DBSCAN(eps=eps_value, min_samples=5).fit(feature_pca)

# # end timing for clustering
# timer_ = time.time() - start_time

# labels_ = cluster.labels_
# df["labels"] = labels_

# df = df[df.labels != -1]

# comp = pd.crosstab(df['intent'], df['labels'])
# print("comp\n", comp)
# purity_ = purity(crosstab_=comp, size_data=len(df))
# print("best eps = ", eps_value)
# print("purity = ", purity_)
# print("timer = ", timer_)