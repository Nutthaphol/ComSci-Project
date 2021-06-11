import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


from function.cleanTextEng import CleanText
from function.findNumberPCA import featurePCA
from function.bestKmeans import best_k
from function.purityCluster import purity

# import data
df = pd.read_csv('dataset/corona.csv')
data = list(df['data'])

# cleaning data
clean_data = CleanText(data)

# create tf-idf
tf_idf = TfidfVectorizer()
feature_extraction = tf_idf.fit_transform(clean_data).todense()

n_component = min(len(feature_extraction), len(tf_idf.get_feature_names()))

feature_pca = featurePCA(feature=feature_extraction, n_component=n_component)

k_value = best_k(feature=feature_extraction, max_=int(0.1*len(data)))

# start timing for clustering
start_time = time.time()

# start clustering
cluster = KMeans(n_clusters=k_value, max_iter=1000).fit(feature_extraction)
distence_point = cluster.transform(feature_extraction)

# end timing for clustering
timer_ = time.time() - start_time

df["centroids_id"] = cluster.labels_

comp = pd.crosstab(df['intent'], df['centroids_id'])
purity_ = purity(crosstab_=comp, size_data=len(data))
print("best k value = ", k_value)
print("purity = ", purity_)
print("timer = ", timer_)
