import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from function.cleanText import CleanText
from function.findNumberPCA import featurePCA
from function.bestKmeans import best_k
from function.purityCluster import purity

from pythainlp.corpus import thai_stopwords
from pythainlp import sent_tokenize, word_tokenize


def identity_fun(text):
    return text

df = pd.read_csv('lineData/food_data.csv')
# df = pd.read_csv('dataset/TH1_csv.csv')
data = list(df['text'])

last_words = {"ครับ","คะ","ค่ะ","ค่า","คับ","ฮ่ะ"}
stopwords = thai_stopwords()
tmp = []

# for i in last_words:
#     tmp.append(i)

for i in stopwords:
    tmp.append(i)

stopwords = frozenset(tmp)


clean_text = []
token_word = []
for sen in data: 
        words = word_tokenize(sen)
        stop_words = [i for i in words if i not in stopwords]
        clean_text.append(' '.join(i for i in stop_words))

for sen in data:
        token_word.append(word_tokenize(sen))

tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)
feature_extraction = tf_idf.fit_transform(token_word).todense()

print(feature_extraction)

n_component = min(len(feature_extraction), len(tf_idf.get_feature_names()))

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

print(df)