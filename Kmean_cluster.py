import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

token_text = []
for sen in clean_text:
        token_text.append(word_tokenize(sen))


tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                   tokenizer=identity_fun, #does no extra tokenizing
                                   preprocessor=identity_fun, #no extra preprocessor
                                   token_pattern=None)

fe = tf_idf.fit_transform(token_text).todense()

n_component = min(len(fe), len(tf_idf.get_feature_names()))

k_value = best_k(feature=fe, max_=10)

cluster = KMeans(n_clusters=k_value, max_iter=1000).fit(fe)
distence_point = cluster.transform(fe)

df["centroids_id"] = cluster.labels_

df = df.sort_values(by=['centroids_id'])

print(df)