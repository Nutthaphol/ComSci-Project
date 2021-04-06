import pandas as pd
import numpy as np
import sys
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib import pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

#### import dataset
# df = pd.read_csv('dataset/atis_intents.csv')
# df = pd.read_csv('dataset/case_routing_intent.csv')
df = pd.read_csv('dataset/corona.csv')
# print("dataset\n", df)

intent = list(df['intent'])
# data = list(df['data'])
data = list(df['data'])

#### cleanning data and creating TF-IDF
stemmer = SnowballStemmer(language='english')
stop_word = stopwords.words('english')
clean_data = []

for sen in data:
    sen = ' '.join([word.lower() for word in sen.split(' ') if word not in stop_word])
    sen = re.sub(r'\'w+', ' ', sen)
    sen = re.sub('[%s]' % re.escape(string.punctuation), ' ', sen)
    sen = re.sub(r'\w*\d+\w*', ' ', sen)
    sen = re.sub(r'\s{2,}', ' ', sen)
    words = word_tokenize(sen)
    st_words = [i for i in words if i not in stop_word]
    clean_data.append(' '.join([stemmer.stem(i) for i in st_words]))

feature_ex = TfidfVectorizer()
tf_idf_ = feature_ex.fit_transform(clean_data).todense()
tf_idf_name_= feature_ex.get_feature_names()

#### find the bast number of components for create PCA
info_data = min(len(tf_idf_name_), len(clean_data))
pca = PCA(n_components=info_data)
pca.fit(tf_idf_)

sum_variance_ratio = np.sum(pca.explained_variance_ratio_)

number_component = 0

for i in pca.explained_variance_ratio_:
    number_component += 1
    if np.sum(pca.explained_variance_ratio_[0:number_component]) >= 0.8:
        break

print("best of component = ", number_component)

#### down dimention with  PCA from the best number of components 
pca = PCA(n_components=number_component).fit(tf_idf_)
features_pca = pca.transform(tf_idf_)
best_purity = -99
for r in range(2,10):
        
    neigh = NearestNeighbors(n_neighbors= 2)
    nn = neigh.fit(features_pca)
    distances, indices = nn.kneighbors(features_pca)

    distances = np.sort(distances, axis=0)
    indices = np.sort(indices, axis=0)
    distances = distances[:,1]
    # print(indices,"\n---------------------\n")
    # print(distances)
    max_ = np.max(distances)
    # print(max_)
    eps_ = 0
    for i in range(len(distances)):
        # print(distances[i]*100/max_)
        if (distances[i]*100/max_) >= 80:
            # print(distances[i]*100/max_)
            eps_ = distances[i]
            max_ = 1000

    eps_ = round(eps_, 1)
    print("eps_", eps_)
    # print("sum = ", sum)

    model_ = DBSCAN(eps=eps_, min_samples=r).fit(features_pca)
    labels_ = model_.labels_
    df["labels"] = labels_

    comp = pd.crosstab(df['intent'], df['labels'])
    max_comp_col = np.max(comp ,axis=0)
    sum_comp_col = np.sum(comp, axis=0)
    purity_score = max_comp_col/sum_comp_col
    purity_mean = np.mean(purity_score)
    print(r, ") mean of purity = ", purity_mean, "\n")

    if purity_mean > best_purity:
        best_purity = purity_mean
        min_ = r

# pts = []
# count = 0.1
# eps_ = 0
# min_ = 0
# while count < 1:
#     count = count + 0.1
#     count = round(count,2)
#     pts.append(count)
# print(pts)
# for i in range(3,10):
#     for j in pts:
#         model_ = DBSCAN(eps=j, min_samples=i).fit(features_pca)
#         labels_ = model_.labels_
#         df["labels"] = labels_
#         max_label_ = np.max(labels_)
        
#         comp = pd.crosstab(df['intent'], df['labels'])
#         max_comp_col = np.max(comp ,axis=0)
#         sum_comp_col = np.sum(comp, axis=0)
#         purity_score = max_comp_col/sum_comp_col
#         purity_mean = np.mean(purity_score)
#         # print(i, ") mean of purity = ", purity_mean, "\n")
#         # print("len(labels) = ", max_label_)
#         # print("len(data) = ",len(data))
#         if (purity_mean > 0.80) and max_label_ < (0.1 * len(data)):
#             best_purity = purity_mean
#             eps_ = j
#             min_ = i
    
print("the best purity = ", best_purity)
print("the best eps = ", eps_)
print("the best MinPts = ", min_, "\n\n** run ** \n")
    # print(pd.crosstab(df['intent'], df['labels']))

model_ = DBSCAN(eps=eps_, min_samples=min_).fit(features_pca)
labels_ = model_.labels_
df["labels"] = labels_
comp = pd.crosstab(df['intent'], df['labels'])
max_comp_col = np.max(comp ,axis=0)
sum_comp_col = np.sum(comp, axis=0)
purity_score = max_comp_col/sum_comp_col
purity_mean = np.mean(purity_score)
print("mean of purity = ", purity_mean, "\n")
print(pd.crosstab(df['intent'], df['labels']))



