import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import math
import string

pd.set_option('display.max_rows', 5000)              
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)

# df = pd.read_csv('dataset/atis_intents.csv')        
df = pd.read_csv('dataset/case_routing_intent.csv')
# df = pd.read_csv('dataset/english.csv')

intent = list(df['intent'])                         
data = list(df['data'])

ss = SnowballStemmer(language='english')
stop_word = stopwords.words('english')        
clean_data = []
dicts = []                                     

for sen in data:
    sen = ' '.join([word.lower() for word in sen.split(' ') if word not in stop_word])
    sen = re.sub(r'\'w+', ' ', sen)
    sen = re.sub('[%s]' % re.escape(string.punctuation), ' ', sen)
    sen = re.sub(r'\w*\d+\w*', ' ', sen)
    sen = re.sub(r'\s{2,}', ' ', sen)
    words = word_tokenize(sen)
    st_words = [i for i in words if i not in stop_word]
    clean_data.append(' '.join([ss.stem(i) for i in st_words]))
    dicts.append(st_words)

# print(df.intent.value_counts().sort_index())                      

vt = TfidfVectorizer()

feature_vector = vt.fit_transform(data).todense()

# print(feature_vector.shape[1])
# # print(f'\nfeature_vector shape before PCA {feature_vector.shape}\n')

# all_dim = feature_vector.shape[1]

# print("all dimention is ", all_dim)

# pca = PCA(n_components=len(data)).fit(feature_vector)                               

# variance_num = []
# sum_variance = 0
# for value in pca.explained_variance_:
#     variance_num.append(value)
#     sum_variance += value


# # plt.plot(range(len(pca.explained_variance_)), variance_num, color='g', linewidth='3')
# # plt.show()  # clear the plot

# per_variance = []
# for i in pca.explained_variance_:
#     per_variance.append(i/sum_variance*100)

# per_variance_cum = np.cumsum(per_variance)     
# components = 0
# for value in per_variance_cum:
#     components += 1
#     if value > 80:
#         break

# print("Component = ", components)
# pca = PCA(n_components = components).fit(feature_vector)  
# features_matrix = pca.transform(feature_vector)    


# print("feature matrix is \n", features_matrix)

# num_sil = []
# # max_k = all_dim
# max_k = 150
# # max_k = int(0.1 * df.data.count())
# print ("number of data = ", max_k)

# for k in range(2, max_k+1):
#     kmean = KMeans(n_clusters=k).fit(features_matrix)
#     label = kmean.labels_
#     num_sil.append(silhouette_score(features_matrix, label, metric='euclidean'))

# best_k = 0
# count = 2
# for sil in num_sil:
#     if sil > best_k:
#         best_k = sil
#         pointer = count
#     count +=1 

# print("The best K value = ", best_k)
# print("point of the best K value = ", pointer)

# # plot the cost against K values
# plt.plot(range(2, max_k+1), num_sil, color='g', linewidth='3')
# plt.show()  # clear the plot

########    END FIND BEST K VALUE ############
#
# model = KMeans(n_clusters=150) 
#
# # data_model = model.fit(normal)  
# data_model = model.fit(normal_matrix)  
#
# print(f'\ndata_model {data_model}')
#
# labels = model.labels_  
#
#
#
# print(f'Labels = {labels}\n')
#
# centroid = model.cluster_centers_   # centroid of KMean model
# # centroid_matrix = pca.transform(centroid)
# df['cluster'] = labels  # add labels to dataframe
# # plt.scatter(x = normal_matrix[:, 0], y = normal_matrix[:, 1], marker="o", alpha=0.2)     
# # plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', color='red')                      
# # plt.show()
#
# print(f'\ncentroid data = \n {centroid}')
#
# ct_tmp = pd.crosstab(df['intent'], df['cluster'])
# print(f"type of crosstab is {type(ct_tmp)}")
# # print(pd.crosstab(df['intent'], df['cluster']))   
