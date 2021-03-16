import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
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

# คำสั่ง explained_variance_ คือการเรียกดู Eigenvalue
# คำสั่ง explained_variance_ratio_ คือ ดูว่าแทนได้กี่เปอร์เซ็น

# df = pd.read_csv('dataset/atis_intents.csv')        
df = pd.read_csv('dataset/case_routing_intent.csv')
# df = pd.read_csv('dataset/english.csv')

intent = list(df['intent'])                         
data = list(df['data'])

print(df.intent.value_counts())
print("data size = ", len(data))

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

# print(df.intent.value_counts().sort_index())                      
vt = TfidfVectorizer()

feature_vector = vt.fit_transform(clean_data).todense()
dicts = vt.get_feature_names()
print("vcap\n", list(vt.vocabulary_.items()))
print(feature_vector, "\n")
print(dicts, "\n")
print("size of features = ", len(dicts), "\n")

info_data = min(len(vt.get_feature_names()), len(clean_data))

print("info_data = ", info_data)

pca = PCA(n_components=info_data)
pca.fit_transform(feature_vector)

sum_variance_ratio = np.sum(pca.explained_variance_ratio_)

number_component = 0

### Find number of component that >= 80%
for i in pca.explained_variance_ratio_:
    number_component += 1
    if np.sum(pca.explained_variance_ratio_[0:number_component]) >= 0.8:
        break

print("best of component = ", number_component)
plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='g', linewidth='3')
plt.show()  # clear the plot

# print("the best component = ", components)
# print("Component = ", components)
# pca = PCA(n_components = components).fit(feature_vector)  
# features_matrix = pca.transform(feature_vector)    


# print("feature matrix is \n", features_matrix)

#########  Find K value ###########
# num_sil = []
# # max_k = all_dim
# max_k = 20
# # max_k = int(0.1 * df.data.count())
# print ("number of data = ", max_k)

# for k in range(2, max_k+1):
#     kmean = KMeans(n_clusters=k).fit(features_matrix)
#     kmean.predict(features_matrix)
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

# ########    END FIND BEST K VALUE ############

