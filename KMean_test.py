import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import  word_tokenize

pd.set_option('display.max_rows', 5000)         # set option pandas for show data formal not ...
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)

df = pd.read_csv('dataset/atis_intents.csv')    # loading dataset

intent = list(df['intent'])     # import dataset to list type
data = list(df['data'])

stemmer = PorterStemmer()
stop_word = set(stopwords.words('english'))      # function for remove stopwords
clean_data = []     # list for keep data after making a cleaning dataset
for i in data:
    i = i.lower()   # set all character to lower font
    i = re.sub('[0-9]'," ", i)   # remove numbering from sentence
    # tokens = [word for word in word_tokenize(i) if len(word) > 1]  # Split word
    tokens = word_tokenize(i)  # Split word
    stems = [stemmer.stem(item) for item in tokens]     # make stemming word
    clean_data.append(' '.join(i for i in stems))   # add word to sentence

cluster = 22    # set value of cluster center

print(df.intent.value_counts().sort_index())    # show number of data in each intent

vt = TfidfVectorizer(stop_words="english")  # call object Tf-idf for transform data

normal = vt.fit_transform(clean_data).todense()     # transform data to tf-idf

print(f'\nshape before to dense {normal.shape}\n')

print(f'\nFeature name is \n {vt.get_feature_names()}')

pca = PCA(n_components=464).fit(normal)   # call object pac for down dimensional to 2d

print(f'eigen vector = \n{pca.components_}\n-------------------\n')

normal_matrix = pca.transform(normal)    # transform matrix to 2 dimension

print(f'{normal_matrix[:, 0]}\n---------------------\n')    # show data
print(f'{normal_matrix[:, 1]}\n---------------------\n')
print(f'shape {normal_matrix.shape}')
print(f'vector\n{normal_matrix}')

# plt.scatter(x = normal_matrix[:, 0], y = normal_matrix[:, 1], marker="o", alpha=0.2, )   # plot scattering of data
# plt.show()

print(f'\neigen value \n{pca.explained_variance_}')

model = KMeans(n_clusters=cluster)  # create KMean model

# data_model = model.fit(normal)  # train model
data_model = model.fit(normal_matrix)  # train model

print(f'\ndata_model {data_model}')

labels = model.labels_  # labels of data from clustering

print(f'Labels = {labels}\n')

centroid = model.cluster_centers_   # centroid of KMean model
# centroid_matrix = pca.transform(centroid)
df['cluster'] = labels  # add labels to dataframe
# plt.scatter(x = normal_matrix[:, 0], y = normal_matrix[:, 1], marker="o", alpha=0.2)     # plot scattering of data and
# plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', color='red')                      # centroid point
# plt.show()

print(f'\ncentroid data = \n {centroid}')

print(pd.crosstab(df['intent'], df['cluster']))     # crosstab
