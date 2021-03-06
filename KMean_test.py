import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import  word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import math

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
#
# print(df.intent.value_counts().sort_index())    # show number of data in each intent

vt = TfidfVectorizer(stop_words="english")  # call object Tf-idf for transform data

normal = vt.fit_transform(clean_data).todense()     # transform data to tf-idf

print(f'\nshape before to dense {normal.shape}\n')
#
# print(f'\nFeature name is \n {vt.get_feature_names()}')

pca = PCA(n_components=464).fit(normal)   # in line 43 - 65 is algorithm for find the best componnt value from algorithm for find the mean

y_label = []
count = 0
sum = 0
for value in pca.explained_variance_:
    y_label.append(value)
    sum += value

plt.plot(range(len(pca.explained_variance_)), y_label, color='g', linewidth='3')
plt.show()  # clear the plot


tmp = sum/100
per_var = []
for i in pca.explained_variance_:
    per_var.append(i/tmp)

av = sum/464
sum = 0

for i in pca.explained_variance_:
    # print(f'i = {i}')       # i not a number 1, 2, 3.... It's value of parameter
    sum += math.pow(i - av, 2)
    
sd = math.sqrt(sum/(464-1))

mdn = pca.explained_variance_[231] - pca.explained_variance_[232]
mdn /= 2

print(f'\nmax = {pca.explained_variance_[0]}')
print(f'min = {pca.explained_variance_[463]}')
print(f'max-min = {pca.explained_variance_[0] - pca.explained_variance_[463]}')
print(f'median = {mdn}')
print(f'mean = {av}')
print(f'sd = {sd}')
# last = sd*2 + mdn
last = pca.explained_variance_[0] - (av+(2*sd))
print(f'last = {last}')
point = 0
for value in pca.explained_variance_:
    point += 1
    if value < last:
        print(f'break point = {value}')
        break



print(f'point = {point}')


pca = PCA(n_components=point).fit(normal)   # call object pac for down dimensional to 2d
normal_matrix = pca.transform(normal)    # transform matrix to 2 dimension
# print(f'{normal_matrix[:, 0]}\n---------------------\n')    # show data
# print(f'{normal_matrix[:, 1]}\n---------------------\n')
# print(f'shape {normal_matrix.shape}')
# print(f'vector\n{normal_matrix}')

# plt.scatter(x = normal_matrix[:, 0], y = normal_matrix[:, 1], marker="o", alpha=0.2, )   # plot scattering of data
# plt.show()

# print(f'\neigen value \n{pca.explained_variance_}')

cost = []
# for i in range(1, 400):                           # in line 94 - 104 is algorithm 1 th for find the best k values
#     KM = KMeans(n_clusters=i, max_iter=500)
#     KM.fit(normal_matrix)
#
#     # calculates squared error
#     # for the clustered points
#     cost.append(KM.inertia_)
#
# # plot the cost against K values
# plt.plot(range(1, 400), cost, color='g', linewidth='3')
# plt.show()  # clear the plot

# for k in range(2, 465):
#     kmeans = KMeans(n_clusters = k).fit(normal_matrix)
#     labels = kmeans.labels_
#     cost.append(silhouette_score(normal_matrix, labels, metric = 'euclidean'))
#
# plt.plot(range(2,465), cost, color='g', linewidth='3')
# plt.show()  # clear the plot

# model = KMeans(n_clusters=150)  # create KMean model
#
# # data_model = model.fit(normal)  # train model
# data_model = model.fit(normal_matrix)  # train model
#
# print(f'\ndata_model {data_model}')
#
# labels = model.labels_  # labels of data from clustering
#
#
#
# print(f'Labels = {labels}\n')
#
# centroid = model.cluster_centers_   # centroid of KMean model
# # centroid_matrix = pca.transform(centroid)
# df['cluster'] = labels  # add labels to dataframe
# # plt.scatter(x = normal_matrix[:, 0], y = normal_matrix[:, 1], marker="o", alpha=0.2)     # plot scattering of data and
# # plt.scatter(centroid[:, 0], centroid[:, 1], marker='o', color='red')                      # centroid point
# # plt.show()
#
# print(f'\ncentroid data = \n {centroid}')
#
# ct_tmp = pd.crosstab(df['intent'], df['cluster'])
# print(f"type of crosstab is {type(ct_tmp)}")
# # print(pd.crosstab(df['intent'], df['cluster']))     # crosstab
