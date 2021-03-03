from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk import word_tokenize

pd.set_option('display.max_rows',5000)
pd.set_option('display.max_columns',5000)
pd.set_option('display.width',5000)

data = pd.read_csv('dataset/atis_intents.csv')

n_clusters=22

kmean = KMeans(n_clusters,random_state=1)

print(data.intent.value_counts().sort_index())

#set_data = data.data.values.tolist()

intent = list(data['intent'])     # import dataset to list type
new_data = list(data['data'])
# clean_data = list(df['data'])

stop_word = set(stopwords.words('english'))      # function for remove stopwords
clean_data = []     # list for keep data after making a cleaning dataset
for i in new_data:
    i = i.lower()   # set all character to lower font
    i = re.sub('[0-9]'," ", i)      # remove numbering from sentence
    i = word_tokenize(i)   # split word from sentence
    clean_data.append(' '.join([j for j in i if j not in stop_word]))  # remove stopwords

vt = TfidfVectorizer()

cdata_matrix = vt.fit_transform(clean_data).todense()

#print('This is center\n',centroids)
#print('\n this is all\n',everything)

tsne_init = 'random'  # could also be 'random'
tsne_perplexity = 50.0
tsne_early_exaggeration = 12
tsne_learning_rate = 10
model = TSNE(n_components=2, random_state= 1, init=tsne_init, perplexity=tsne_perplexity,
         early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)

ts = model.fit_transform(cdata_matrix)

#plt.scatter(ts[:, 0], ts[:, 1], marker='o')
plt.show()

new_data = kmean.fit(ts)

centroids = kmean.cluster_centers_

cc =  kmean.labels_

data['cluster'] = cc

plt.scatter(x = ts[:, 0], y = ts[:, 1], marker="o", alpha=0.2)     # plot scattering of data and
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', color='red')                      # centroid point
plt.show()
#print('\nkmean label\n',data)


print(f'crosstab data is \n {pd.crosstab(data["intent"],data["cluster"])}')
