from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def best_k (feature, max_):
    num_sil = []
    k = 0
    maximum = 0
    
    for i in range(2, max_+1):
        model = KMeans(n_clusters=i).fit(feature)
        label_ = model.labels_
        centroids = model.cluster_centers_
        num_sil.append(silhouette_score(feature,label_,metric='euclidean'))
        if num_sil[-1] > maximum:
            maximum = num_sil[-1]
            k = i
    return k