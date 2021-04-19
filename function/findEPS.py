import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_eps (feature):
    neigh = NearestNeighbors(n_neighbors=2)
    nn = neigh.fit(feature)
    distances, indices = nn.kneighbors(feature)
    
    distances = np.sort(distances, axis=0)
    indices = np.sort(indices,axis=0)
    distances = distances[:,1]

    max_ = np.max(distances)
    esp_ = 0
    for i in range(len(distances)):
        # distances = 75% is the best eps value
        if (distances[i]*100/max_) >= 75:
            # print(distances[i]*100/max_)
            eps_ = distances[i]
            max_ = 1000

    eps_ = round(eps_, 1)

    return eps_
    