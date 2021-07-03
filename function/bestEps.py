import numpy as np
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt

def bestEps (feature):
    neigh = NearestNeighbors(n_neighbors=2)
    nn = neigh.fit(feature)
    distances, indices = nn.kneighbors(feature)
    
    # indices = np.sort(indices,axis=0)
    distances = distances[:,1]
    distances = list(set(distances))
    distances = np.sort(distances, axis=0)

    # max_ = np.max(distances)
    eps_ = 0
    for i in range(len(distances)-1):
        # distances = 75% is the best eps value
        # if (distances[i]*100/max_) >= 75:
        #     # print(distances[i]*100/max_)
        #     eps_ = distances[i]
        #     max_ = 1000
        if distances[i+1] - distances[i] > eps_:
            eps_ = distances[i+1]

    eps_ = round(eps_, 1)

    return eps_
    