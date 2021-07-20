import numpy as np
from sklearn.decomposition import PCA

def bestPCA (feature, n_component):
    pca = PCA(n_components=n_component)
    pca.fit(feature)
    sum_variance_ratio = np.sum(pca.explained_variance_ratio_)
    number_component = 0

    for i in pca.explained_variance_ratio_:
        number_component += 1
        if np.sum(pca.explained_variance_ratio_[0:number_component]) >= 0.8:
            break
    
    pca = PCA(n_components=number_component).fit(feature)

    return pca.transform(feature)
