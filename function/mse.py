import numpy as np
import pandas as pd

def MSE (data_):
        len_data = len(data_.number)
        
        id_ = data_.centroids_id.tolist()
        mse = 0
        for i in id_:
                ci = int(data_[data_.centroids_id == i].number)
                sse = float(data_[data_.centroids_id == i].sse)
                mse += ci/len_data*(sse/ci)
        return mse