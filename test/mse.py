import pandas as pd
import numpy as np

if __name__ == '__main__':
        result_data = pd.read_csv('benchmark/benchmark_dataset_03.csv')
        kmean_ = pd.read_csv('benchmark/SSE_kmean_normal_03.csv')
        kmean_3lv_ = pd.read_csv('benchmark/SSE_kmean_three_level_03.csv')
        dbscan_3_ = pd.read_csv('benchmark/SSE_DBSCAN_3_03.csv')
        dbscan_5_ = pd.read_csv('benchmark/SSE_DBSCAN_5_03.csv')

        mse = []
        len_data = len(kmean_.number)
        

        # MSE Kmeans
        id_ = kmean_.centroids_id.tolist()
        tmp_ = []
        for i in id_:
                ci = int(kmean_[kmean_.centroids_id == i].number)
                sse_ = float(kmean_[kmean_.centroids_id == i].sse)
                h_mse = sse_/ci
                h_mse = ci/len_data*h_mse
                tmp_.append(h_mse)
        mse.append(sum(tmp_))

         # MSE deep Kmeans
        id_ = kmean_3lv_.centroids_id.tolist()
        tmp_ = []
        for i in id_:
                ci = int(kmean_3lv_[kmean_3lv_.centroids_id == i].number)
                sse_ = float(kmean_3lv_[kmean_3lv_.centroids_id == i].sse)
                h_mse = sse_/ci
                h_mse = ci/len_data*h_mse
                tmp_.append(h_mse)
        mse.append(sum(tmp_))

        #MSE dbscan 3
        id_ = dbscan_3_.centroids_id.tolist()
        tmp_ = []
        for i in id_:
                ci = int(dbscan_3_[dbscan_3_.centroids_id == i].number)
                sse_ = float(dbscan_3_[dbscan_3_.centroids_id == i].sse)
                h_mse = sse_/ci
                h_mse = ci/len_data*h_mse
                tmp_.append(h_mse)
        mse.append(sum(tmp_))

        # MSE dbscan 5
        id_ = dbscan_5_.centroids_id.tolist()
        tmp_ = []
        for i in id_:
                ci = int(dbscan_5_[dbscan_5_.centroids_id == i].number)
                sse_ = float(dbscan_5_[dbscan_5_.centroids_id == i].sse)
                h_mse = sse_/ci
                h_mse = ci/len_data*h_mse
                tmp_.append(h_mse)
        mse.append(sum(tmp_))

        result_data['mse'] = mse

        result_data.to_csv('benchmark/benchmark_dataset_03.csv', encoding='utf-8', index=False)