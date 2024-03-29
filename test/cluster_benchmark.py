import pandas as pd
from function.kmean_three_level import Kmean_three_level
from function.kmean import Kmean
from function.dbscan import DBSCAN
from function.sse import SSE
from function.tf_idf import TF_IDF
from function.purityCluster import purity
from function.bestKmeans import best_k
from function.bestPCA import bestPCA

if __name__ == '__main__':
        df = pd.read_csv("dataset/case_routing_intent.csv")
        text = df["text"]

        fe = TF_IDF(text=text,format="english")
        n_component = min(fe.shape[0], fe.shape[1])
        fe_pca = bestPCA(feature=fe, n_component=n_component)

        k_value = best_k(feature=fe_pca, max_= int(len(fe) *0.1))

        kmean_normal = Kmean(fe=fe_pca,data_= df.copy(), k_value=k_value)
        kmean_three_level = Kmean_three_level(fe=fe_pca,data_= df.copy(),k_value=k_value)
        dbscan_3 = DBSCAN(fe=fe_pca,data_= df.copy(), min_pts=3)
        dbscan_5 = DBSCAN(fe=fe_pca,data_= df.copy(), min_pts=5)

        sse_kmean_normal = SSE(kmean_normal.copy())
        sse_kmean_three_level = SSE(kmean_three_level.copy())
        sse_dbscan_3 = SSE(dbscan_3.copy())
        sse_dbscan_5 = SSE(dbscan_5.copy())
    
        comp_Kmean_normal = pd.crosstab(kmean_normal['intent'], kmean_normal['centroids_id'])
        comp_kmean_three_level = pd.crosstab(kmean_three_level['intent'], kmean_three_level['centroids_id'])
        comp_dbscan_3 = pd.crosstab(dbscan_3['intent'], dbscan_3['centroids_id'])
        comp_dbscan_5 = pd.crosstab(dbscan_5['intent'], dbscan_5['centroids_id'])
        
        purity_kmean_normal = purity(crosstab_=comp_Kmean_normal, size_data=len(text))
        purity_kmean_three_level = purity(crosstab_=comp_kmean_three_level, size_data=len(text))
        purity_dbscan_3 = purity(crosstab_=comp_dbscan_3, size_data=len(text))
        purity_dbscan_5 = purity(crosstab_=comp_dbscan_5, size_data=len(text))

        sse_kmean_normal["purity"] = purity_kmean_normal
        sse_kmean_three_level["purity"] = purity_kmean_three_level
        sse_dbscan_3["purity"] = purity_dbscan_3
        sse_dbscan_5["purity"] = purity_dbscan_5

        '''set show real data'''
        kmean_three_level = pd.DataFrame({"text":kmean_three_level["text"],\
                                                                        "centroids_id":kmean_three_level["centroids_id"],\
                                                                        "dist_score":kmean_three_level["dist_score"],\
                                                                        "avg_dist":kmean_three_level["avg_dist_score"]})  # select the desired column
        kmean_three_level = kmean_three_level.sort_values(by=['centroids_id'])
        kmean_three_level.to_csv("benchmark/Kmean_three_level_03.csv",encoding='utf-8-sig')
        sse_kmean_three_level.to_csv("benchmark/SSE_kmean_three_level_03.csv",encoding='utf-8-sig')

        kmean_normal = kmean_normal.sort_values(by=["centroids_id"])
        kmean_normal.to_csv("benchmark/Kmean_normal_03.csv",encoding='utf-8-sig')
        sse_kmean_normal.to_csv("benchmark/SSE_kmean_normal_03.csv",encoding='utf-8-sig')

        dbscan_3 = dbscan_3.sort_values(by=["centroids_id"])
        dbscan_3.to_csv("benchmark/DBSCAN_3_03.csv",encoding='utf-8-sig')
        sse_dbscan_3.to_csv("benchmark/SSE_DBSCAN_3_03.csv",encoding='utf-8-sig')

        dbscan_5 = dbscan_5.sort_values(by=["centroids_id"])
        dbscan_5.to_csv("benchmark/DBSCAN_5_03.csv",encoding='utf-8-sig')
        sse_dbscan_5.to_csv("benchmark/SSE_DBSCAN_5_03.csv",encoding='utf-8-sig')

        sum_sse_kmean_normal = sse_kmean_normal["sse"].tolist()
        sum_sse_kmean_three_level = sse_kmean_three_level["sse"].tolist()
        sum_sse_dbscan_3 = sse_dbscan_3["sse"].tolist()
        sum_sse_dbscan_5 = sse_dbscan_5["sse"].tolist()

        type = ["Kmean_normal","Kmean_three_level","dbscan"]
        sum_sse = [sum(sum_sse_kmean_normal), sum(sum_sse_kmean_three_level), sum(sum_sse_dbscan)]
        purity_ = [purity_kmean_normal, purity_kmean_three_level, purity_dbscan]

        type = ["DBSCAN_3","DBSCAN_5"]
        sum_sse = [sum(sum_sse_dbscan_3), sum(sum_sse_dbscan_5)]
        purity_ = [purity_dbscan_3, purity_dbscan_5]

        conclusion = pd.DataFrame({"type":type, "sum_sse":sum_sse, "purity":purity_})
        
        conclusion.to_csv("benchmark/fix_cluster_dbscan_03.csv", encoding='utf-8-sig')


