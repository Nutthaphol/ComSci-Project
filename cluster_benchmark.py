import pandas as pd
from function.kmean_three_level import Kmean_three_level
from function.kmean import Kmean
from function.dbscan import DBSCAN
from function.sse import SSE
from function.tf_idf import TF_IDF
from function.purityCluster import purity

if __name__ == '__main__':
        df = pd.read_csv("dataset/corona.csv")
        text = df["text"]

        fe = TF_IDF(text=text,format="english")

        kmean_normal = Kmean(fe=fe,data_= df.copy())
        kmean_three_level = Kmean_three_level(fe=fe,data_= df.copy())
        dbscan = DBSCAN(fe=fe,data_= df.copy())

        sse_kmean_normal = SSE(kmean_normal.copy())
        sse_kmean_three_level = SSE(kmean_three_level.copy())
        sse_dbscan = SSE(dbscan.copy())
    
        comp_Kmean_normal = pd.crosstab(kmean_normal['intent'], kmean_normal['centroids_id'])
        comp_kmean_three_level = pd.crosstab(kmean_three_level['intent'], kmean_three_level['centroids_id'])
        comp_dbscan = pd.crosstab(kmean_normal['intent'], kmean_normal['centroids_id'])
        
        purity_kmean_normal = purity(crosstab_=comp_Kmean_normal, size_data=len(text))
        purity_kmean_three_level = purity(crosstab_=comp_kmean_three_level, size_data=len(text))
        purity_dbscan = purity(crosstab_=comp_dbscan, size_data=len(text))

        sse_kmean_normal["purity"] = purity_kmean_normal
        sse_kmean_three_level["purity"] = purity_kmean_three_level
        sse_dbscan["purity"] = purity_dbscan

        '''set show real data'''
        kmean_three_level = pd.DataFrame({"text":kmean_three_level["text"],\
                                                                        "centroids_id":kmean_three_level["centroids_id"],\
                                                                        "dist_score":kmean_three_level["dist_score"],\
                                                                        "avg_dist":kmean_three_level["avg_dist_score"]})  # select the desired column
        kmean_three_level = kmean_three_level.sort_values(by=['centroids_id'])
        kmean_three_level.to_csv("comsci_result/Kmean_three_level.csv",encoding='utf-8-sig')
        sse_kmean_three_level.to_csv("comsci_result/SSE_kmean_three_level.csv",encoding='utf-8-sig')

        kmean_normal = kmean_normal.sort_values(by=["centroids_id"])
        kmean_normal.to_csv("comsci_result/Kmean_normal.csv",encoding='utf-8-sig')
        sse_kmean_normal.to_csv("comsci_result/SSE_kmean_normal.csv",encoding='utf-8-sig')

        dbscan = dbscan.sort_values(by=["centroids_id"])
        dbscan.to_csv("comsci_result/DBSCAN.csv",encoding='utf-8-sig')
        sse_dbscan.to_csv("comsci_result/SSE_DBSCAN.csv",encoding='utf-8-sig')

        sum_sse_kmean_normal = sse_kmean_normal["sse"].tolist()
        sum_sse_kmean_three_level = sse_kmean_three_level["sse"].tolist()
        sum_sse_dbscan = sse_dbscan["sse"].tolist()

        type = ["Kmean_normal","Kmean_three_level","dbscan"]
        sum_sse = [sum(sum_sse_kmean_normal), sum(sum_sse_kmean_three_level), sum(sum_sse_dbscan)]
        purity_ = [purity_kmean_normal, purity_kmean_three_level, purity_dbscan]

        conclusion = pd.DataFrame({"type":type, "sum_sse":sum_sse, "purity":purity_})
        
        conclusion.to_csv("benchmark_dataset_conclusion.csv", encoding='utf-8-sig')