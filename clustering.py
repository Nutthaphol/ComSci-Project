from re import M
from function.bestPCA import bestPCA
from function.bestKmeans import best_k
import pandas as pd
from function.kmean_three_level import Kmean_three_level
from function.kmean import Kmean
from function.dbscan import DBSCAN
from function.sse import SSE
from function.tf_idf import TF_IDF
from pythainlp import  word_tokenize
from function.cleanTextTH import CleanText


if __name__ == '__main__':
    df = pd.read_csv("lineData/comsci_data.csv") 
    text = df.text.tolist()
    word_text, new_text = CleanText(text)

    # print(word_text)

    # for i in new_text:
    #     print(i)
    

    # fe = TF_IDF(text=text,format="thai") # create tf-idf format "thai"/"english"
    # n_component = min(fe.shape[0], fe.shape[1])
    # fe_pca = bestPCA(feature=fe, n_component=n_component)

    # k_value = best_k(feature=fe_pca, max_= int(len(fe) *0.1))
    
    # kmean_normal = Kmean(fe=fe,data_= df.copy(),k_value=k_value)
    # kmean_three_level = Kmean_three_level(fe=fe,data_= df.copy(),k_value=k_value)
    # dbscan_3 = DBSCAN(fe=fe,data_= df.copy(), min_pts=3)
    # dbscan_5 = DBSCAN(fe=fe,data_= df.copy(), min_pts=5)

    # # test sse function
    # sse_kmean_normal = SSE(kmean_normal.copy())
    # sse_kmean_three_level = SSE(kmean_three_level.copy())
    # sse_dbscan_3 = SSE(dbscan_3.copy())
    # sse_dbscan_5 = SSE(dbscan_5.copy())

    # ''' set show original '''
    # # tm = kmean_three_level
    # # tm = tm.sort_values(by=['centroids_id'])
    # # tm.to_csv("tmp_show.csv",encoding='utf-8-sig')

    # '''set show real data'''
    # kmean_three_level = pd.DataFrame({"text":kmean_three_level["text"],\
    #                                                                 "centroids_id":kmean_three_level["centroids_id"],\
    #                                                                 "dist_score":kmean_three_level["dist_score"],\
    #                                                                 "avg_dist":kmean_three_level["avg_dist_score"]})  # select the desired column
    # kmean_three_level = kmean_three_level.sort_values(by=['centroids_id'])
    # kmean_three_level.to_csv("comsci_result/Kmean_three_level.csv",encoding='utf-8-sig',index=False)
    # sse_kmean_three_level.to_csv("comsci_result/SSE_kmean_three_level.csv",encoding='utf-8-sig',index=False)

    # kmean_normal = kmean_normal.sort_values(by=["centroids_id"])
    # kmean_normal.to_csv("comsci_result/Kmean_normal.csv",encoding='utf-8-sig',index=False)
    # sse_kmean_normal.to_csv("comsci_result/SSE_kmean_normal.csv",encoding='utf-8-sig',index=False)

    # dbscan_3 = dbscan_3.sort_values(by=["centroids_id"])
    # dbscan_3.to_csv("comsci_result/DBSCAN_3.csv",encoding='utf-8-sig',index=False)
    # sse_dbscan_3.to_csv("comsci_result/SSE_DBSCAN_3.csv",encoding='utf-8-sig',index=False)

    # dbscan_5 = dbscan_5.sort_values(by=["centroids_id"])
    # dbscan_5.to_csv("comsci_result/DBSCAN_5.csv",encoding='utf-8-sig',index=False)
    # sse_dbscan_5.to_csv("comsci_result/SSE_DBSCAN_5.csv",encoding='utf-8-sig',index=False)

    # sum_sse_kmean_normal = sse_kmean_normal["sse"].tolist()
    # sum_sse_kmean_three_level = sse_kmean_three_level["sse"].tolist()
    # sum_sse_dbscan_3 = sse_dbscan_3["sse"].tolist()
    # sum_sse_dbscan_5 = sse_dbscan_5["sse"].tolist()

    # number_cluster = [len(sse_kmean_normal.centroids_id.tolist()), len(sse_kmean_three_level.centroids_id.tolist()), len(sse_dbscan_3.centroids_id.tolist()),len(sse_dbscan_3.centroids_id.tolist())]
    # type = ["Kmean_normal","Kmean_three_level","dbscan(minPts = 3)","dbscan(minPts = 5)"]
    # sum_sse = [sum(sum_sse_kmean_normal), sum(sum_sse_kmean_three_level), sum(sum_sse_dbscan_3), sum(sum_sse_dbscan_5)]

    # conclusion = pd.DataFrame({"type":type, 'number':number_cluster, "sum_sse":sum_sse})

    # conclusion.to_csv("comsci_result/comsci_final.csv",encoding='utf-8-sig',index=False)