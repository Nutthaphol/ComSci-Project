import pandas as pd
from function.kmean_three_level import Kmean_three_level
from function.kmean import Kmean
from function.dbscan import DBSCAN
from function.sse import SSE
from function.tf_idf import TF_IDF
from pythainlp import  word_tokenize


if __name__ == '__main__':
    df = pd.read_csv("lineData/comsci_data.csv") 
    text = df.text
    text = df.text
    clean_text = []
    for sen in text:
        words = word_tokenize(sen,engine='newmm')
        stop_words = [i for i in words if i not in "\n"]
        clean_text.append(''.join(i for i in stop_words))
    df.text = clean_text

    fe = TF_IDF(text=text,format="thai") # create tf-idf format "thai"/"english"
    
    kmean_normal = Kmean(fe=fe,data_= df.copy())
    kmean_three_level = Kmean_three_level(fe=fe,data_= df.copy())
    dbscan = DBSCAN(fe=fe,data_= df.copy())

    # test sse function
    sse_kmean_normal = SSE(kmean_normal.copy())
    sse_kmean_three_level = SSE(kmean_three_level.copy())
    sse_dbscan = SSE(dbscan.copy())
    
    # clustering result quality
    # sse
    # accuracy
    # purity

    ''' set show original '''
    # tm = kmean_three_level
    # tm = tm.sort_values(by=['centroids_id'])
    # tm.to_csv("tmp_show.csv",encoding='utf-8-sig')

    '''set show real data'''
    kmean_three_level = pd.DataFrame({"text":kmean_three_level["text"],\
                                                                    "centroids_id":kmean_three_level["centroids_id"],\
                                                                    "dist_score":kmean_three_level["dist_score"],\
                                                                    "avg_dist":kmean_three_level["avg_dist_score"]})  # select the desired column
    kmean_three_level = kmean_three_level.sort_values(by=['centroids_id'])
    kmean_three_level.to_csv("comsci_result/Kmean_three_level.csv",encoding='utf-8-sig',index=False)
    sse_kmean_three_level.to_csv("comsci_result/SSE_kmean_three_level.csv",encoding='utf-8-sig',index=False)

    kmean_normal = kmean_normal.sort_values(by=["centroids_id"])
    kmean_normal.to_csv("comsci_result/Kmean_normal.csv",encoding='utf-8-sig',index=False)
    sse_kmean_normal.to_csv("comsci_result/SSE_kmean_normal.csv",encoding='utf-8-sig',index=False)

    dbscan = dbscan.sort_values(by=["centroids_id"])
    dbscan.to_csv("comsci_result/DBSCAN.csv",encoding='utf-8-sig',index=False)
    sse_dbscan.to_csv("comsci_result/SSE_DBSCAN.csv",encoding='utf-8-sig',index=False)
