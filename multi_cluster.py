import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import CleanText
from function.kmean_three_level import Kmean_three_level
from function.kmean import Kmean
from function.dbscan import DBSCAN

def identity_fun(text):
    return text


if __name__ == '__main__':
    df = pd.read_csv("lineData/comsci_data.csv") 
    text = df.text
    clean_text = CleanText(text)
    tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                    tokenizer=identity_fun, #does no extra tokenizing
                                    preprocessor=identity_fun, #no extra preprocessor
                                    token_pattern=None)
    fe = tf_idf.fit_transform(clean_text).todense()
    
    kmean_normal = Kmean(fe=fe,data_= df.copy())
    kmean_three_level = Kmean_three_level(fe=fe,data_= df.copy())
    dbscan = DBSCAN(fe=fe,data_= df.copy())
    
    # clustering result quality
    # sse
    # accuracy
    # purity

    ''' set show original '''
    # tm = kmean_three_level
    # tm = tm.sort_values(by=['centroids_id'])
    # tm.to_csv("tmp_show.csv",encoding='utf-8-sig')

    '''set show real data'''
    kmean_three_level = pd.DataFrame({"text":kmean_three_level["text"],"centroids_id":kmean_three_level["centroids_id"],"dist_score":kmean_three_level["dist_score"],"avg_dist":kmean_three_level["avg_dist_score"]})
    kmean_three_level = kmean_three_level.sort_values(by=['centroids_id'])
    kmean_three_level.to_csv("Kmean_three_level.csv",encoding='utf-8-sig')

    kmean_normal = kmean_normal.sort_values(by=["centroids_id"])
    kmean_normal.to_csv("Kmean_normal.csv",encoding='utf-8-sig')
    
    dbscan = dbscan.sort_values(by=["centroids_id"])
    dbscan.to_csv("DBSCAN.csv",encoding='utf-8-sig')
