import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from function.cleanTextTH import CleanText
from function.bestKmeans import best_k
from function.kmean_three_level import Kmean_three_level

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
        new_df = Kmean_three_level(fe,df)

        list_dist_score = new_df["avg_dist_score"].tolist()
        list_level_score = new_df["avg_dist_score_from_level"].tolist()


        # clustering result quality
        # sse
        # accuracy
        # purity
        tmp_df = new_df
        # tmp_df = new_df.sort_values(by=['centroids_id_level_1'])    

        tmp_df.to_csv("cluster_comsci_multi_kmean.csv",encoding='utf-8-sig')