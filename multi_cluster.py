import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from function.cleanTextTH import CleanText
from function.bestKmeans import best_k
from function.kmean_three_level import Kmean_three_level, avg_dist

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

        pattern = []
        for i in range(len(new_df)):
                tmp = []                
                tmp.append(new_df["centroids_id_level_1"].loc[i])
                tmp.append(new_df["centroids_id_level_2"].loc[i])
                tmp.append(new_df["centroids_id_level_3"].loc[i])
                if tmp in pattern:
                        continue
                pattern.append(tmp)


        true_df = pd.DataFrame(columns=["text","centroids_id","avg_distance"])

        for i in range(len(pattern)):
                set_pattern = pattern[i]
                tmp_group = new_df.loc[(new_df.centroids_id_level_1 == set_pattern[0] ) & \
                                (new_df.centroids_id_level_2 == set_pattern[1]) & \
                                (new_df.centroids_id_level_3 == set_pattern[2])]
                print(tmp_group)
                data = {"text":tmp_group["text"],"centroids_id":i,"avg_distance":tmp_group["avg_dist_score"]}
                data_df = pd.DataFrame(data)
                true_df = true_df.append(data_df, ignore_index=True)

        # clustering result quality
        # sse
        # accuracy
        # purity

        tmp_df = pd.DataFrame({"text":new_df["text"],"centroids_id":new_df["centroids_id"],"avg_dist":new_df["avg_dist_score"]})
        tmp_df = tmp_df.sort_values(by=['centroids_id'])


        tmp_df.to_csv("cluster_comsci_multi_kmean.csv",encoding='utf-8-sig')