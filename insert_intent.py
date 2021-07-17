import pandas as pd
import numpy as np

if __name__ == "__main__": 
        df = pd.read_csv("comsci_result/kmean_three_level.csv") # get data from keam 3 level

        label_group = df.centroids_id.unique() # get label number
        df["intent"] = np.NaN

        for i in label_group: # insert label name by user
                insert_label = df[df.centroids_id == i]
                ex_text = insert_label[insert_label.dist_score == insert_label.dist_score.min()].text.sample(1).to_string(index=False)
                print("example message : ", ex_text)
                label_name = input("assigned intent : ")
                df.at[df.centroids_id == i, "intent"] = label_name
        
        label_name = df.intent.unique().tolist()
        
        for i in label_name: # save file with label name
                data_group = df[df.intent == i].text
                save_at = "data_training/"+str(i)+".csv"
                data_group.to_csv(save_at,encoding='utf-8-sig',index=False)

        







        