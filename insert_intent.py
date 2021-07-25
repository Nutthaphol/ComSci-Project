import pandas as pd
import numpy as np

if __name__ == "__main__": 
        deep_Kmean = "kmean_three_level.csv"
        df = pd.read_csv("comsci_result/"+deep_Kmean) # get data from deep Kmean

        label_group = df.centroids_id.unique() # get label number
        df["intent"] = np.NaN
        df['target'] = np.NaN

        count_target = -1
        for i in label_group: # insert label name by user
                insert_label = df[df.centroids_id == i]
                ex_text = insert_label[insert_label.dist_score == insert_label.dist_score.min()]\
                                                        .text.sample(1).to_string(index=False)
                print("example message : ", ex_text)
                label_name = input("assigned intent : ")
                intent = df.intent.tolist()
                target = -1
                if label_name not in intent:
                        count_target += 1
                        target = count_target
                else:
                        target = df[df.intent == label_name].target.tolist()[0]
                df.at[df.centroids_id == i, "intent"] = label_name
                df.at[df.centroids_id == i, "target"] = target
        
        label_name = df.intent.unique().tolist()

        data_ = pd.DataFrame({'target':df.target, 'intent':df.intent, 'text':df.text })
        save_at = "data_training/intent_group.csv"
        data_.to_csv(save_at,encoding='utf-8-sig',index=False)


        # for i in label_name: # save file with label name
        #         data_group = df[df.intent == i].text
        #         save_at = "data_training/"+str(i)+".csv"
        #         data_group.to_csv(save_at,encoding='utf-8-sig',index=False)

        







        