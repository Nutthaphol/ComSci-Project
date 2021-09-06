from posix import listdir
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
        df = pd.read_csv('data_training/new_intent_group.csv')

        list_res_intent = listdir('intent_data')
        list_res_intent.remove('.DS_Store')

        intent_label = df.intent.unique().tolist()

        for i in intent_label:
                print(i)
                res = pd.read_csv('intent_data/'+i+'.csv')
                res_ = res.response.tolist()
                query = df[df.intent == i].text.tolist()
                intent_name = [i] * len(query)
                data_ = pd.DataFrame({'intent':pd.Series(intent_name), 'query':pd.Series(query), 'response': pd.Series(res_)})
                data_.to_csv('intent_data/'+i+'.csv',encoding='utf-8-sig',index=False)
                