import numpy as np
import pandas as pd

if __name__ == '__main__':
        df = pd.read_csv('data_training/intent_group.csv')

        intent_label = df.intent.unique().tolist()

        for i in intent_label:
                query = df[df.intent == i].text.tolist()
                intent_name = [i] * len(query)
                data_ = pd.DataFrame({'intent':intent_name, 'query':query})
                data_.to_csv('intent_data/'+i+'.csv',encoding='utf-8-sig',index=False)
                