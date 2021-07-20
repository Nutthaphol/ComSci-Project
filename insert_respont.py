import pandas as pd 
import numpy as np


if __name__ == '__main__':
        df = pd.read_csv('data_training/data_comsci.csv')

        label_number = df.target.unique().tolist()
        for i in label_number:
                data_= df[df.target == i].text
                intent_name = df[df.target]
                data_.to_csv('test.csv',encoding='utf-8-sig',index=False)
