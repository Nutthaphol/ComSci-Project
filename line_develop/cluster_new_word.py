import pandas as pd
import os

new_message = []

for i in os.listdir('line_develop'):
        if('.py' not in i and '.csv' not in i) :
                for j in os.listdir('line_develop/'+i) :
                        reader = pd.read_csv('line_develop/'+i+'/'+j,skiprows=3)
                        texts = reader[['Sender name','Message']]
                        index = reader[(reader['Sender name']=='Unknown') & (reader['Message']=='ขอโทษค่ะ อะไรนะคะ')].index-1
                        for k in index :
                                new_message.append(reader['Message'][k])

data = pd.DataFrame({'message' : new_message})
save_ = "line_develop/new_message.csv"
data.to_csv(save_,encoding='utf-8-sig',index=False)
     