import pandas as pd
from tf_idf import cleanData
from tf_idf import termDict
from tf_idf_tmp import TFCount

df = pd.read_csv('dataset/case_routing_intent.csv')

data = df['data']

for i in range(0,10):
    print(i, ": ", data[i])

clean_data = cleanData(data)

print("\n")

for i in range(0, 10):
    print(i, ": ", clean_data[i])

dict_data = termDict(clean_data)

print("\n", dict_data)

# tf = TFcount(clean_data, dict_data)

# for i in tf:
#     print(i, ": ", tf[i])
