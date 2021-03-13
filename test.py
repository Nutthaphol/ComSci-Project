import pandas as pd
from tf_idf import cleanData
# from tf_idf import preparingTermSet
from tf_idf import termdict
from nltk.corpus import stopwords

df = pd.read_csv('dataset/case_routing_intent.csv')

data = df['data']


clean_data = cleanData(data)

for sen in range(len(clean_data)):
    print(sen, ": ", clean_data[sen])

# dict_data = termdict(clean_data)

# for i in dict_data:
#     print(i)

# print(dict_data)
# print(len(dict_data))
