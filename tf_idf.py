import pandas as pd
import re
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import  WordNetLemmatizer

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 5000)

df = pd.read_csv('dataset/atis_intents.csv')

intent = list(df['intent'])
# data = list(df['data'])

data = ["What is your name.", "Hello, my name is 1234"]

clean = []

for s in data:
    sw = re.sub(r'[^a-z A-Z 0-9]', " ", s)
    sw = re.findall('\D',sw)
    # sw = sw.lower()
    # sw = sw.split()
    # sw = sw
    print(f'data = {sw}, tpye = {type(sw)}')

