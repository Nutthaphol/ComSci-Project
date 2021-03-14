import re
import string
import gensim
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from gensim import models
from gensim import corpora
from gensim import similarities



stop_word = stopwords.words('english')
# stemmer = PorterStemmer()
stemmer = SnowballStemmer(language='english')

def cleanData (data):
    clean_data = []
    for sen in data:
        sen = ' '.join([word.lower()
                        for word in sen.split(' ') if word not in stop_word])
        sen = re.sub(r'\'w+', ' ', sen)
        sen = re.sub('[%s]' % re.escape(string.punctuation), ' ', sen)
        sen = re.sub(r'\w*\d+\w*', ' ', sen)
        sen = re.sub(r'\s{2,}', ' ', sen)
        words = word_tokenize(sen)
        st_words = [i for i in words if i not in stop_word]
        clean_data.append(' '.join([stemmer.stem(i) for i in st_words]))

    return clean_data


def termDict (data):
    dicts = {}
    for text in data:
        text = text.split(" ")
        dicts = set(dicts).union(text)
    return dicts


# for i in total:
#         dicts.append(i)
#     dat = pd.Series((total))
#     return dat

def TFCount (dict, data):
    dict = list(dict)