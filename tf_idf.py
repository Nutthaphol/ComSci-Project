import re
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string



stop_word = stopwords.words('english')
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

# def termdict (data):
#     dicts = {}
#     term = []
#     for word in data:
#         word = word_tokenize(word)
#         for tokens in word:
#             if tokens not in dicts.keys():
#                 dicts[tokens] = 1
#             else:
#                 dicts[tokens] += 1
# 
    # for i in dicts:
    #     term.append(i)
    # term.sort()
    # return term

def termdict (data):
    totol = {}
    for text in data:
        text = text.split(" ")
        text = set(text).union()

# def TFcount(data, dicts):

    