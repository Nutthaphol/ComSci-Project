from pythainlp.corpus import thai_stopwords
from pythainlp import  word_tokenize
import pandas as pd

def CleanText(data):
        df = pd.read_csv('function/stop_word_th.csv')
        list_stop_word = df.word.tolist()
        stopwords = frozenset(list_stop_word)
        clean_text = []
        new_text = []
        for sen in data:
                # print("sen = ", sen)
                words = word_tokenize(sen, engine='newmm')
                stop_words = [i for i in words if i not in stopwords]
                stop_words = [i for i in stop_words if i not in "\n"]
                stop_words = [i for i in stop_words if i not in " "]
               #  stop_words = [i for i in stop_words if i not in " "]
                clean_text.append(stop_words)
                new_text.append(''.join(i for i in stop_words))
                # print('clean = ',clean_text[-1])
                # print("new = ", new_text[-1])
                # print('---------------\n')
        #  print(clean_text)
        return clean_text