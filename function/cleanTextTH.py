from pythainlp.corpus import thai_stopwords
from pythainlp import  word_tokenize

def CleanText(data):
         tmp = {"มี","บ้าง","คะ","ครับ","ไหม",}
        #  stopwords = thai_stopwords()
         stopwords = frozenset(tmp)
         clean_text = []
         for sen in data:
                 words = word_tokenize(sen)
                 stop_words = [i for i in words if i not in stopwords]
                 stop_words = [i for i in words if i not in "\n"]
                 clean_text.append(''.join(i for i in stop_words))
        #  print(clean_text)
         return clean_text