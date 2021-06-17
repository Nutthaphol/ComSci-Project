from pythainlp.corpus import thai_stopwords
from pythainlp import  word_tokenize

def CleanText(data):
         tmp = {"มี","บ้าง","คะ","ครับ","ไหม","ขอ","หน่อย","ครับ","ค่ะ","คะ","จ้า","อะไร"," "}
        #  stopwords = thai_stopwords()
         stopwords = frozenset(tmp)
         clean_text = []
         for sen in data:
                 words = word_tokenize(sen,engine='newmm')
                 stop_words = [i for i in words if i not in stopwords]
                 stop_words = [i for i in stop_words if i not in "\n"]
                #  stop_words = [i for i in stop_words if i not in " "]
                 clean_text.append(stop_words)
                #  clean_text.append(''.join(i for i in stop_words))
        #  print(clean_text)
         return clean_text