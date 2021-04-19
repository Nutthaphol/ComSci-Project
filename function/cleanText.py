import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


def CleanText (data):
    stemmer = SnowballStemmer(language='english')
    stopword = stopwords.words('english')
    clean_text = []

    for sen in data:
        sen = ' '.join([word.lower() for word in sen.split(' ')])
        sen = re.sub(r'\'w+', ' ', sen)
        sen = re.sub('[%s]' % re.escape(string.punctuation), ' ', sen)
        sen = re.sub(r'\w*\d+\w*', ' ', sen)
        sen = re.sub(r'\s{2,}', ' ', sen)
        words = word_tokenize(sen)
        stop_words = [i for i in words if i not in stopword]
        stem_words = [stemmer.stem(i) for i in stop_words]
        clean_text.append(' '.join(i for i in stem_words))

    return clean_text