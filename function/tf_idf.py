from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import CleanText as CleanTextTH
from function.cleanTextEng import CleanText as CleanTextENG

def identity_fun(text):
    return text

def TF_IDF(text,format):
        if format == "thai":
                clean_text = CleanTextTH(text)
                tf_idf = TfidfVectorizer(analyzer = 'word', #this is default
                                    tokenizer=identity_fun, #does no extra tokenizing
                                    preprocessor=identity_fun, #no extra preprocessor
                                    token_pattern=None)
                fe = tf_idf.fit_transform(clean_text).todense()
                return fe
        elif format == "english":
                clean_text = CleanTextENG(text)
                tf_idf = TfidfVectorizer()
                fe = tf_idf.fit_transform(clean_text).todense()
                return fe
