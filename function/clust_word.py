from pythainlp import word_tokenize

def Clust_word(text):
        token_words = []
        for sen in text:
                words = word_tokenize(sen, engine='attacut')
                token_words.append(words)
        return token_words