from nltk import text
import numpy as np
import pandas as pd
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import  CleanText
from function.tf_idf import TF_IDF

def identity_fun(text):
    return text



if __name__ == '__main__':
        df = pd.read_csv('data_training/intent_group.csv')
        X = df.text
        y = df.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

        feature_ = TfidfVectorizer(analyzer = 'word', #this is default
                                    tokenizer=identity_fun, #does no extra tokenizing
                                    preprocessor=identity_fun, #no extra preprocessor
                                    token_pattern=None)

        feature_.fit(X_train)
        save_feature = 'text_feature.pkl'
        pickle.dump(feature_, open(save_feature, 'wb'))

        X_train_fe = feature_.transform(X_train)
        

        mlp = MLPClassifier(max_iter=1000)

        mlp.fit(X_train_fe, y_train)

        X_test_fe = feature_.transform(X_test)

       tmp = mlp.predict(X_test_fe)

        for i in range(len(tmp)):
                print(" {} / {}".format(list(y_test)[i], list(tmp)[i]))

