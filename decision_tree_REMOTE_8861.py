from operator import index
from nltk import text
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import pickle

from sklearn import tree
# from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import  CleanText
from function.tf_idf import TF_IDF
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

def identity_fun(text):
    return text


if __name__ == '__main__':
    df = pd.read_csv('data_training/intent_group.csv')
    data = df.text
    target = df.target

    f1_score_avg = []

    kf = StratifiedKFold(n_splits=10)

    

    for train, test in  kf.split(data, target):
        X_train = []
        X_test = []

        y_train = []
        y_test = []

        for i in train:
            X_train.append(data[i])
            y_train.append(target[i])
        for i in test:
            X_test.append(data[i])
            y_test.append(target[i])

        feature_ = TfidfVectorizer(analyzer='word', tokenizer=identity_fun, preprocessor=identity_fun, token_pattern=None)
        
        feature_.fit(X_train)

        X_train_fe = feature_.transform(X_train)

        desicion_tree = tree.DecisionTreeClassifier(max_depth=X_train_fe.shape[1])

        desicion_tree.fit(X=X_train_fe, y=y_train)

        X_test_fe = feature_.transform(X_test)

        y_predict = desicion_tree.predict(X_test_fe)

        f1_score_ = f1_score(y_true=y_test, y_pred=y_predict, average='micro')

        print("f1-score", f1_score_)
        f1_score_avg.append(f1_score_)

    print('\nf1 score avg = ', np.mean(f1_score_avg))