from operator import index
from nltk import text
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import pickle

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import  CleanText
from function.tf_idf import TF_IDF
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from function.clust_word import Clust_word

def identity_fun(text):
    return text

def test_model(data, target):
    f1_d = []
    acc_d = []

    f1_dd = []
    acc_dd = []

    f1_dd2 = []
    acc_dd2 = []

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

        X_train = Clust_word(X_train)
        X_test = Clust_word(X_test)

        feature_ = TfidfVectorizer(analyzer='word', tokenizer=identity_fun, preprocessor=identity_fun, token_pattern=None)
        
        feature_.fit(X_train)

        X_train_fe = feature_.transform(X_train)
        X_test_fe = feature_.transform(X_test)

        # print('shape = ', feature_.get_feature_names())


        dimension_ =X_train_fe.shape[1]

        #  1 layer - dimension
        model_d = MLPClassifier(hidden_layer_sizes=(dimension_,), learning_rate_init=0.001, activation='relu', max_iter=1000).fit(X=X_train_fe, y=y_train)

        y_predict = model_d.predict(X_test_fe)

        f1_d.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_d.append(accuracy_score(y_test, y_predict))


        # 2 layer => dimension + dimension
        model_dd= MLPClassifier(hidden_layer_sizes=(dimension_, dimension_), learning_rate_init=0.001, activation='relu', max_iter=1000).fit(X=X_train_fe, y=y_train)

        y_predict = model_dd.predict(X_test_fe)

        f1_dd.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_dd.append(accuracy_score(y_test, y_predict))


        # 2 layer => dimension + dimension/2        
        model_dd2= MLPClassifier(hidden_layer_sizes=(dimension_,int(dimension_/2)), learning_rate_init=0.001, activation='relu', max_iter=1000).fit(X=X_train_fe, y=y_train)

        y_predict = model_dd2.predict(X_test_fe)

        f1_dd2.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_dd2.append(accuracy_score(y_test, y_predict))
        

    print('\n----------------------------------------')
    print('**d**')
    print('\nf1 score avg = ', np.mean(f1_d))
    print('\nacc score avg = ', np.mean(acc_d))
    
    print('\n----------------------------------------')
    print('**d+d**')
    print('\nf1 score avg = ', np.mean(f1_dd))
    print('\nacc score avg = ', np.mean(acc_dd))
    
    print('\n----------------------------------------')
    print('**d+d/2**')
    print('\nf1 score avg = ', np.mean(f1_dd2))
    print('\nacc score avg = ', np.mean(acc_dd2))


def create_model(data, target):
    feature_ = TF_IDF()

if __name__ == '__main__':
    df = pd.read_csv('data_training/intent_group.csv')
    data = df.text
    target = df.target
    
    test_model(data=data, target=target)

    # create_model(data=data, target=target)