from operator import index
from nltk import text
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import pickle

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from function.cleanTextTH import  CleanText
from function.tf_idf import TF_IDF
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def identity_fun(text):
    return text

def test_model(data, target):
    f1_feature = []
    acc_feature = []

    f1_feature_2 = []
    acc_feature_2 = []

    f1_feature_4 = []
    acc_feature_4 = []

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

        #for thai data
        # feature_ = TfidfVectorizer(analyzer='word', tokenizer=identity_fun, preprocessor=identity_fun, token_pattern=None)
        
        # for eng data
        feature_ = TfidfVectorizer()

        feature_.fit(X_train)

        X_train_fe = feature_.transform(X_train)
        X_test_fe = feature_.transform(X_test)

        depth = X_train_fe.shape[1]

        #max depth = feature
        desicion_tree = tree.DecisionTreeClassifier(max_depth=depth).fit(X=X_train_fe, y=y_train)

        y_predict = desicion_tree.predict(X_test_fe)

        f1_feature.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_feature.append(accuracy_score(y_test, y_predict))
        
        #max depth = feature/2
        desicion_tree = tree.DecisionTreeClassifier(max_depth=depth/2).fit(X=X_train_fe, y=y_train)

        y_predict = desicion_tree.predict(X_test_fe)

        f1_feature_2.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_feature_2.append(accuracy_score(y_test, y_predict))
        
        #max depth = feature/4
        desicion_tree = tree.DecisionTreeClassifier(max_depth=depth/4).fit(X=X_train_fe, y=y_train)

        y_predict = desicion_tree.predict(X_test_fe)

        f1_feature_4.append(f1_score(y_true=y_test, y_pred=y_predict, average='micro'))
        acc_feature_4.append(accuracy_score(y_test, y_predict))
        
    print('\n----------------------------------------')
    print('**feature**')
    print('\nf1 score avg = ', np.mean(f1_feature))
    print('acc score avg = ', np.mean(acc_feature))
    
    print('\n----------------------------------------')
    print('**feature/2**')
    print('\nf1 score avg = ', np.mean(f1_feature_2))
    print('acc score avg = ', np.mean(acc_feature_2))
    
    print('\n----------------------------------------')
    print('**feature/4**')
    print('\nf1 score avg = ', np.mean(f1_feature_4))
    print('acc score avg = ', np.mean(acc_feature_4))


def create_model(data, target):
    feature_ = TF_IDF()

if __name__ == '__main__':
    df = pd.read_csv('dataset/atis_intents.csv')
    data = df.text
    intent_ = df.intent.tolist()
    # target = df.target

    target = list(LabelEncoder().fit_transform(intent_))

    feature_ = TfidfVectorizer()
    test_fe = feature_.fit_transform(data).todense()

    # for i in range(len(intent_)):
    #     print(label_[i], ' : ', intent_[i])

    test_model(data=data, target=target)

    
