from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from function.clust_word import Clust_word

import pandas as pd
import pickle

def identity_fun(text):
        return text

if __name__ == '__main__':
        df = pd.read_csv('data_training/intent_group.csv')
        X = df.text
        y = df.target
        
        X = Clust_word(text=X)

        feature_ = TfidfVectorizer(analyzer='word', tokenizer=identity_fun, preprocessor=identity_fun, token_pattern=None)
        
        pickle.dump(feature_ ,open('feature.pkl', 'wb'))

        X = feature_.transform(X)
        
        model = MLPClassifier(hidden_layer_sizes=(X.shape[1]),learning_rate=0.0001, activation='relu', max_iter=1000)

        pickle.dump(model, open('model_mlp.model'), 'wb')