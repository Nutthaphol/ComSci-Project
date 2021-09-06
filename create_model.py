from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from function.cut_word import Cut_word
from function.cleanTextTH import CleanText

import pandas as pd
import pickle

def identity_fun(text):
        return text

if __name__ == '__main__':
        df = pd.read_csv('data_training/intent_group.csv')
        X = df.text
        y = df.target.tolist()
        intent  = df.intent.unique().tolist()
        target = df.target.unique().tolist()
        
        X = CleanText(data=X)

        feature_ = TfidfVectorizer(analyzer='word', tokenizer=identity_fun, preprocessor=identity_fun, token_pattern=None).fit(X)
        
        pickle.dump(feature_ , open('model/feature.pkl', 'wb'))

        X = feature_.transform(X)
        
        model = MLPClassifier(hidden_layer_sizes=(X.shape[1]),learning_rate_init=0.001, activation='relu', max_iter=1000)
        model.fit(X=X, y=y)

        pickle.dump(model, open('model/model_mlp.pkl', 'wb'))

        intent_name = pd.DataFrame({'target':target, 'intent':intent})
        intent_name.to_csv('model/intent_name',encoding='utf-8-sig',index=False)

        # pipeline = Pipeline(steps=[('cutWord', Cut_word()), ('feature_', )])
        