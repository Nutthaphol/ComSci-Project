from nltk import text
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from function.cleanTextTH import  CleanText
from function.tf_idf import TF_IDF




if __name__ == '__main__':
        df = pd.read_csv('data_training/intent_group.csv')
        