from function.bestPCA import bestPCA
from function.mse import MSE
from function.cleanTextTH import CleanText
from function.tf_idf import TF_IDF
import pandas as pd

if __name__ == '__main__':
        df = pd.read_csv('lineData/comsci_data.csv')
        
        text = df.text

        tmp = CleanText(data=text)

        for i in tmp:
                print(i)

        fe = TF_IDF(text=text,format="thai") # create tf-idf format "thai"/"english"
        n_component = min(fe.shape[0], fe.shape[1])
        fe_pca = bestPCA(feature=fe, n_component=n_component)
        print(fe_pca.shape)

        next = input('key enter to next....')
        for i in fe_pca:
                print(i)

        