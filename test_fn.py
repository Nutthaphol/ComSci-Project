from function.mse import MSE

import pandas as pd

if __name__ == '__main__':
        df = pd.read_csv('comsci_result/SSE_kmean_normal.csv')
        
        tmp = MSE(data_=df.copy())
        
        print(tmp)