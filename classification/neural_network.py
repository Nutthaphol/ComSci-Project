import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
        list_file_data = os.listdir('data_training')
        intent_name = []
        train_x = []
        target_y = []

        if '.DS_Store' in list_file_data:
                list_file_data.remove('.DS_Store')
        
        for file_data in list_file_data:
                name, type = os.path.splitext(file_data)
                intent_name.append(name)
                
                df = pd.read_csv('data_training/'+file_data)
                tmp_text = df.text.tolist()
                for text in tmp_text:
                        train_x.append(text)
                        target_y.append(len(intent_name) - 1)
