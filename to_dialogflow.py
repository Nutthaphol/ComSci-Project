import os
from function.csv_to_dialogflow import csvToDialogflow

if __name__ == '__main__':
        
        list_file = os.listdir('intent_data')

        for i in list_file:
                csvToDialogflow(file_name='intent_data/'+i)