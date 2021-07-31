import os
from function.csv_to_dialogflow import csvToDialogflow

if __name__ == '__main__':
        
        list_file = os.listdir('intent_data')
        list_file.remove('.DS_Store')

        for i in list_file:
                file_location = str('intent_data/'+i)
                csvToDialogflow(file_name=file_location)