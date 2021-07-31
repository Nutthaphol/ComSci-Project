import re, csv, json
from copy import deepcopy
from pprint import pprint

def csvToDialogflow(file_name):
        test_ = open('function/test.json',)
        question=json.loads(test_.read())
        test_ = open('function/test_usersays_en.json') 
        userSays=json.loads(test_.read())
        
        file_ = open(file_name, encoding="utf-8")
        reader = csv.reader(file_, delimiter=",")

        render_line = 0
        intent = None
        message = []
        respons = []

        for row in reader:
                if render_line != 0:
                        if intent == None:
                                intent = row[0]
                        if row[1] != '':
                                message.append(row[1])
                        if row[2] != '':
                                respons.append(row[2])
                render_line += 1
        
        question_ = deepcopy(question)
        userSays_ = deepcopy(userSays)
        text_format = userSays_[0]
        userSays_ = []

        question_['name'] = intent
        question_['responses'][0]['messages'][0]['speech'] = respons

        # save file
        name_ = intent+".json"
        name_user_ = intent+"_usersays_th.json"

        with open('dialogflow_file/'+name_,'w') as f:
                f.write(json.dumps(question_, ensure_ascii=False))
         
        id = 0
        for i in range(len(message)):
                tmp = deepcopy(text_format)
                tmp['id'] = str(id)
                tmp['data'][0]['text'] = message[i]
                id += 1
                userSays_.append(tmp)
                # userSays_[0]['data'][0]['text'] = message[i]
                # userSays_[0]['id'] = id
                # id += 1

        with open('dialogflow_file/'+name_user_,'w') as f:
                f.write(json.dumps(userSays_, ensure_ascii=False))
        
                

        

        
        