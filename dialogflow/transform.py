import re, csv, json
from copy import deepcopy
from pprint import pprint

with open(r'test.json') as test_:
        question=json.loads(test_.read())
with open(r'test_usersays_en.json') as test_:
        userSays=json.loads(test_.read())

with open('air_line.csv')  as file_:
        reader = csv.reader(file_, delimiter=",")
        line_count = 0
        index = line_count
        intent = []
        responses = []
        responses_str = ""
        for row in reader:
                if line_count!=0:
                        if row[0] != "" :
                                intent.append(row[0])
                                responses_str = row[1]
                                responses.append([])
                                responses[len(intent)-1].append(responses_str)
                        else:
                                responses_str = row[1]
                                responses[len(intent)-1].append(responses_str)
                line_count += 1

        for i in range(len(intent)):
                question_ = deepcopy(question)
                userSays_ = deepcopy(userSays)

                question_['name']=intent[i]
                question_['responses'][0]['messages'][0]['speech'] = responses[i]
                userSays_[0]['data'][0]['text'] = intent[i]
                name_ = intent[i]+".json"
                name_user_ = intent[i]+"_usersays_en.json"

                with open(name_,'w') as f:
                        f.write(json.dumps(question_))
                with open(name_user_,'w') as f:
                        f.write(json.dumps(userSays_))