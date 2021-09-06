import pickle
import pandas as pd
import numpy as np
from function.cleanTextTH import CleanText

def identity_fun(text):
        return text


if __name__ == '__main__':
        model = pickle.load(open('model/model_mlp.pkl', 'rb'))
        feature_  = pickle.load(open('model/feature.pkl', 'rb'))
        intent_name = pd.read_csv('model/intent_name.csv')

        # message = ['สวัสดี', 'หวัดดี', 'ที่พัก', 'สบายดีไหม', 'คุณเป็นใคร']               # select message from new data from line development history chat

        df = pd.read_csv('line_develop/new_message.csv')
        message = df.message.tolist()
        text = CleanText(message)

        data = feature_.transform(text)

        predict_ = model.predict(data).tolist()
        threshold = model.predict_proba(data)

        message_ = []
        target_ = []
        intentName_ = []
        message_not_predict = []
        
        
        for i in range(len(predict_)):
                threshold_ = list(threshold[i,:])
                if threshold_[predict_[i]] >= 0.8:  # if threshold of message >= 80%, save it 
                        message_.append(message[i])
                        target_.append(predict_[i])
                        intentName_.append(intent_name[intent_name.target == predict_[i]].intent.to_string(index=False))
                        print('number message at ', i, ': Finish, \t', threshold_[predict_[i]])

                elif threshold_[predict_[i]] >= 0.5:
                        tmp_threshold_ = threshold_.copy()
                        top_three = []
                        for loop in range(0,3):
                                top_three.append(max(range(len(tmp_threshold_)), key=tmp_threshold_.__getitem__))
                                tmp_threshold_[top_three[-1]] = 0
                        print('number message at ', i, ': Unsure \t', threshold_[predict_[i]])
                        print('message : ', message[i])
                        c_1 = intent_name[intent_name.target == top_three[0]].intent.to_string(index=False) # choice 1
                        c_2 = intent_name[intent_name.target == top_three[1]].intent.to_string(index=False) # choice 2
                        c_3 = intent_name[intent_name.target == top_three[2]].intent.to_string(index=False) # choice 1
                        print('choice : (1) {} \t (2) {} \t (3) {} \t (4) None'.format(c_1, c_2, c_3))
                        select_ = int(input('select : '))
                        if select_ == 1:
                                message_.append(message[i])
                                target_.append(top_three[0])
                                intentName_.append(intent_name[intent_name.target == top_three[0]].intent.to_string(index=False))
                        elif select_ == 2:
                                message_.append(message[i])
                                target_.append(top_three[1])
                                intentName_.append(intent_name[intent_name.target == top_three[1]].intent.to_string(index=False))
                        elif select_ == 3:
                                message_.append(message[i])
                                target_.append(top_three[2])
                                intentName_.append(intent_name[intent_name.target == top_three[2]].intent.to_string(index=False))
                        else :
                                message_not_predict.append(message[i])
                        print('number message at ', i, ': Finish')
                
                else :
                        print('number message at ', i, ': Finish, \t', threshold_[predict_[i]])
                        message_not_predict.append(message[i])

        print('\n-----------------------')        
        print('can predict')
        for i in range(len(message_)):
                print('target = {}, intent = {}, message = {}'.format(target_[i], intentName_[i], message_[i]))
        print('\n-----------------------')
        print('can\'t predict')
        for i in message_not_predict:
                print('message = ', i)

        df_train = pd.read_csv('data_training/intent_group.csv')

        tmp_train = pd.DataFrame({'target':target_, 'intent': intentName_, 'text': message_})

        df_train = df_train.append(tmp_train, ignore_index=True)
        
        df_comsci = pd.DataFrame({'message': message_not_predict})
        
        df_train.to_csv('data_training/new_intent_group.csv',encoding='utf-8-sig',index=False)
        df_comsci.to_csv('line_data/new_comsci_data.csv',encoding='utf-8-sig',index=False)