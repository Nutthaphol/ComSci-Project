import pandas as pd
import os


if __name__ == '__main__':
        
        new_message = []

        list_sentence_ = ['ขอโทษครับ ตอนนี้ผมยังไม่สามารถตอบข้อความนี้ได้ครับ', 'ขออภัยครับ ผมยังตอบคำถามนี้ไม่ได้ (╥﹏╥)', 'ผมขอโทษครับ ตอนนี้ผมไม่รู้คำตอบจริงๆ']

        for i in os.listdir('line_develop'):
                if('.py' not in i and '.csv' not in i and i not in '.DS_Store') :
                        for j in os.listdir('line_develop/'+i) :
                                reader = pd.read_csv('line_develop/'+i+'/'+j,skiprows=3)
                                texts = reader[['Sender name','Message']]
                                index = list(reader[(reader['Sender name']=='Unknown') & ((reader['Message']== list_sentence_[0]) | (reader['Message']== list_sentence_[1]) | (reader['Message']== list_sentence_[2]))].index-1)
                                for k in index:
                                        new_message.append(reader['Message'][k])

        data = pd.DataFrame({'message' : new_message})
        save_ = "line_develop/new_message.csv"
        data.to_csv(save_,encoding='utf-8-sig',index=False)
        