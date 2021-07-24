import json, csv,codecs
from pythainlp import sent_tokenize, word_tokenize

with open('comsci_res.txt',"r") as reader:
        texts = reader.readlines()
        set_texts = []
        check = True;
        for text in texts:
                if len(text) <= 1: 
                        check = False
                if (text.find("joined the group.") != -1):
                        check = False
                if (text.find("[Notes]") != -1):
                        check = False
                # add text
                if (check):
                        set_texts.append(text.split("\t"))
                check = True
        
        last_texts = ["text"]

        for text in set_texts:
                message = ""
                for index in range(2,len(text)):
                        message += text[index]
                if message != "" and message !="BE":
                        last_texts.append(message)
        
        with codecs.open("comsci_data.csv","w", "utf-8") as write:
                text_write = csv.writer(write)
                for text in last_texts:
                        text_write.writerow([text])



