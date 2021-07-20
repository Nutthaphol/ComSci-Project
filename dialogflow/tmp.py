import re, csv, json
from copy import deepcopy
from pprint import pprint


if __name__ == '__main__':
        test_ = open('test.json')
        question=json.loads(test_.read())
        test_ = open('test_usersays_en.json') 
        userSays=json.loads(test_.read())
        
        file_ = open()