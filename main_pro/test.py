import requests
import json
import jieba

import configs

noneType = "none"

def generate_none_response(num):
    ret = []
    for i in range(num):
        ret.append({"msgType": noneType, "content": "", "userId": ""})
    return ret

def get_model_response(model, query):
    try:
        r = requests.post(configs.urls[model], json=json.dumps(query))
        return r.json()
    except:
        return generate_none_response(len(query))

if __name__ == '__main__':
    while True:
        # f = open('/data/share/corpus/turing.query')
        # for line in f:
        while True:
            query = input('Query: ')
            # print(query)
            x = [{
                'userId': 'as1nadf',
                'msgType': 'text',
                # 'content': line.replace(' ', '')
                'content': query
            }]
            r = requests.post('http://127.0.0.1:5002', json=x)
            print(r.json()[0]['content'])
        # input()
