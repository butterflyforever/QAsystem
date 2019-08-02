import requests
import json
import jieba

if __name__ == '__main__':

    x = [{
        'userId': 'sdkfja1231',
        'msgType': 'text',
        'content': ['不是', '不','？']}]
    # }, {
    #     'userId': 'asdfjanadf',
    #     'msgType': 'text',
    #     'content': '今天天气怎么样？'
    # }, {
    #     'userId': 'asdfjanadf',
    #     'msgType': 'text',
    #     'content': '人工智能会毁灭世界吗'
    # }, {
    #     'userId': 'asdfjanadf',
    #     'msgType': 'text',
    #     'content': '你能通过图灵测试吗'
    # }, {
    #     'userId': 'asdfjanadf',
    #     'msgType': 'text',
    #     'content': '我的天啊，这家伙太可怕了'
    # }]

    r = requests.post('http://0.0.0.0:8888/', json=x)
    print(r.json())
    #app.run(port=8000)
    