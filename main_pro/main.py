import requests
import json
import flask
from flask import jsonify, request
import jieba
import numpy as np
from urllib.parse import quote

import configs
#import gif

noneType = "none"


def generate_none_response(num):
    ret = []
    for i in range(num):
        ret.append({"msgType": noneType, "content": "人家都不知道说什么好了 [no response]", "userId": ""})
    return ret


def generate_error_response(num):
    ret = []
    for i in range(num):
        ret.append({
            "msgType": "text",
            "content": "啊哦...服务器好像宕机了...",
            "userId": ""
        })
    return ret


def get_model_response(model, query):
    try:
        r = requests.post(configs.urls[model], json=query)
        return r.json()
    except Exception as e:
        print('Error in get_model_response:', e)
        return generate_none_response(len(query))


def possible_gif(r, postfix):
    if np.random.random() < configs.gif_trigger_prob:
        selected_gif = gif.get_gif(r[0]["content"], configs.gif_sim_thres, configs.gif_k)
    else:
        selected_gif = None
    if selected_gif is None:
        r[0]["content"] += postfix
    else:
        r[0]["msgType"] = 'image'
        r[0]["content"] = selected_gif
    return r


zhihu_questions = ['如何看待', '怎么看待', '如何评价', '怎么评价', '怎样的体验', '什么体验', '如何做到']


def get_response(query):
    if True:
        r = get_model_response("rule", query)
        try:
            if r[0]["msgType"] != noneType:
                #print("rule")
                #print(r)
                if r[0]["msgType"] == 'text':
                    r[0]["content"] += '[Rule]'
                return r[0:1]
        except Exception as e:
            print('Error in get_rule_response:', e)

    if True and query[0]['msgType'] == 'text':
        try:
            msg = ''.join(query[0]['content'])
            for zhihu_question in zhihu_questions:
                if zhihu_question in msg:
                    r = [{
                        "msgType": "text",
                        "content": "有问题，上知乎！\nhttps://www.zhihu.com/search?type=content&q=" + quote(msg),
                        "userId": query[0]['userId']
                    }]
                    return r
        except Exception as e:
            print('Error in Zhihu:', e)

    if True and query[0]['msgType'] == 'text':
        try:
            r = get_model_response("retrieval", query)
            if r[0]["msgType"] != noneType:
                if float(r[0]["score"]) >= configs.retrieval_threshold:
                    #print("retrieval:")
                    #print(r)
                    r = possible_gif(r, "[Retrieval]")
                    return r[0:1]
                else:
                    retrieval_r = r
        except Exception as e:
            print('Error in get_retrieval_response:', e)
            

    if True and query[0]['msgType'] == 'text':
        try:
            r = get_model_response("seq2seq", query)
            if r[0]["msgType"] != noneType:
                print("seq2seq:")
                print(r)
                r = possible_gif(r, "[Seq2seq]")
                return r[0:1]
            else:
                r = retrieval_r
                if float(r[0]["score"]) >= configs.retrieval_soft_threshold:
                    print("retrieval:")
                    print(r)
                    r = possible_gif(r, "[Retrieval]")
                    return r[0:1]
        except Exception as e:
            print('Error in get_seq_response:', e)

    # r = get_model_response("dpgan", query)
    # if r[0]["msgType"] != noneType:
    #    return r
    return generate_none_response(1)


app = flask.Flask(__name__)
app.config["DEBUG"] = False


@app.route("/", methods=["POST"])
def process():
    try:
        queries = request.get_json()
        for i in range(len(queries)):
            queries[i]['content'] = jieba.lcut(queries[i]['content'])
        return jsonify(get_response(queries))
    except Exception as e:
        print(e)
        return jsonify(generate_error_response(1))


if __name__ == "__main__":
    #print(get_model_response("seq2seq", [{"userId": "", "msgType": "text", "content": "今天 天气 怎么样 ？"}]))
    app.run(host='0.0.0.0', port=5002)
    
