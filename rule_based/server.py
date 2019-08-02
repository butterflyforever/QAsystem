from flask import Flask, Response, request
from random import choice
from utils.q import Q
from utils.image_mode import ImageBot
from utils import corpus
from utils.sensitive import SensitiveFilter
import re
import json
import random

app = Flask(__name__)

MSG_TEXT = 'text'
MSG_IMAGE = 'image'
MSG_FOLLOW = 'new'
MSG_OTHER = 'other'
MSG_NONE = 'none'
MSG_EMOJI = 'emoji'
MSG_DOUTU = 'doutu'
MSG_VOICE = 'voice'

DISCARD_PERCENTAGE = 0.2


# 处理post请求
@app.route('/', methods=['POST'])
def message():
    data = request.get_json()
    resp = []
    for da in data:
        userid = da['userId']
        msgtype = da['msgType']
        content = ''.join(da['content'])
        rpl_userid, rpl_msgtype, rpl_content = process(userid, msgtype, content)
        reply = {
            'userId': rpl_userid,
            'msgType': rpl_msgtype,
            'content': rpl_content,
        }
        resp.append(reply)
    return Response(json.dumps(resp), mimetype='application/json')


# 存放用户聊天上下文
session = {}

# 敏感词过滤器
sensitive_filter = SensitiveFilter()


# 斗图模式机器人
# image_bot = ImageBot()

# 处理消息
def process(userid, msgtype, content):
    # # 新添加好友，回复欢迎语
    # if msgtype == MSG_FOLLOW:
    # 	reply = choice(corpus.follow)
    # 	return userid, reply[0], reply[1]

    # 为首次发来消息的用户创建上下文
    print(userid)
    print(content)
    if userid not in session:
        session[userid] = Q()

    # 处理不支持的类型
    if msgtype == MSG_OTHER:
        reply = choice(corpus.not_support)
        return userid, reply[0], reply[1]

    # 处理图片消息
    if msgtype == MSG_IMAGE:
        reply = choice(corpus.image)
        return userid, reply[0], reply[1]

    # 处理语音消息
    if msgtype == MSG_VOICE:
        reply = choice(corpus.voice)
        return userid, reply[0], reply[1]

    # 获取用户上下文
    queue = session[userid]

    # 处理表情消息
    if msgtype == MSG_EMOJI:
        # 斗图模式
        if queue.mode == 'emoji':
            reply = choice(corpus.doutu_bad)
            return userid, reply[0], reply[1]
        # return userid, MSG_IMAGE, image_bot.choose_image()
        # 非斗图模式
        # 将图片类型 append 到上下文中
        queue.append('EMOJI')
        # 判断是否应该开启斗图模型
        if queue.should_image_mode():
            queue.mode = 'emoji'
            reply = choice(corpus.enter_emoji)
            return userid, reply[0], reply[1]
        # 不开启斗图模式，正常回复
        reply = choice(corpus.emoji)
        return userid, reply[0], reply[1]

    # 将文本append到上下文中
    queue.append(content)

    # 退出斗图模式
    queue.mode = 'normal'

    # 判断是否复读机
    if queue.is_repeat(content):
        reply = choice(corpus.repeat)
        return userid, reply[0], reply[1]

    # 判断是否空
    if not content.strip():
        reply = choice(corpus.blank)
        return userid, reply[0], reply[1]

    # # 判断是否全英语
    # if all(ord(c) < 128 for c in content):
    # 	reply = choice(corpus.all_english)
    # 	return userid, reply[0], reply[1]

    # 过滤敏感词
    if sensitive_filter.is_sensitive(content):
        reply = choice(corpus.sensitive)
        return userid, reply[0], reply[1]

    # 进行关键字匹配
    for kw in corpus.keywords:
        all_match = True
        for word_list in kw['words']:
            in_current_list = False
            for word in word_list:
                if word in content:
                    in_current_list = True
                    break
            if not in_current_list:
                all_match = False
                break
        if all_match:
            if kw['type'] == 'random':
                if random.random() < DISCARD_PERCENTAGE:
                    # 非必备规则，以一定概率放弃匹配
                    return userid, MSG_NONE, 'rule not match'
            # 匹配
            reply = choice(kw['reply'])
            return userid, reply[0], reply[1]

    # 没有匹配到
    # reply = choice(corpus.hehe)
    return userid, MSG_NONE, 'rule not match'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
