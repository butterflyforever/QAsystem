import tensorflow as tf
import numpy as np
import utils
import os
from config import config
import seq2seqModel
from flask import Flask, request, jsonify
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import jieba
import json
import random
app = Flask(__name__)

# build inference graph
loadding_graph = tf.Graph().as_default()

word2idx, idx2word, vocab_size = utils.get_word_dict(config['word_dict_path'])

lr = tf.placeholder(tf.float32, shape=[])
dropout_keep_prob = tf.placeholder(tf.float32, shape=[])
wordTable = utils.get_lookup_table(config['word_dict_path'])

ask = tf.placeholder(tf.int32, [None, None])
ask_len = tf.placeholder(tf.int32, [None])
ans_in = tf.placeholder(tf.int32, [None, None])
ans_len = tf.placeholder(tf.int32, [None])
max_ans_len = tf.constant(30)

embedding = tf.Variable(tf.random_normal(
    [vocab_size, config['embedding_dim']]), name="embedding", dtype=tf.float32)

go_int = tf.cast(wordTable.lookup(tf.constant('<GO>')), tf.int32)
eos_int = tf.cast(wordTable.lookup(tf.constant('<EOS>')), tf.int32)

training_logits, training_predict, inference_predict = seq2seqModel.build_seq2seq_model(
    ask, ask_len, ans_in, ans_len, max_ans_len,
    config['rnn_type'], config['rnn_size'], config['rnn_layers'], dropout_keep_prob,
    config['beam_width'], True,
    embedding, vocab_size,
    go_int, eos_int
)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True


@app.route('/', methods=['GET', 'POST'])
def get_response():
    try:
        print(request.get_json(force=True))
        req = request.get_json(force=True)[0]['content']
        resp = inference(req)
        print("ask:%s\nans:%s" % (req, resp))

        resp = [x for x in resp if x.find("UNK") < 0]
        if len(resp) == 0:
            msg = [
                {'msgType': 'none', 'content': ''}
            ]
            return jsonify(msg)

        msg = []
        for x in resp:
            msg.append({
                'msgType': "text",
                'content': x
            })
        return jsonify(msg)
    except:
        msg = [{
            'msgType': "text",
            'content': "你妹啊，完全不知道应该回你什么"
        }]
        return jsonify(msg)


def inference(msg):
    input_data = []
    for w in msg:
        input_data.append(word2idx.get(w, word2idx['<UNK>']))

    print(input_data)
    feed_dict = {
        ask: [input_data],
        ask_len: [len(msg)],
        dropout_keep_prob: config['dropout_keep_prob']
    }

    print(training_logits)

    loss, ans_prediction = sess.run(
        [training_logits, inference_predict], feed_dict=feed_dict)

    # print("loss: \n", loss)

    ans_prediction = ans_prediction[0]
    new_ans_prediction = [[] for _ in range(np.shape(ans_prediction)[1])]
    for a in range(np.shape(ans_prediction)[0]):
        for b in range(np.shape(ans_prediction)[1]):
            new_ans_prediction[b].append(ans_prediction[a][b])

    print(new_ans_prediction)

    pad_int = word2idx['<PAD>']
    eos_int = word2idx['<EOS>']

    new_response = []
    for x in new_ans_prediction:
        new_response.append(
            "".join(utils.seq2text(x, idx2word, pad_int, eos_int, True)))

    print(new_response)

    random.shuffle(new_response)
    return new_response


with tf.Session(config=sess_config) as sess:
    sess.run(tf.tables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,
                  tf.train.latest_checkpoint(os.path.dirname(config['checkpoint_path'])))

    if __name__ == '__main__':
        app.debug = True
        app.run(host='0.0.0.0', port=5001)
