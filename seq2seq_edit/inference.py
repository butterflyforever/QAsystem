import tensorflow as tf
import numpy as np
import utils
import os
from config import config
import seq2seqModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

# load data
ask_data = utils.load_inference_data(config['inference_path'])
ask_data = utils.text2seq(ask_data, word2idx)
ask_data_len = [len(x) for x in ask_data]
ask_data = utils.padding(ask_data, word2idx['<PAD>'])

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    sess.run(tf.tables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,
                  tf.train.latest_checkpoint(os.path.dirname(config['checkpoint_path'])))

    feed_dict = {
        ask: ask_data,
        ask_len: ask_data_len,
        dropout_keep_prob: 1
    }

    ans_prediction = sess.run([inference_predict], feed_dict=feed_dict)[0]

    new_ans_prediction = [[] for _ in range(np.shape(ans_prediction)[0])]
    for a in range(np.shape(ans_prediction)[0]):
        for b in range(np.shape(ans_prediction)[1]):
            new_ans_prediction[a].append(ans_prediction[a][b][0])

    ans_prediction = new_ans_prediction
    print(new_ans_prediction[0])
    print(np.shape(new_ans_prediction))

    pad_int = word2idx['<PAD>']
    eos_int = word2idx['<EOS>']

    for rs_ask, rs_ans in zip(ask_data, ans_prediction):
        ask_seq = " ".join(utils.seq2text(
            rs_ask, idx2word, pad_int, eos_int, True))
        ans_seq = " ".join(utils.seq2text(
            rs_ans, idx2word, pad_int, eos_int, True))
        print("ask: %s\nans: %s\n" % (ask_seq, ans_seq))
