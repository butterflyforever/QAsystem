import utils
import tensorflow as tf
import seq2seqModel
import numpy as np
import time
from config import config
import sys
import random
import shutil

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

word2idx, idx2word, vocab_size = utils.get_word_dict(config['word_dict_path'])

# build training graph
print("--------------------------build training graph----------------")
train_graph = tf.Graph()

with train_graph.as_default():
    lr = tf.placeholder(tf.float32, shape=[])
    dropout_keep_prob = tf.placeholder(tf.float32, shape=[])
    wordTable = utils.get_lookup_table(config['word_dict_path'])
    ask, ans_in, ans_out, ask_len, ans_len, ans_mask, max_ans_len, iterator = utils.get_dataset(
        config['train_data'], wordTable, config['batch_size'])

    # ask = tf.Print(ask, [ask[0], ans_out[0]])

    # embedding = utils.get_preTraining_embedding()
    embedding = tf.Variable(tf.random_normal(
        [vocab_size, config['embedding_dim']]), name="embedding", dtype=tf.float32)
    go_int = tf.cast(wordTable.lookup(tf.constant('<GO>')), tf.int32)
    eos_int = tf.cast(wordTable.lookup(tf.constant('<EOS>')), tf.int32)

    training_logits, training_predict, inference_predict = seq2seqModel.build_seq2seq_model(
        ask, ask_len, ans_in, ans_len, max_ans_len,
        config['rnn_type'], config['rnn_size'], config['rnn_layers'], dropout_keep_prob,
        config['beam_width'], False,
        embedding, vocab_size,
        go_int, eos_int
    )

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits, ans_out, ans_mask)

        tf.summary.scalar("cost", cost)
        merged_summary = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer()
        gradients = optimizer.compute_gradients(cost)
        copped_gradients = [(tf.clip_by_value(grad, -5.0, 5.0), var)
                            for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(copped_gradients)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

with tf.Session(graph=train_graph, config=sess_config) as sess:
    saver = tf.train.Saver()
    # load params
    sess.run(tf.tables_initializer())
    sess.run(iterator.initializer)
    if 'new' not in sys.argv:
        print("load last params...")

        saver.restore(sess,
                      tf.train.latest_checkpoint(os.path.dirname(config['checkpoint_path'])))
    else:
        if os.path.exists(config['log']):
            shutil.rmtree(config['log'])
        sess.run([tf.global_variables_initializer()])

    writer = tf.summary.FileWriter(config['log'], sess.graph)

    print("all params: "+str(utils.get_params_info()))
    print("vocab: %d" % len(word2idx))
    print("--------------------------begin training----------------------")

    min_loss = 100000
    for global_step in range(config['global_step']):
        try:
            start = time.time()
            feed_dict = {
                lr: config['learning_rate'],
                dropout_keep_prob: config['dropout_keep_prob']
            }
            _, loss, pre = sess.run(
                [train_op, cost, training_predict], feed_dict=feed_dict)
            # print(loss)
            # print(pre)
            # print(global_step)
            # print(feed_dict)
            if global_step % config['display_step'] == 0 and global_step > 0:
                end = time.time()
                each_step = (end-start) / (config['display_step'])
                start = time.time()

                feed_dict = {
                    lr: config['learning_rate'],
                    dropout_keep_prob: config['dropout_keep_prob']
                }

                # print(feed_dict)

                loss, batch_ask, batch_ans, pre, summary = sess.run(
                    [cost, ask, ans_out, inference_predict, merged_summary], feed_dict=feed_dict)


                writer.add_summary(summary, global_step=global_step)

                print("valid global step %d/%d,loss %.4f,each batch %.2f s" %
                      (global_step, config['global_step'], loss, each_step))

                for a, b, c in zip(batch_ask[:3], batch_ans[:3], pre[:3]):
                    print("ask: "+" ".join(utils.seq2text(
                        a, idx2word, word2idx['<PAD>'])))
                    print("ans: "+" ".join(utils.seq2text(
                        b, idx2word, word2idx['<PAD>'])))
                    print("pre: "+" ".join(utils.seq2text(
                        c, idx2word, word2idx['<PAD>'])))
                    print('-------------------------')

                # if min_loss > loss:
                min_loss = loss
                saver.save(sess, config['checkpoint_path'])

                end = time.time()
                each_step = end-start
                start = time.time()
                print("model save to " +
                      config['checkpoint_path']+", %.2f s" % each_step)
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
