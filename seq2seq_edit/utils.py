import tensorflow as tf
from functools import reduce
from operator import mul
import pickle
import numpy as np
from tensorflow.python.ops import lookup_ops


def get_word_dict(path):
    word_dict = []

    with open(path) as f:
        for line in f:
            word = line.strip()
            word_dict.append(word)

    word2idx = {w: i for i, w in enumerate(word_dict)}
    idx2word = {i: w for i, w in enumerate(word_dict)}
    word_size = len(word_dict)

    return word2idx, idx2word, word_size


def get_preTraining_embedding(path):
    with open(path, 'rb') as f:
        embedding = pickle.load(f)
    print("load pre training embedding, shape=", np.shape(embedding))
    return tf.constant(np.asarray(embedding), dtype=tf.float32), len(embedding)


def load_train_data(path):
    ask, ans = [], []
    with open(path) as f:
        print("loadding train data from %s ..." % path)
        for line in f:
            l = line.strip().split("\t")
            if len(l) == 2:
                ask.append(l[0])
                ans.append(l[1])

    print("loadding %d data" % len(ask))
    return ask, ans


def load_inference_data(path):
    ask = []

    with open(path) as f:
        for line in f:
            ask.append(line.strip())

    return ask


def text2seq(text, word2idx, is_target=False):
    rs = []
    for t in text:
        seq = []
        for w in t.split(' '):
            seq.append(word2idx.get(w, word2idx['<UNK>']))

        rs.append(seq)
    return rs


def seq2text(seq, idx2word, pad_int, eos_int=1, hidden=False):
    if hidden:
        return [idx2word[idx] for idx in seq if idx != pad_int and idx != eos_int]
    else:
        return [idx2word[idx] for idx in seq]


def get_batch(ask, ans, batch_size, word2idx):
    batch_count = int(len(ask)/batch_size)
    pad_int = word2idx['<PAD>']
    eos_int = word2idx['<EOS>']

    ans = [q+[eos_int] for q in ans]

    ask_len = [len(q) for q in ask]
    ans_len = [len(q) for q in ans]

    batch_ask_list, batch_ans_list, batch_ask_len_list, batch_ans_len_list = [], [], [], []

    for batch_i in range(batch_count):
        offset = batch_i*batch_size
        offset_max = offset+batch_size  # 必须保证每个batch的大小

        batch_ask = ask[offset:offset_max]
        batch_ans = ans[offset:offset_max]
        batch_ask_len = ask_len[offset:offset_max]
        batch_ans_len = ans_len[offset:offset_max]

        batch_ask = padding(batch_ask, pad_int)
        batch_ans = padding(batch_ans, pad_int)

        batch_ask_list.append(batch_ask)
        batch_ans_list.append(batch_ans)
        batch_ask_len_list.append(batch_ask_len)
        batch_ans_len_list.append(batch_ans_len)

    return batch_ask_list, batch_ans_list, batch_ask_len_list, batch_ans_len_list


def padding(seqList, pad_int):
    max_len = max([len(s) for s in seqList])
    return [seq+[pad_int]*(max_len-len(seq)) for seq in seqList]


def get_input():
    ask_input = tf.placeholder(tf.int32, [None, None], name="ask-input")
    ans_input = tf.placeholder(tf.int32, [None, None], name="ans-input")
    lr = tf.placeholder(tf.float32, name='learning-rate')

    ask_input_len = tf.placeholder(tf.int32, [None], name="ask-input-len")
    ans_input_len = tf.placeholder(tf.int32, [None], name="ans-input-len")
    max_ans_input_len = tf.reduce_max(ans_input_len, name="max-ask-input-len")

    ans_input_mask = tf.sequence_mask(
        ans_input_len, max_ans_input_len, dtype=tf.float32, name="ans_input_masks")
    return ask_input, ans_input, lr, ask_input_len, ans_input_len, max_ans_input_len, ans_input_mask


def get_params_info():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def get_lookup_table(wordFile):
    unk_int = 0
    # wordFile = "/data/stone/data/word_dict_v2.tsv"
    wordTable = lookup_ops.index_table_from_file(
        wordFile, default_value=unk_int, delimiter='\n')
    return wordTable


def get_dataset(path, wordTable, batch_size, buffer_size=100000):
    with tf.variable_scope("input"):
        eos_int = tf.cast(wordTable.lookup(tf.constant("<EOS>")), tf.int32)
        go_int = tf.cast(wordTable.lookup(tf.constant("<GO>")), tf.int32)
        pad_int = tf.cast(wordTable.lookup(tf.constant("<PAD>")), tf.int32)

        fileList = tf.data.Dataset.from_tensor_slices([path])
        dataset = fileList.flat_map(lambda filename:
                                    tf.data.TextLineDataset(filename))

        dataset = dataset.map(
            lambda x: tf.string_split([x], delimiter='\t').values)
        dataset = dataset.map(lambda x: (tf.string_split(
            [x[0]]).values, tf.string_split([x[1]]).values))
        dataset = dataset.map(lambda ask, ans:
                              (tf.cast(wordTable.lookup(ask), tf.int32),
                               tf.cast(wordTable.lookup(ans), tf.int32)))

        dataset = dataset.map(
            lambda ask, ans: (ask[:30], ans[:30]))

        dataset = dataset.map(lambda ask, ans: (
            ask,
            tf.concat(([go_int], ans), 0),
            tf.concat((ans, [eos_int]), 0)
        ))
        dataset = dataset.map(lambda ask, ans_in, ans_out: (
            ask,
            ans_in,
            ans_out,
            tf.size(ask),
            tf.size(ans_out)
        ))
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=([None], [None], [
                                           None], [], []),
                                       padding_values=(pad_int, pad_int, pad_int, 0, 0))

        dataset = dataset.prefetch(buffer_size)

        iterator = dataset.make_initializable_iterator()
        ask, ans_in, ans_out, ask_len, ans_len = iterator.get_next()

        max_ans_len = tf.reduce_max(ans_len)

        ans_mask = tf.sequence_mask(
            ans_len, max_ans_len, dtype=tf.float32, name="ans_input_masks")

    return ask, ans_in, ans_out, ask_len, ans_len, ans_mask, max_ans_len, iterator
