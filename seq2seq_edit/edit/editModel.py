import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import embed_sequence
from tensorflow.python.util import nest


def bulid_rnn_cell(rnn_type, rnn_size, rnn_layer, dropout_keep_prob=1):
    def create_single_rnn():
        if rnn_type == 'lstm':
            cell = tf.contrib.rnn.LSTMCell(rnn_size)
        else:
            cell = tf.contrib.rnn.GRUCell(rnn_size)

        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)
        return cell
    return tf.contrib.rnn.MultiRNNCell([create_single_rnn() for _ in range(rnn_layer)])


def build_encoder_layer(input_data, input_data_len,
                        rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                        embedding, encode_bi=True):
    with tf.variable_scope("seq2seq_encoder"):
        emb_data_input = tf.nn.embedding_lookup(embedding, input_data)

        if encode_bi:
            print("decode use bi-"+rnn_type+"...")
            enc_cell_fw = bulid_rnn_cell(
                rnn_type, rnn_size, rnn_layers, dropout_keep_prob)
            enc_cell_bw = bulid_rnn_cell(
                rnn_type, rnn_size, rnn_layers, dropout_keep_prob)

            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                enc_cell_fw, enc_cell_bw, emb_data_input, sequence_length=input_data_len, dtype=tf.float32)

            enc_output = tf.concat([enc_output[0], enc_output[1]], -1)

            new_enc_state = []
            for i in range(len(enc_state[0])):
                new_enc_state_c = tf.concat(
                    [enc_state[0][i].c, enc_state[0][i].c], -1)
                new_enc_state_h = tf.concat(
                    [enc_state[1][i].h, enc_state[1][i].h], -1)
                new_enc_state.append(tf.contrib.rnn.LSTMStateTuple(
                    c=new_enc_state_c, h=new_enc_state_h))

            return enc_output, tuple(new_enc_state)
        else:
            enc_cell = bulid_rnn_cell(rnn_type, rnn_size, rnn_layers)
            enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, emb_data_input,
                                                      sequence_length=input_data_len, dtype=tf.float32)

            return enc_output, enc_state


def preprocessing_decoder_input(input_data, go_int, batch_size):
    with tf.variable_scope("preprocessing_decoder_input"):
        endding = tf.strided_slice(
            input_data, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], go_int), endding], 1)
    return dec_input


def build_decoder_layer(enc_output, enc_input_len, enc_state,
                        dec_input, dec_input_len, max_dec_len,
                        rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                        beam_width, beam_search,
                        embedding, vocab_size,
                        go_int, eos_int, is_attention=True):
    with tf.variable_scope("seq2seq_decoder"):
        # embedding
        dec_cell = bulid_rnn_cell(
            rnn_type, rnn_size, rnn_layers, dropout_keep_prob)

        # attention
        if is_attention:
            print("decode use attention...")

            if beam_search:
                enc_output = tf.contrib.seq2seq.tile_batch(
                    enc_output, beam_width)
                enc_input_len = tf.contrib.seq2seq.tile_batch(
                    enc_input_len, beam_width)

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                rnn_size, enc_output, memory_sequence_length=enc_input_len, name="attention")
            dec_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_cell, attention_mechanism, attention_layer_size=rnn_size)

        output_layer = tf.layers.Dense(
            vocab_size, kernel_initializer=tf.truncated_normal_initializer(0, 0.1))

        # train graph
        with tf.variable_scope("decode"):
            attention_enc_state = dec_cell.zero_state(
                tf.shape(enc_state)[2], tf.float32)
            # .clone(cell_state=enc_state)
            # dec_input_len = tf.Print(dec_input_len, [dec_input_len])
            dec_embedding_input = tf.nn.embedding_lookup(embedding, dec_input)
            training_helper = tf.contrib.seq2seq.TrainingHelper(
                dec_embedding_input, sequence_length=dec_input_len, time_major=False)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(
                dec_cell, training_helper, attention_enc_state, output_layer)
            training_output = tf.contrib.seq2seq.dynamic_decode(
                training_decoder, impute_finished=False, maximum_iterations=max_dec_len)[0]

        # inference graph
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile([go_int],
                                   [tf.shape(enc_state)[2]], name="start_tokens")
            if beam_search:
                attention_enc_state = nest.map_structure(
                    lambda s: tf.contrib.seq2seq.tile_batch(s, beam_width), enc_state)

                if is_attention:
                    attention_enc_state = dec_cell.zero_state(
                        tf.shape(attention_enc_state)[2], tf.float32)
                    # .clone(cell_state=attention_enc_state)

                inference_decode = tf.contrib.seq2seq.BeamSearchDecoder(
                    dec_cell, embedding, start_tokens, eos_int,
                    attention_enc_state, beam_width, output_layer=output_layer
                )
                inference_output = tf.contrib.seq2seq.dynamic_decode(
                    inference_decode, impute_finished=False, maximum_iterations=max_dec_len)[0]
            else:
                if is_attention:
                    attention_enc_state = dec_cell.zero_state(
                        tf.shape(enc_state)[2], tf.float32)
                    # .clone(cell_state=enc_state)

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    embedding, start_tokens, eos_int)
                inference_decode = tf.contrib.seq2seq.BasicDecoder(
                    dec_cell, inference_helper, attention_enc_state, output_layer)
                inference_output = tf.contrib.seq2seq.dynamic_decode(
                    inference_decode, impute_finished=True, maximum_iterations=max_dec_len)[0]

            training_logits = tf.identity(training_output.rnn_output, 'logits')
            training_predict = tf.identity(
                training_output.sample_id, "train_prediction")

            if beam_search:
                training_logits = inference_output.beam_search_decoder_output
                inference_predict = inference_output.predicted_ids
            else:
                inference_predict = tf.identity(
                    inference_output.sample_id, "train_prediction")
        return training_logits, training_predict, inference_predict


def build_multiseq2seq_model(enc_input_query, enc_input_len_query, enc_input_response, enc_input_len_response,
                        dec_input, dec_input_len, max_dec_len,
                        rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                        beam_width, beam_search, embedding, vocab_size,
                        go_int, eos_int,
                        encode_bi=True, attention=True):
    enc_output_query, enc_state_query = build_encoder_layer(enc_input_query, enc_input_len_query,
                                                rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                                                embedding, encode_bi)
    enc_output_response, enc_state_response = build_encoder_layer(enc_input_response, enc_input_len_response,
                                                rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                                                embedding, encode_bi)

    # enc_output = tf.Print(enc_output,
    #                       [tf.shape(enc_output),
    #                        tf.shape(enc_state)[2],
    #                        tf.shape(dec_input),
    #                        tf.shape(enc_input_len)])

    # 如果双层的话，decode要翻倍
    rnn_size = rnn_size*2 if encode_bi else rnn_size
    
    enc_output = tf.layers.Dense(tf.concat([enc_output_query, enc_output_response]), rnn_size=tf.nn.relu)
    enc_intput_len = tf.concat([enc_input_len_query, enc_input_len_response])
    enc_state = tf.concat([enc_state_query, enc_state_response])

    training_logits, training_prediction, inference_prediction = build_decoder_layer(enc_output, enc_input_len, enc_state,
                                                                                     dec_input, dec_input_len, max_dec_len,
                                                                                     rnn_type, rnn_size, rnn_layers, dropout_keep_prob,
                                                                                     beam_width, beam_search,
                                                                                     embedding, vocab_size,
                                                                                     go_int, eos_int,
                                                                                     is_attention=attention)

    return training_logits, training_prediction, inference_prediction
