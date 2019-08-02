config = {
    'display_step': 10,
    'rnn_size': 256,
    'rnn_type': "lstm",
    'rnn_layers': 2,
    'embedding_dim': 300,
    'beam_width': 3,
    'batch_size': 200,
    'checkpoint_path': "data/stone/model_no_hidden_state/best_model.ckpt",
    'word_dict_path': "data/voc",
    'train_data': "data/shuffled_corpus.txt",
    'global_step': 100000,
    'learning_rate': 0.0001,
    'dropout_keep_prob': 0.7,
    'log': "data/stone/log",
    "inference_path": "/data/stone/data/preprocessing/tunning_test",
    "attention": True,
    "decode_bidirection": True,
    "dataset_buffer": 100000
}
