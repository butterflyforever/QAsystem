import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config)

