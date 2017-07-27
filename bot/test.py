import tensorflow.contrib.slim as slim
import tensorflow as tf

with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                    weights_regularizer=slim.l2_regularizer(0.0005)):
    inputs = tf.placeholder(tf.float32, [None, 10], name='s')
    net = slim.fully_connected(inputs, 1000)
