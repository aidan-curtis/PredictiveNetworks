__author__ = 'victor'

import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell

def rnn(images, mask_true, num_layers, num_hidden, filter_size, stride=1,
        seq_length=20, input_length=10, tln=True):
    num_hidden = num_hidden[0]
    gen_images = []
    shape = images.get_shape().as_list()
    output_channels = shape[-1]


    lstm = ConvLSTMCell(
                conv_ndims=2,
                input_shape=[shape[2], shape[3], num_hidden],
                output_channels=num_hidden,
                kernel_shape=[filter_size, filter_size],
                name='lstm'
            )

    reuse = bool(gen_images)
    with tf.variable_scope('stacked_convlstm', reuse=reuse):
        cell = tf.zeros([shape[0], shape[2], shape[3], num_hidden], dtype=tf.float32)
        hidden = tf.zeros([shape[0], shape[2], shape[3], num_hidden], dtype=tf.float32)

    for t in xrange(seq_length-1):
        reuse = bool(gen_images)
        with tf.variable_scope('convlstm', reuse=reuse):
            if t < input_length:
                inputs = images[:,t]
            else:
                inputs = mask_true[:,t-10]*images[:,t] + (1-mask_true[:,t-10])*x_gen

            _, new_state = lstm(inputs, tf.contrib.rnn.LSTMStateTuple(c=cell, h=hidden))
            cell, hidden = new_state

            x_gen = tf.layers.conv2d(inputs=hidden,
                                     filters=output_channels,
                                     kernel_size=1,
                                     strides=1,
                                     padding='same',
                                     name="back_to_pixel")
            gen_images.append(x_gen)

    gen_images = tf.stack(gen_images)
    # [batch_size, seq_length, height, width, channels]
    gen_images = tf.transpose(gen_images, [1,0,2,3,4])
    loss = tf.nn.l2_loss(gen_images - images[:,1:])
    #loss += tf.reduce_sum(tf.abs(gen_images - images[:,1:]))
    return [gen_images, loss]