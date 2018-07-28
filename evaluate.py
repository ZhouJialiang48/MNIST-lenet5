#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 17:09
# @Author  : zhoujl
# @Site    : 
# @File    : evaluate.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import forward
from config import *


def evaluate(mnist):
    x = tf.placeholder(tf.float32, shape=[mnist.test.num_examples, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_NODE])
    y = forward.forward(x, train=False, regularizer=None)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    ema_restore = ema.variables_to_restore()
    saver = tf.train.Saver(ema_restore)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                reshaped_x = np.reshape(mnist.test.images,
                                        [mnist.test.num_examples, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
                accuracy_val = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
                print('Iter {}, accuracy is {}'.format(global_step, accuracy_val))
            else:
                print('No checkpoint file found')
                return
        time.sleep(5)


def main():
    mnist = input_data.read_data_sets('~/tf/data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
