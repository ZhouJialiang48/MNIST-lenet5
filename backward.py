#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 15:04
# @Author  : zhoujl
# @Site    : 
# @File    : backward.py
# @Software: PyCharm
import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
from config import *


def backward(mnist):
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    y_ = tf.placeholder(tf.float32, shape=(None, OUTPUT_NODE))
    y = forward.forward(x, train=True, regularizer=REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    # 交叉熵
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 参数滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    # 依赖控制，合并前两步操作
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        # 全局变量初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # 每训练CYCLE_OBSERVED轮数，打印训练信息
            if i % CYCLE_OBSERVED == 0:
                print('Iter {}, loss is {}'.format(step, loss_val))
                saver.save(sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('~/tf/data/', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
