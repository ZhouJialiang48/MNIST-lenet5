#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 14:15
# @Author  : zhoujl
# @Site    : 
# @File    : forward.py
# @Software: PyCharm
import tensorflow as tf
from config import *


# 主方法，定义前向传播网络结构
def forward(x, train, regularizer=None):
    conv1_w = get_weight(shape=[CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer=regularizer)
    conv1_b = get_bias(shape=[CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight(shape=[CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer=regularizer)
    conv2_b = get_bias(shape=[CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    print(type(pool2), nodes, pool_shape)
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight(shape=[nodes, FC_SIZE], regularizer=regularizer)
    fc1_b = get_bias(shape=[FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    fc2_w = get_weight(shape=[FC_SIZE, OUTPUT_NODE], regularizer=regularizer)
    fc2_b = get_bias(shape=[OUTPUT_NODE])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    y = fc2
    return y


# 获取权重变量
def get_weight(shape, regularizer):
    """
    传入指定的shape和regularizer(lambda)
    返回tensorflow的Variable类型变量，用于优化weight
    """
    w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), dtype=tf.float32)
    if regularizer:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


# 获取偏置变量
def get_bias(shape):
    """
    传入指定的shape
    返回tensorflow的Variable类型变量，用于优化bias
    """
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


# 定义卷积操作
def conv2d(x, w):
    """
    :param x: 输入张量
    :param w: 卷积核
    """
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')


# 定义2x2池化操作
def max_pool_2x2(x):
    """
    :param x: 输入张量
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
