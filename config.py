#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/28 15:09
# @Author  : zhoujl
# @Site    : 
# @File    : config.py
# @Software: PyCharm

DATA_DIR = '~/tf/data/'

# forward.py
IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

# backward.py
STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'mnist_model'
CYCLE_OBSERVED = 100

# evaluate.py
TEST_INTERVAL_SECS = 5
