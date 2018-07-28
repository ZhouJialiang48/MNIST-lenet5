#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/27 10:32
# @Author  : zhoujl
# @Site    : 
# @File    : app.py
# @Software: PyCharm
import numpy as np
from PIL import Image
import tensorflow as tf
import forward
import backward
from config import *


def restore_model(test_pic_arr):
    with tf.Graph().as_default() as tg:

        x = tf.placeholder(tf.float32, shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        y = forward.forward(x, train=False, regularizer=None)
        pred_num = tf.argmax(y, 1)

        # 读取参数滑动平均值
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        saver = tf.train.Saver(ema.variables_to_restore())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                pred_num_val = sess.run(pred_num, feed_dict={x: test_pic_arr})
                return pred_num_val
            else:
                print('No checkpoint file found')
                return -1


def pre_img(picName, threshold=50, white_ground=True):
    """
    将输入的RGB图片转化为黑子白字的28*28numpy数组，使其符合模型接口
    :param picName: 图片路径
    :param threshold: 阈值，越低对数字笔记越敏感(0 <= threshold <= 255)
    :param white_ground: 默认输入白底黑字的RGB图片，需进行黑白反转
    :return:
    """
    img_rgb = Image.open(picName)
    img_rgb_resize = img_rgb.resize((28, 28), resample=Image.ANTIALIAS)
    # RGB转换成灰度图
    img_L = img_rgb_resize.convert(mode='L')
    # 图片类型转换为numpy数组类型
    img_arr = np.array(img_L)
    # 若输入图片为白底，需进行黑白反转
    if white_ground:
        img_arr = 255 - img_arr
    # 小于阈值则设为纯白， 否则为纯黑
    # 阈值大小与容错率相关
    img_ready = np.where(img_arr < threshold, 0, img_arr).reshape(1, 28, 28, 1)
    print(img_ready.shape)
    return img_ready


def application():
    test_num = input('Input the number of test picture: ')
    for i in range(int(test_num)):
        test_pic = input('Input the path of test pictures: ')
        # 功过图片与处理，将测试图片数组化，是指符合模型输入接口pip
        test_pic_arr = pre_img(test_pic)
        pred_num = restore_model(test_pic_arr)
        print('The prediction number is {}'.format(pred_num))


if __name__ == '__main__':
    application()
