# -*- coding: utf-8 -*-
import numpy as np
import os
import gc
import time
import struct
import sys
import multiprocessing
import random
import mxnet as mx
from mxnet import nd
from mxnet.ndarray import concatenate
from mxnet.io import DataIter, DataDesc
from collections import OrderedDict, namedtuple
import config
import symmetry
import copy
import board
import util
import math
import go_plot
# 如果需要打印numpy数组，取消下面这一行的注释
np.set_printoptions(threshold='nan')
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=1000)


"""

手动定义一个softmax函数

这里我想 取 top k 个可选点，再用一次softmax, 来随机选取下一个点位

"""

path = "E:/PikachuGoNN/ReinforcementNN/numpy_data/"

K = 10

sym, arg_params, aux_params = mx.model.load_checkpoint(config.model_prefix, config.n_epoch_load)
module = mx.mod.Module(symbol=sym, context=config.train_device)
go = board.Go()
feature = go.generate()

iter = mx.io.NDArrayIter(data=feature)
module.bind(for_training=True, data_shapes=iter.provide_data)
module.set_params(arg_params, aux_params)



def softmax(lst):
    total = 0
    for x in lst:
        total += math.exp(x)
    return [math.exp(x) / total for x in lst]


def random_pick(lst):
    # print softmax(lst)
    index = np.random.choice([i for i in range(len(lst))], p = softmax(lst))
    return index
# l = [2.4, 2.2, 2.1, 0.1, 0.1]





def self_play():
    global module
    label_black = []
    label_white = []


    t0 = time.clock()

    black = None
    white = None


    TOTAL = 350
    NUM = 300
    go = board.Go()
    for i in range(TOTAL):
        # 生成特征
        if i % 10 == 0:
            print i,
        feature = go.generate()
        # 把特征喂入神经网络，获得预测
        iter = mx.io.NDArrayIter(data=feature)
        pred = module.predict(iter).asnumpy()
        # 将不可入点扔掉
        pred = feature[0][2].reshape(1, 361) * pred * 10
        # 排个序
        out = np.argsort(-pred)
        predsort = - np.sort(-pred)
        # print predsort[0: K]
        # 看看前K个的值
        process = out[0][0: K]
        idx = random_pick(predsort[0][0: K].reshape(-1))

        if i == 0:
            black = feature.reshape(-1, 16, 19, 19)
            label_black.append(out[0][idx])
            go.place_stone_num(out[0][idx])
        elif i == 1:
            white = feature.reshape(-1, 16, 19, 19)
            label_white.append(out[0][idx])
            go.place_stone_num(out[0][idx])
        elif i < NUM:
            f = feature.reshape(-1, 16, 19, 19)
            if i % 2 == 0:
                black = np.vstack((black, f))
                label_black.append(out[0][idx])
            else:
                white = np.vstack((white, f))
                label_white.append(out[0][idx])
            go.place_stone_num(out[0][idx])
        else:
            go.place_stone_num(out[0][0])

    go = None
    go = board.Go()

    # print len(label_black)
    for i in range(TOTAL):
        # 生成特征
        if i % 10 == 0:
            print i,
        feature = go.generate()
        # 把特征喂入神经网络，获得预测
        iter = mx.io.NDArrayIter(data=feature)
        pred = module.predict(iter).asnumpy()
        # 将不可入点扔掉
        pred = feature[0][2].reshape(1, 361) * pred * 10
        # 排个序
        out = np.argsort(-pred)
        predsort = - np.sort(-pred)
        # print predsort[0: K]
        # 看看前K个的值
        process = out[0][0: K]
        idx = random_pick(predsort[0][0: K].reshape(-1))

        if i < NUM:
            f = feature.reshape(-1, 16, 19, 19)
            if i % 2 == 0:
                black = np.vstack((black, f))
                label_black.append(out[0][idx])
            else:
                white = np.vstack((white, f))
                label_white.append(out[0][idx])
            go.place_stone_num(out[0][idx])
        else:
            go.place_stone_num(out[0][0])
    print
    cnt = 0
    for x in label_black:

        print x,
        cnt += 1
        if cnt == 150:
            print


    print label_white

    BLACK_LABEL = np.zeros((300, 361), dtype=np.int16)
    WHITE_LABEL = np.zeros((300, 361), dtype=np.int16)
    for i in range(600):
        if i % 2 == 0:  # 偶数，黑棋盘面
            # print "BLACK", i
            BLACK_LABEL[i // 2][label_black[i // 2]] = 1
        else:
            # print "WHITE", i
            WHITE_LABEL[(i - 1) // 2][label_white[(i - 1) // 2]] = 1

    permutation = np.random.permutation(black.shape[0])
    black = black[permutation, :]
    white = white[permutation, :]

    BLACK_LABEL = BLACK_LABEL.reshape(-1, 19, 19)
    WHITE_LABEL = WHITE_LABEL.reshape(-1, 19, 19)
    BLACK_LABEL = BLACK_LABEL.reshape(-1, 361)
    WHITE_LABEL = WHITE_LABEL.reshape(-1, 361)
    BLACK_LABEL = BLACK_LABEL[permutation, :]
    WHITE_LABEL = WHITE_LABEL[permutation, :]
    BLACK_LABEL = np.argmax(BLACK_LABEL, axis=1)
    WHITE_LABEL = np.argmax(WHITE_LABEL, axis=1)






    print BLACK_LABEL
    print WHITE_LABEL

    """
    # 判断胜负, 黑胜返回1，白胜利返回0
    def evaluate(self):

    """
    # go_plot.go_plot(terminal // 50)
    result = go.evaluate()

    if result == 1:
        print("B+")
    else:
        print("W+")

    print "generate_used", time.clock() - t0

    BLACK = black
    WHITE = white

    # print self.BLACK
    # print self.WHITE
    """
    0代表黑胜利，1代表白胜利
    """
    if result == 1:
        winner = 1
    else:
        winner = 0
    go = None
    label_black = None
    label_white = None

    return result


self_play()