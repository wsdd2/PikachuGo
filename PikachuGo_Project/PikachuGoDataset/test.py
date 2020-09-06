# -*- coding: utf-8 -*-
import numpy as np
import board
import go_plot
import os
import util
import sys
import random
import gc
"""
0->本方棋子的位置
1->对方棋子的位置
2->空点
3->1气
4->2气
5->3气
6->4气
7->5气
8->上3手
9->上一手
10->极有可能成为眼位
11->填充自己眼位的走子
12->打劫规则不允许的棋子
13->本方征子不利
14->对方征子不利
15->本方是否执黑

np.set_printoptions(threshold='nan')
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)


def compress_manual(manual):
    assert isinstance(manual, np.ndarray)
    manual = manual.reshape((16, 361))
    return np.dot(exp2, manual)


data = np.load('E:/target_data/0.npy')
label = np.load('E:/target_label/0.npy')


aaa = data.reshape(-1, 1, 361)
exp2 = exp2.T
exp2 = exp2.reshape(1, 16, 1)
recover = ((np.bitwise_and(aaa, exp2) > 0) + 0).reshape(-1, 16, 19, 19)

print recover[7]
print label[7]
"""
label_data = np.random.rand(19, 361)
label_data = label_data.reshape(-1, 361)
label_data = np.argmax(label_data, axis=1)
print label_data.shape