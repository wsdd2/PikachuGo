# -*- coding: utf-8 -*-
import numpy as np
import board
import go_plot
import os
import util
import sys
import random

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
"""
np.set_printoptions(threshold='nan')
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)


def compress_manual(manual):
    assert isinstance(manual, np.ndarray)
    manual = manual.reshape((16, 361))
    return np.dot(exp2, manual)

path = os.listdir('E:/value_shuffle')


first = True
data = None
label = None
i = 0
for file in path:
    n_data = np.load("E:/value_shuffle/" + file)
    n_label = np.load("E:/value_shuffle_label/" + file)

    # print file, data.shape, label.shape
    if first is False:
        data = np.vstack((data, n_data))
        label = np.vstack((label, n_label))
    else:
        data = n_data
        label = n_label
        first = False
    print i
    i += 1
    if i % 10 == 0:
        permutation = np.random.permutation(data.shape[0])
        shuffled_dataset = data[permutation, :]
        shuffled_labels = label[permutation, :]
        print shuffled_dataset.shape
        print shuffled_labels.shape
        np.save('E:/value_shuffle_2/' + str(i), shuffled_dataset)
        np.save('E:/value_shuffle_label_2/' + str(i), shuffled_labels)
        data = None
        label = None
        first = True

permutation = np.random.permutation(data.shape[0])
shuffled_dataset = data[permutation, :]
shuffled_labels = label[permutation, :]
print shuffled_dataset.shape
print shuffled_labels.shape
np.save('E:/value_shuffle_2/' + str(i), shuffled_dataset)
np.save('E:/value_shuffle_label_2/' + str(i), shuffled_labels)




"""
for file in path:
    data = np.load("E:/npy/" + file)
    label = np.load("E:/label_npy/" + file)

    # print file, data.shape, label.shape
    

    droplst = []
    rd = np.random.rand(label.shape[0])
    for i in range(label.shape[0]):
        if label[i][1] == +1:
            if rd[i] > 0.5:
                droplst.append(i)
        elif label[i][1] == -1:
            if rd[i] > 0.5:
                droplst.append(i)

    data = np.delete(data, droplst, axis=0)
    label = np.delete(label, droplst, axis=0)

    np.save('E:/c_npy_value/' + file, data)
    np.save('E:/c_label_npy_value/' + file, label)

    # print data.shape, label.shape

"""

"""
print data.shape
print label.shape
aaa = data.reshape(-1, 1, 361)
exp2 = exp2.T
exp2 = exp2.reshape(1, 16, 1)
recover = ((np.bitwise_and(aaa, exp2) > 0) + 0).reshape(-1, 16, 19, 19)
cnt1, cnt2 = 0, 0
for i in range(label.shape[0]):
    if label[i][1] == 1:
        cnt1 += 1
    else:
        cnt2 += 1

print cnt1, cnt2

"""






"""
shape = (n, 2)
第一个数是预测点，第二个数是胜负
"""

"""
for file in path:
    fp = open('D:/Process/' + str(block) + '/' + file)
    con = fp.read()
    res = con.split('|')
    BLACK_LABEL = 0
    WHITE_LABEL = 0
    if res[1] == 'B':
        BLACK_LABEL = 1
        WHITE_LABEL = -1
    elif res[1] == 'W':
        WHITE_LABEL = 1
        BLACK_LABEL = -1

    res = res[3: ]
    g = board.Go()
    dataset = g.generate()
    compressed = compress_manual(dataset)

    if g.current_player() == board.SIDE_BLACK:
        labelset = np.array([[util.sgf_to_num(res[0]), BLACK_LABEL]], dtype=np.int16)
    else:
        labelset = np.array([[util.sgf_to_num(res[0]), WHITE_LABEL]], dtype=np.int16)

    g.place_stone(res[0])
    for i in range(1, len(res)):
        if res[i] == 'tt':
            break
        f = g.generate()
        f = compress_manual(f)
        compressed = np.vstack((compressed, f))
        if g.current_player() == board.SIDE_BLACK:
            lab_x = np.array([[util.sgf_to_num(res[i]), BLACK_LABEL]], dtype=np.int16)
        else:
            lab_x = np.array([[util.sgf_to_num(res[i]), WHITE_LABEL]], dtype=np.int16)
        labelset = np.vstack((labelset, lab_x))

        g.place_stone(res[i])
    np.save('D:/npy/' + file, compressed)
    np.save('D:/label_npy/' + file, labelset)
    print file, compressed.shape, labelset.shape
    fp.close()

"""


"""
        
        
        aaa = compressed[0].reshape(-1, 1, 361)
        exp2 = exp2.T
        exp2 = exp2.reshape(1, 16, 1)
        recover = ((np.bitwise_and(aaa, exp2) > 0) + 0).reshape(-1, 16, 19, 19)
        print recover[0]
        print labelset[0]
"""





















"""

aaa = np.load('D:/npy/1998-02-05a.sgf.npy')
aaa = aaa.reshape(155, 1, 361)
exp2 = exp2.T
exp2 = exp2.reshape(1, 16, 1)
recover = ((np.bitwise_and(aaa, exp2) > 0) + 0).reshape(-1, 16, 19, 19)
for x in recover:
    print x
    while raw_input() != 'n':
        pass

lst = ["本方棋子",
       "对方棋子",
       "空点",
       "1qi",
       "2qi",
       "3qi",
       "4qi",
       "5qi", "前3手", "前1手", "极有可能眼位",
       "眼位","打劫", "本方征子不利", "对方征子不利", "本方是否执黑"]
i = 0
for x in dataset:
    print i
    for ii in range(16):
        print lst[ii]
        print x[ii]
    raw_input()
    i += 1
"""