# -*- coding: utf-8 -*-
import numpy as np
import board
import go_plot
import os
import util
import sys

block = sys.argv[1]
"""
                    STEP3           data_maker.py
                    
                    

粗略制作npy数据集
每一个sgf文件生成2个npy数据集，分别是盘面和标签。


1. 之后还会将它该数据集的部分进行筛选，防止过拟合
2. 之后还会将其数据集进行打乱

运行本py文件时，要指定参数：
参数1：block，即在根目录的哪一个文件夹下：



文件夹需要按1/2/3/4...来命名。
这么做的原因是可以同时开启多个Python脚本运行，加快数据处理的速度。

dataset的形状：
shape = (n, 16, 361)
经过了一次压缩
label的矩阵形状：
shape = (n, 2)      n行2列，第一列是预测点，第二列是胜负

"""


"""
------------------------------------参数设置-----------------------------------------
"""
"""
预处理之后的棋谱的根目录：在这个根目录之下还有文件夹1、2、3...
"""
root_dir                    = 'E:/PikachuGoDataSample/2_data_maker/'
"""
走子完毕，生成的npy数据集的存放位置（一个个样本）          保存位置1   PikachuP
"""
dataset_dir                 = 'E:/PikachuGoDataSample/3_data_maker_output/dataset/'
"""
走子完毕，生成的npy标签集的存放位置（一个个label）         保存位置2     PiakchuP
"""
label_dir                   = 'E:/PikachuGoDataSample/3_data_maker_output/label/'
"""
------------------------------------参数设置-----------------------------------------
"""

path = os.listdir(root_dir + str(block))

# np.set_printoptions(threshold='nan')

"""
用于数据压缩的np矩阵exp2
"""
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)


def compress_manual(manual):
    assert isinstance(manual, np.ndarray)
    manual = manual.reshape((16, 361))
    return np.dot(exp2, manual)


"""
shape = (n, 2)
第一个数是预测点，第二个数是胜负
"""

for file in path:
    fp = open(root_dir + str(block) + '/' + file)
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
        # 如果出现了tt(停一手)，那么终止生成棋谱
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
    np.save(dataset_dir + file, compressed)
    np.save(label_dir + file, labelset)
    print file, compressed.shape, labelset.shape
    fp.close()

print("PikachuP: Processed. ")
