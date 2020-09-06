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
                            STEP7  Final  激动人心的最终数据集生成 PikachuP
                            之前已经乱过数据，现在要做的是以10000个盘面为单位，重新生成新的数据集

                            之前的乱过的文件是大文件。
                            有些不能被10000整除，所以它是多余出来的，于是我就把它当做验证集使用了，前缀加上了val

                            有val开头的均为验证集        可以选取一个最大的使用

"""

"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-参数设置-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""
"""
输入的大文件 数据集
"""
dataset_dir                                                     = 'E:/PikachuGoDataSample/6_policy_shuffle/dataset/'
"""
输入的大文件 标签集
"""
label_dir                                                       = 'E:/PikachuGoDataSample/6_policy_shuffle/label/'
"""
输出的最终数据集文件的输出目录
"""
output_dataset_dir                                              = 'E:/PikachuGoDataSample/8_ultimate_policy/dataset/'
"""
输出的最终数据集文件的输出目录（标签集）
"""
output_label_dir                                                = 'E:/PikachuGoDataSample/8_ultimate_policy/label/'
"""
每N条数据为一个文件
"""
N                                                               = 1000
"""
一个大文件里面最多可以有多少个N个训练样本
"""
max_train                                                       = 3000
"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-参数设置-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""


# np.set_printoptions(threshold='nan')
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)


def compress_manual(manual):
    assert isinstance(manual, np.ndarray)
    manual = manual.reshape((16, 361))
    return np.dot(exp2, manual)


path = os.listdir(dataset_dir)

data = None
label = None

no = 0
for file in path:
    data = np.load(dataset_dir + file)
    label = np.load(label_dir + file)
    for i in range(max_train):
        if i*N+N > data.shape[0]:
            np.save(output_dataset_dir + 'val' + str(no), data[i * N: min(i * N + N, data.shape[0])])
            np.save(output_label_dir + 'val' + str(no), label[i * N: min(i * N + N, label.shape[0])])
            data = None
            label = None
            break
        np.save(output_dataset_dir + str(no), data[i*N: min(i*N+N, data.shape[0])])
        np.save(output_label_dir + str(no), label[i*N: min(i*N+N, label.shape[0])])
        no += 1

print("PikachuP: Processed. ")