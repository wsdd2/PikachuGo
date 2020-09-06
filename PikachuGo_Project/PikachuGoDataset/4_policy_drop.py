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
            STEP4           为防止过拟合，删去一定的盘面特征
            
            两个文件夹下（数据集文件  和  标签集文件）的文件名现在是相同的
            
            删除的盘面特征将会以一定的策略随机删去，更多保留了赢棋一方的策略  PikachuP

"""


"""
------------------------------------参数设置-----------------------------------------
"""
"""
第一次的生成的npy文件的所在位置, 第一个是dataset
"""
npy_dir                                                 = "E:/PikachuGoDataSample/3_data_maker_output/dataset/"
"""
第二次生成的npy文件的所在位置，第二个是labelset
"""
npy_label_dir                                           = "E:/PikachuGoDataSample/3_data_maker_output/label/"


"""
                筛选掉部分用于策略网络训练的之后的数据集的保存位置
"""
"""
数据集的保存位置
"""
save_dir                                                = 'E:/PikachuGoDataSample/4_policy_drop/dataset/'
"""
数据集标签的保存位置
"""
save_label_dir                                          = 'E:/PikachuGoDataSample/4_policy_drop/label/'
"""
在赢棋的情况下，被选中的概率阈值和输棋的情况下，被选中的阈值
如果以选取40%的棋谱为目标
并且其中：3/4的盘面为赢棋方的走子策略
          1/4的盘面为输棋方的走子策略       设置阈值1为 0.6 ，阈值2为 0.2 
"""
dropout_rate_win                                        = 0.6
dropout_rate_loss                                       = 0.2
"""
------------------------------------参数设置-----------------------------------------
"""


path = os.listdir(npy_dir)

for file in path:
    data = np.load(npy_dir + file)
    label = np.load(npy_label_dir + file)

    # print file, data.shape, label.shape

    droplst = []
    rd = np.random.rand(label.shape[0])
    for i in range(label.shape[0]):
        if label[i][1] == +1:
            if rd[i] > dropout_rate_win:
                droplst.append(i)
        elif label[i][1] == -1:
            if rd[i] > dropout_rate_loss:
                droplst.append(i)

    data = np.delete(data, droplst, axis=0)
    label = np.delete(label, droplst, axis=0)

    np.save(save_dir + file, data)
    np.save(save_label_dir + file, label)

print("PikachuP: Processed. ")