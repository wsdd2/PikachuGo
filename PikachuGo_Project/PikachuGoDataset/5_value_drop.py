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
            STEP5           为防止过拟合，删去一定的盘面特征 这里的是价值网络
            
            使用STEP3的相同的文件，制作出两个不同的数据集
            
            一个是策略网络数据集
            一个是价值网络数据集
            
            两个文件夹下（数据集文件  和  标签集文件）的文件名现在是相同的
            
            两边都同时删去一半  PikachuP

"""


"""
------------------------------------参数设置-----------------------------------------
"""
"""
第一次的生成的npy文件的所在位置, 第一个是dataset
"""
npy_dir                                                 = 'E:/PikachuGoDataSample/3_data_maker_output/dataset/'
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
save_dir                                                = 'E:/PikachuGoDataSample/5_value_drop/dataset/'
"""
数据集标签的保存位置
"""
save_label_dir                                          = 'E:/PikachuGoDataSample/5_value_drop/label/'
"""
价值网络：各删去一半
全都设置为0.5
"""
dropout_rate_win                                        = 0.5
dropout_rate_loss                                       = 0.5
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