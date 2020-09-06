# -*- coding: utf-8 -*-
import numpy as np
import board
import go_plot
import os
import util
import sys
import random

"""
            STEP6           shuffle   第1次       将shuffle_batch(1000)个npy文件为1组进行合并，然后打乱
                            然后以10个为一组再打乱一次
            
            第一次：drop后的文件夹，1000个为1组，输出到shuffle文件夹
            第二次：shuffle文件夹下的内容。10个为1组，输出到shuffle'文件夹下
            
            接下来把这些文件送入   STEP 7 
            
            
            PikachuP
            
            修改了打乱策略，使得打乱后的冗余数据较少 YCK 2019.8.29
            
            
"""


"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-参数设置-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""
"""
经过第4步或第5步操作后，数据集文件的位置
"""
dataset_dir                                                 = 'E:/PikachuGoDataSample/4_policy_drop/dataset/'
"""
经过第4步或第5步操作后，数据集标签的位置
"""
label_dir                                                   = "E:/PikachuGoDataSample/4_policy_drop/label/"

"""
输出的目标目录（数据集）
"""
output_data_dir                                             = 'E:/PikachuGoDataSample/6_policy_shuffle/dataset/'
"""
输出的目标目录（标签集）
"""
output_label_dir                                            = 'E:/PikachuGoDataSample/6_policy_shuffle/label/'

"""
多少个文件打乱一次
"""
shuffle_batch                                               = 30

"""
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
"""




# np.set_printoptions(threshold='nan')
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)


def compress_manual(manual):
    assert isinstance(manual, np.ndarray)
    manual = manual.reshape((16, 361))
    return np.dot(exp2, manual)

path = os.listdir(dataset_dir)


first = True
data = None
label = None
i = 0
for file in path:
    n_data = np.load(dataset_dir + file)
    n_label = np.load(label_dir + file)

    # print file, data.shape, label.shape
    if first is False:
        data = np.vstack((data, n_data))
        label = np.vstack((label, n_label))
    else:
        data = n_data
        label = n_label
        first = False
    # print i
    i += 1
    if i % shuffle_batch == 0:
        permutation = np.random.permutation(data.shape[0])
        shuffled_dataset = data[permutation, :]
        shuffled_labels = label[permutation, :]
        print i, shuffled_dataset.shape, shuffled_labels.shape
        np.save(output_data_dir + str(i), shuffled_dataset)
        np.save(output_label_dir + str(i), shuffled_labels)
        data = None
        label = None
        first = True

if i % shuffle_batch != 0:    
    permutation = np.random.permutation(data.shape[0])
    shuffled_dataset = data[permutation, :]
    shuffled_labels = label[permutation, :]
    print i, shuffled_dataset.shape, shuffled_labels.shape
    np.save(output_data_dir + str(i), shuffled_dataset)
    np.save(output_label_dir + str(i), shuffled_labels)

print("PikachuP: Processed. ")

