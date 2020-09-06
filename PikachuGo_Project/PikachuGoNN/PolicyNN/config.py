# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx


"""
            config.py          配置神经网络
            
            PikachuP 2018年9月9日02:45:08
"""


"""numpy输出的行宽
"""
np.core.arrayprint._line_width                      = 120
"""使用gpu作为数据设备
"""
data_device                                         = mx.gpu()
"""使用gpu作为训练设备
"""
train_device                                        = [mx.gpu()]
"""使用最快的卷积算法
"""
auto_turn                                           = 'fastest'
"""模型所存放的目录
"""
model_directory                                     = 'model'
"""模型输出时的前缀
"""
model_prefix                                        = model_directory + '/model'
"""如果是延续上一次训练结果的话，继续上一次的位置
"""
n_epoch_load                                        = 0
"""是否使用棋盘对称来增强数据
"""
apply_symmetry                                      = True
"""是否将数据打散
"""
shuffle_data                                        = True
"""模型的储存间隔
"""
save_period                                         = 30
"""喂入神经网络的批大小
"""
batch_size                                          = 5
"""初始学习速率
"""
learning_rate                                       = 0.1
"""weight decay(L2正则化强度)
"""
wd                                                  = 0
"""使用sgd时的动量参数
"""
momentum                                            = 0.9
"""每次调整学习率的衰减度
"""
learning_decay                                      = 0.95
"""学习率在衰减到多少之后退出
"""
exit_learning_rate                                  = 0.0005
"""训练集的位置
"""
train_prefix                                        = 'E:/PikachuGoDataSample/8_ultimate_policy/dataset/'
"""训练集标签的位置
"""
label_prefix                                        = 'E:/PikachuGoDataSample/8_ultimate_policy/label/'
"""训练集块号的起始值
"""
train_begin_index                                   = 0
"""训练集块号的结束值
"""
train_end_index                                     = 5
