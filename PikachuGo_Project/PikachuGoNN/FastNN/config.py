#-*-coding:utf-8-*-
import numpy as np
import mxnet as mx
np.core.arrayprint._line_width = 120    # numpy输出的行宽

data_device = mx.gpu()                  # 使用gpu
train_device = [mx.gpu()]
auto_turn = 'fastest'                   # 使用四最快的卷积算法

model_directory = 'model'
model_prefix = model_directory + '/model'

n_epoch_load = 0                       # 从哪里开始训练

apply_symmetry = True                   # 使用棋盘对称
shuffle_data = True                     # 是否打散数据

save_period = 30                        # 存储模型的间隔

batch_size = 32                        # 批大小
learning_rate = 0.1
wd = 0                                  # L2正则化强度

learning_decay = 0.93                    # 每次调整学习率的衰减度
exit_learning_rate = 0.005               # 在衰减到多少之后退出

train_prefix = 'E:/PikachuGoDataSample/8_ultimate_policy/dataset/'
label_prefix = 'E:/PikachuGoDataSample/8_ultimate_policy/label/'

train_begin_index = 0
train_end_index = 87
