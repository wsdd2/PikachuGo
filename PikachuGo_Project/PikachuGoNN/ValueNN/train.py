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
import copy
import logging
import symmetry

# 如果需要打印numpy数组，取消下面这一行的注释
# np.set_printoptions(threshold='nan')
my_metric = [mx.metric.create('mse')]
epoch_accuracy = 0
epoch_loss = 0
epoch_loss_last = -1
time_last = None


logging.getLogger().setLevel(logging.DEBUG)

"""
                MXNet框架           用于训练深度卷积网络
                该文件位于ValueNN下，用于训练策略网络，属于有监督学习
                网络结构为
                                1. 3*3卷积
                                2. 8个残差块,BACBAC结构，192通道
                                3. 收尾，BAC
                                4. 摊平，Softmax

"""
"""
---------------------------------------残差网络的参数设置------------------------------------------------
"""
"""
每个残差块的通道数
"""
n_filter                                                    = 192

"""
残差块的数目
"""
num_blocks                                                  = 6
"""
输入的通道数
"""
input_filters                                               = 16
"""
压缩矩阵的定义方式
"""
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)
"""
---------------------------------------残差网络的参数设置------------------------------------------------
"""



"""
print 'ok'
print 'start'
_train = np.load('../TrainData/train_1.npy')
_label = np.load('../LabelData/label_1.npy')
print _train[2][2]
ori = copy.deepcopy(_train)
apply_random_symmetry(_train, _label)
print _train[2][2]
cnt = 0
for i in range(1000):
    if (_train[i][1] == ori[i][1]).all():
        cnt += 1

print 'cnt', cnt
"""

def my_print(str, *args):
    sys.stdout.write(str % args)
    sys.stdout.flush()

# 定义网络，这里先用较小的网络
def main(argv):
    net = mx.sym.Variable('data')
    Y = mx.symbol.Variable('tanh_label')
    # 预处理
    net = mx.sym.Convolution(net, name='ConvPRE', kernel=(3, 3), pad=(1, 1), num_filter=n_filter)
    # 残差结构
    for i in range(num_blocks):
        identity = net
        net = mx.sym.BatchNorm(net, name='BN_A_'+str(i), fix_gamma=False)
        net = mx.sym.Activation(net, name='ACT_A_'+str(i), act_type='relu')
        net = mx.sym.Convolution(net, name='CONV_A_'+str(i), kernel=(3, 3), pad=(1, 1), num_filter=n_filter)

        net = mx.sym.BatchNorm(net, name='BN_B_' + str(i), fix_gamma=False)
        net = mx.sym.Activation(net, name='ACT_B_' + str(i), act_type='relu')
        net = mx.sym.Convolution(net, name='CONV_B_' + str(i), kernel=(3, 3), pad=(1, 1), num_filter=n_filter)
        net = net + identity

    # 收尾
    net = mx.sym.BatchNorm(net, name='FinalBN', fix_gamma=False)
    net = mx.sym.Activation(net, name='FinalACT', act_type='relu')
    # 合并为1个通道
    net = mx.sym.Convolution(net, name='FinalConv', kernel=(1, 1), num_filter=1)
    net = mx.sym.Flatten(net)
    net = mx.sym.FullyConnected(net, num_hidden=1, name="FC")
    net = mx.sym.Activation(net, act_type='tanh', name='tanh_layer')
    net = mx.sym.LinearRegressionOutput(net, label=Y)

    shape = {"data": (32, input_filters, 19, 19)}
    mx.viz.print_summary(symbol=net, shape=shape)

    if not os.path.exists(config.model_directory):
        os.mkdir(config.model_directory)
    # 建立日志文件

    log_file = os.open(config.model_directory + '/_train_.csv', os.O_RDWR|os.O_CREAT|os.O_APPEND)
    # 新建模型，或载入之前的模型
    if config.n_epoch_load == 0:
        module = mx.mod.Module(symbol=net, data_names=['data'], label_names=['tanh_label'], context=config.train_device)
        arg_params = None
        aux_params = None
    else:
        sym, arg_params, aux_params = mx.model.load_checkpoint(config.model_prefix, config.n_epoch_load)
        module = mx.mod.Module(symbol=sym,data_names=['data'], label_names=['tanh_label'], context=config.train_device)

    # 建立迭代器
    data_iter = MyDataIter(config.batch_size, True)
    val_iter = MyDataIter(config.batch_size, False)

    # 将棋谱分为多个文件，这里定义每完成一个文件为1个虚拟的epoch
    def epoch_callback(epoch, symbol, arg_params, aux_params):
        global time_last, epoch_accuracy, epoch_loss, epoch_loss_last

        # 输出真正的epoch数
        real_epoch = float(epoch) / (config.train_end_index - config.train_begin_index)
        my_print(' %.2f', (real_epoch))
        # 输出性能指标
        # batch_accuracy = my_metric[0].get_name_value()[0][1]
        #metric = mx.metric.MSE()
        #mse = module.score(, metric)
        #cross_loss = mse[0][1]
        # my_print(' : 1/2/5/10 %.2f%%', (100.0 * batch_accuracy))
        # my_print('-%.2f%%', (100.0 * my_metric[2].get_name_value()[0][1]))
        # my_print('-%.2f%%', (100.0 * my_metric[3].get_name_value()[0][1]))
        # my_print('-%.2f%%', (100.0 * my_metric[4].get_name_value()[0][1]))
        # my_print('-mse: %.6f -- ', (my_metric[0].get_name_value()[0][1]))
        # 输出当前学习速率
        my_print('lr: %.6f' % (config.learning_rate))


        time_now = time.time()
        if time_last is None:
            time_last = time_now
            my_print(': n/a\n')
        else:
            my_print(' : %.2fs\n' % (time_now - time_last))
        time_last = time_now

        if epoch % config.save_period == 0:
            # 保存模型
            module.save_checkpoint(config.model_prefix, epoch, save_optimizer_states=True)
            # 测试在性能集上的指标
            val_metric = module.score(val_iter, [mx.metric.MSE()])

            val_loss = val_metric[0][1]
            # 输出性能指标
            #epoch_loss = float(epoch_loss) / config.save_period
            print ("saved  : mse %.6f " % (val_loss))
            # 保存到日志中
            #os.write(log_file, str(real_epoch) + ',' + str(epoch_loss) + ',' + str(config.learning_rate) + '\n')
            #os.fsync(log_file)

            if epoch_loss > epoch_loss_last and epoch_loss_last != -1:
                config.learning_rate = config.learning_rate * config.learning_decay
            epoch_loss_last = epoch_loss

            epoch_loss = 0.0
            # 学习速率低时终止

            if config.learning_rate < config.exit_learning_rate:
                exit(0)

    def batch_callback(epoch):
        data_iter.can_load_file.set()   # 训练开始，发送信号表示CPU可以接着加载后面的文件

    # 开始训练

    module.fit(
        data_iter,
        eval_data=None,
        eval_metric=mx.metric.create('mse'),
        initializer=mx.initializer.MSRAPrelu(factor_type='avg', slope=0.0),
        optimizer='adam',
        # optimizer_params={'learning_rate': config.learning_rate, 'wd':config.wd, 'momentum':0.9},
        optimizer_params={'learning_rate': config.learning_rate},

        num_epoch=9999999,
        batch_end_callback=batch_callback,
        epoch_end_callback=epoch_callback,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=config.n_epoch_load+1, # 延续之前的训练进度

    )




class MyDataIter(DataIter):
    def load_file(self):
        while True:
            filename_train = ''
            filename_label = ''
            # 获取需要载入的文件名
            if self.is_train:
                index = self.train_list[self.train_index]
                my_print ('[pl %s' % str(index).rjust(4))
                filename_train = config.train_prefix + str(index) + '.npy'
                filename_label = config.label_prefix + str(index) + '.npy'
            else:
                filename_train = config.train_prefix + 'val.npy'
                filename_label = config.label_prefix + 'val.npy'
                my_print('[pl validate data')

            # 载入数据文件
            training_data = np.load(filename_train)
            training_data = training_data.reshape(-1, 1, 361)
            exp = exp2.T
            exp = exp.reshape(1, input_filters, 1)
            training_data = ((np.bitwise_and(training_data, exp) > 0) + 0).reshape(-1, input_filters, 19, 19)
            label = np.load(filename_label)

            label_data = label[ : ,1]

            if self.is_train and config.apply_symmetry:
                symmetry.apply_random_symmetry_without_label(training_data)


            # 表示加载完成
            my_print(']')

            if self.is_train:
                self.queue.put(obj=[training_data, label_data], block=True, timeout=None)
                self.train_index = self.train_index + 1
                # 如果已经完成全部文件的训练，那么就重新打散文件的顺序。
                if self.train_index >= len(self.train_list):
                    self.train_index = 0
                    random.shuffle(self.train_list)
            else:# 如果是测试数据，那么一次载入
                self.data_list = [mx.ndarray.array(training_data, config.data_device), \
                                      mx.ndarray.array(label_data, config.data_device)]
            gc.collect()        # 要求垃圾回收

            if not self.is_train:
                return
                # 停下来等待信号
            if self.is_train:
                self.can_load_file.wait()
                self.can_load_file.clear()

    # 负责加载数据
    def init_data(self):
        if self.is_train:
            tmp = self.queue.get(block=True, timeout=None)  # 从队列加载数据
            self.data_list = [mx.ndarray.array(tmp[0], config.data_device),
                              mx.ndarray.array(tmp[1], config.data_device)]

        # 按MXNet所要求的规范设置
        self.data = [('data', self.data_list[0])]
        self.label = [('tanh_label', self.data_list[1])]
        # 设置数据个数
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= self.batch_size, "batch_size need to be smaller than data size."

    # 负责初始化迭代器
    def __init__(self, batch_size=1, is_train=True):
        super(MyDataIter, self).__init__()

        self.can_load_file = multiprocessing.Event()

        self.cursor = -batch_size
        self.batch_size = batch_size

        # 打散加载文件的列表
        self.train_index = 0
        self.train_list = range(config.train_begin_index, config.train_end_index + 1)
        random.shuffle(self.train_list)

        self.is_train = is_train
        if self.is_train:  # 如果是训练数据，则开启队列和加载数据的线程
            if __name__ == '__main__':
                self.queue = multiprocessing.Queue(maxsize=1)
                self.worker = multiprocessing.Process(target=self.load_file)
                self.worker.daemon = True
                self.worker.start()
                self.init_data()
                self.init_misc()
        else:  # 如果是测试数据，则直接加载数据
            self.load_file()
            self.init_data()
            self.init_misc()

    # 下面是一些细节函数，基本来自于MXNet源代码中迭代器的定义的复制粘贴，毋需特别关注
    def init_misc(self):
        self.num_source = len(self.data_list)
        self.provide_data = [DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
                             for k, v in self.data]
        self.provide_label = [DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
                              for k, v in self.label]

    def hard_reset(self):
        self.cursor = -self.batch_size

    def reset(self):
        if self.is_train:
            self.init_data()
        self.cursor = -self.batch_size

    def next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return OneBatch(data=self.getdata(), label=self.getlabel(), pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        if self.cursor + self.batch_size <= self.num_data:  # no pad
            return [x[1][self.cursor:self.cursor + self.batch_size] for x in data_source]
        else:  # with pad
            pad = self.batch_size - self.num_data + self.cursor
            return [concatenate([x[1][self.cursor:], x[1][:pad]]) for x in data_source]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0






class OneBatch(object):
    def __init__(self, data, label, pad=None, index=None, bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

# 运行主函数
if __name__ == '__main__':
    main(sys.argv)