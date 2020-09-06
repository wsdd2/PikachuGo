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
import symmetry
import copy
import board
import util
import math
import go_plot
# 如果需要打印numpy数组，取消下面这一行的注释
np.set_printoptions(threshold='nan')
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=1000)


current_learning_rate = config.learning_rate

"""

self-play中观察的前K个值

"""

K = 10

"""
                MXNet框架           用于训练深度卷积网络
                该文件位于ReinforcementNN下，通过策略网络训练强化学习网络，属于强化学习

                使用的算法为策略梯度

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
n_filter = 192

"""
残差块的数目
"""
num_blocks = 8
"""
输入的通道数
"""
input_filters = 16
"""
压缩矩阵的定义方式
"""
exp2 = np.array([[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]], dtype=np.uint16)
"""
---------------------------------------残差网络的参数设置------------------------------------------------
"""

"""
Tensorflow代码

        # 计算 loss（和平时说的 loss 不一样）有一个负号
        # 因为 TensorFlow 自带的梯度下降只能 minimize（最小化）loss
        # 而 Policy Gradient 里面是要让这个所谓的 loss 最大化
        # 因此需要反一下。对负的去让它最小化，就是让它正向最大化
        self.loss = -tf.reduce_mean(tf.log(self.outputs) * self.reward_holder)

在这里翻译为MXNet框架代码如下

这里的label不仅仅指label，而是经过Softmax层的NN输出

pred指的是reward，也就是0和1的数组，赢的一方的那步棋为1，输的一方的那步棋为-1，其余的
                                    点都是0.

"""


def reinforce_loss(labels, preds):
    # print "reinforce!@@@@@@@@@@@@@@@@@@@@@@@@@@@PPP"


    arr = np.zeros((1, 361))
    arr[0][labels.astype(int)] = 1
    #print "ARR, Pika", arr
    #print "PREDS, Pike", preds
    # print - np.log(np.sum(preds * arr))
    return 0
    return - np.log(np.sum(preds * arr))


"""
假如输出为[0.6, 0.4], 奖励值为[0, 1]  loss = -0.2 -> 要做的是增大奖励值为1的那个输出概率
向负的方减小，其实就是增大概率，假设loss = -0.3
那么他的输出就会是[不关心, 0.6]， loss = -0.3 , 达到了增大那个输出概率的效果


"""

"""
自定义度量指标，为reinforce_loss
"""

metric1 = mx.metric.create(reinforce_loss)
# metric1 = mx.metric.create('ce')
my_metric = [metric1]
epoch_accuracy = 0
epoch_loss = 0
epoch_loss_last = -1
time_last = None


def my_print(str, *args):
    sys.stdout.write(str % args)
    sys.stdout.flush()

# 定义网络，这里先用较小的网络
def main(argv):
    net = mx.sym.Variable('data')
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
    net = mx.sym.SoftmaxOutput(net, name='softmax')


    shape = {"data": (32, input_filters, 19, 19)}
    mx.viz.print_summary(symbol=net, shape=shape)

    if not os.path.exists(config.model_directory):
        os.mkdir(config.model_directory)
    # 建立日志文件

    log_file = os.open(config.model_directory + '/_train_.csv', os.O_RDWR|os.O_CREAT|os.O_APPEND)
    # 新建模型，或载入之前的模型
    module = None
    if config.n_epoch_load == 0:
        module = mx.mod.Module(symbol=net, context=config.train_device)
        arg_params = None
        aux_params = None
    else:
        sym, arg_params, aux_params = mx.model.load_checkpoint(config.model_prefix, config.n_epoch_load)
        module = mx.mod.Module(symbol=sym, context=config.train_device)

    """
    go = board.Go()
    feature = go.generate()

    iter = mx.io.NDArrayIter(data=feature)
    module.bind(for_training=True, data_shapes=iter.provide_data)
    module.set_params(arg_params, aux_params)

    """

    # 建立迭代器
    data_iter = MyDataIter(config.batch_size, mod = module)

    # 将棋谱分为多个文件，这里定义每完成一个文件为1个虚拟的epoch
    def epoch_callback(epoch, symbol, arg_params, aux_params):
        global time_last, epoch_accuracy, epoch_loss, epoch_loss_last, current_learning_rate

        data_iter.change(sym, arg_params, aux_params)

        # 输出真正的epoch数
        my_print('Epoch: %.2f', epoch)
        # 输出性能指标

        my_print('PolicyG, OK, Pika')

        # 输出当前学习速率
        my_print(' lr: %.5f' % (current_learning_rate))

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
            print ("saved : OK")
        if data_iter.status == 1:
            current_learning_rate = - config.learning_rate
        elif data_iter.status == 0:
            current_learning_rate = config.learning_rate


    def batch_callback(epoch):
        pass
        return
        data_iter.can_load_file.set()   # 训练开始，发送信号表示CPU可以接着加载后面的文件

    # 开始训练

    # for t in range(10):


    module.fit(
        data_iter,
        eval_data=None,
        eval_metric=my_metric,
        # initializer=mx.initializer.MSRAPrelu(factor_type='avg', slope=0.0),
        optimizer='sgd',
        optimizer_params={'learning_rate': current_learning_rate},
        # optimizer_params={'learning_rate': config.learning_rate},

        num_epoch=9999999,
        batch_end_callback=batch_callback,
        epoch_end_callback=epoch_callback,
        arg_params=arg_params,
        aux_params=aux_params,
        begin_epoch=config.n_epoch_load+1,  # 延续之前的训练进度

        # force_rebind = True

    )


"""

手动定义一个softmax函数

这里我想 取 top k 个可选点，再用一次softmax, 来随机选取下一个点位

"""
def softmax(lst):
    total = 0
    for x in lst:
        total += math.exp(x)
    return [math.exp(x) / total for x in lst]


def random_pick(lst):
    # print softmax(lst)
    index = np.random.choice([i for i in range(len(lst))], p = softmax(lst))
    return index
# l = [2.4, 2.2, 2.1, 0.1, 0.1]




class MyDataIter(DataIter):

    def change(self, sym, arg_params, aux_params):
        self.module = mx.mod.Module(symbol=sym)
        go = board.Go()
        feature = go.generate()
        iter = mx.io.NDArrayIter(data=feature)
        self.module.bind(for_training=False, data_shapes=iter.provide_data)
        self.module.set_params(arg_params, aux_params)

    def self_play(self):

        label_black = []
        label_white = []

        go = board.Go()
        t0 = time.clock()

        black = None
        white = None
        self.BLACK = None
        self.WHITE = None

        TOTAL = 350
        NUM = 300

        for i in range(TOTAL):
            # 生成特征
            my_print("." if i % 10 == 0 else "")
            feature = go.generate()
            # 把特征喂入神经网络，获得预测
            iter = mx.io.NDArrayIter(data=feature)
            pred = self.module.predict(iter).asnumpy()
            # 将不可入点扔掉
            pred = feature[0][2].reshape(1, 361) * pred * 10
            # 排个序
            out = np.argsort(-pred)
            predsort = - np.sort(-pred)
            # print predsort[0: K]
            # 看看前K个的值
            process = out[0][0: K]
            idx = random_pick(predsort[0][0: K].reshape(-1))

            if i == 0:
                black = feature.reshape(-1, 16, 19, 19)
                label_black.append(out[0][idx])
                go.place_stone_num(out[0][idx])
            elif i == 1:
                white = feature.reshape(-1, 16, 19, 19)
                label_white.append(out[0][idx])
                go.place_stone_num(out[0][idx])
            elif i < NUM:
                f = feature.reshape(-1, 16, 19, 19)
                if i % 2 == 0:
                    black = np.vstack((black, f))
                    label_black.append(out[0][idx])
                else:
                    white = np.vstack((white, f))
                    label_white.append(out[0][idx])
                go.place_stone_num(out[0][idx])
            else:
                go.place_stone_num(out[0][0])

                #f = go.generate().reshape(-1, 16, 19, 19)
                #if i % 2 == 0:
                #    black = np.vstack((black, f))
                #    label_black.append(out[0][0])
                #else:
                #    white = np.vstack((white, f))
                #    label_white.append(out[0][0])


        # print len(label_black)

        print label_black
        print label_white


        self.BLACK_LABEL = np.zeros((150, 361), dtype=np.int16)
        self.WHITE_LABEL = np.zeros((150, 361), dtype=np.int16)
        for i in range(NUM):
            if i % 2 == 0:# 偶数，黑棋盘面
                # print "BLACK", i
                self.BLACK_LABEL[i // 2][label_black[i // 2]] = 1
            else:
                # print "WHITE", i
                self.WHITE_LABEL[(i-1)//2][label_white[(i-1)//2]] = 1

        permutation = np.random.permutation(black.shape[0])
        black = black[permutation, :]
        white = white[permutation, :]



        self.BLACK_LABEL = self.BLACK_LABEL.reshape(-1, 19, 19)
        self.WHITE_LABEL = self.WHITE_LABEL.reshape(-1, 19, 19)
        self.BLACK_LABEL = self.BLACK_LABEL.reshape(-1, 361)
        self.WHITE_LABEL = self.WHITE_LABEL.reshape(-1, 361)
        self.BLACK_LABEL = self.BLACK_LABEL[permutation, :]
        self.WHITE_LABEL = self.WHITE_LABEL[permutation, :]
        self.BLACK_LABEL = np.argmax(self.BLACK_LABEL, axis=1)
        self.WHITE_LABEL = np.argmax(self.WHITE_LABEL, axis=1)



        print self.BLACK_LABEL
        print self.WHITE_LABEL
        self.module = None
        gc.collect()



        """
        # 判断胜负, 黑胜返回1，白胜利返回0
        def evaluate(self):

        """
        # go_plot.go_plot(terminal // 50)
        result = go.evaluate()

        if result == 1:
            print("B+")
        else:
            print("W+")

        print "generate_used", time.clock() - t0

        self.BLACK = black
        self.WHITE = white

        #print self.BLACK
        #print self.WHITE
        """
        0代表黑胜利，1代表白胜利
        """
        if result == 1:
            self.winner = 1
        else:
            self.winner = 0
        go = None
        label_black = None
        label_white = None

        return result

    def load_file(self):

        filename_train = ''
        filename_label = ''
        # 获取需要载入的文件名
        """
        if self.is_train:
            index = self.train_list[self.train_index]
            my_print('[pl %s' % str(index).rjust(4))
            filename_train = config.train_prefix + str(index) + '.npy'
            filename_label = config.label_prefix + str(index) + '.npy'
        else:
            filename_train = config.train_prefix + 'val.npy'
            filename_label = config.label_prefix + 'val.npy'
            my_print('[pl validate data')

        # 载入数据文件
        #training_data = np.load(filename_train)
        #training_data = training_data.reshape(-1, 1, 361)
        #exp = exp2.T
        #exp = exp.reshape(1, input_filters, 1)
        # training_data = ((np.bitwise_and(training_data, exp) > 0) + 0).reshape(-1, input_filters, 19, 19)

        training_data = self.self_play()
        training_data = training_data.reshape(-1, 16, 19, 19)
        training_data = np.vstack((training_data, ) * 10)    # 坑


        # label = np.load(filename_label)
        NUM = training_data.shape[0]
        NUM = 10
        label_data_ = np.zeros((NUM, 361), dtype=np.int16)
        for i in range(NUM):
            label_data_[i][i] = 1
        label_data_ = label_data_.reshape(-1, 19, 19)

        label_data = label_data_.reshape(-1, 361)
        label_data = np.argmax(label_data, axis=1)
        #print("TRAIN   DATA", training_data.shape)
        #print("oookkvbgyuivgyigbvyg7uyfgfgy")
        #print(type(label_data))
        label = None
        label_data_ = None
        #print("LABEL  !!", label_data.shape)
        # 表示加载完成
        my_print(']')



        obj = [training_data, label_data]
        self.data_list = [mx.ndarray.array(obj[0], config.data_device),
                          mx.ndarray.array(obj[1], config.data_device)]
        self.data = [('data', self.data_list[0])]
        self.label = [('softmax_label', self.data_list[1])]



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
            pass
            # self.can_load_file.wait()
            # self.can_load_file.clear()
        """

    # 负责加载数据
    def init_data(self):


        """        if self.is_train:
            tmp = self.queue.get(block=True, timeout=None)  # 从队列加载数据
            self.data_list = [mx.ndarray.array(tmp[0], config.data_device),
                              mx.ndarray.array(tmp[1], config.data_device)]

        """
        # 按MXNet所要求的规范设置

        self.data = None
        self.label = None


        if self.first_run:
            self.first_run = False
            go = board.Go()
            training_data = go.generate().reshape(-1, 16, 19, 19)
            training_data = np.vstack((training_data, ) * 256)
            label_data = np.zeros((256, 361))
            label_data[-1][72] = 1
            label_data = np.argmax(label_data, axis=1)
            obj = [training_data[0: 256], label_data[0: 256]]

            self.data_list = [mx.ndarray.array(obj[0], config.data_device),
                              mx.ndarray.array(obj[1], config.data_device)]

            self.data = [('data', self.data_list[0])]
            self.label = [('softmax_label', self.data_list[1])]
            self.num_data = 1
            return

        if self.status == 0:            # 如果是0,说明新的棋谱需要生成
            res = self.self_play()      # 生成新的棋谱
            print self.BLACK.shape, self.BLACK_LABEL.shape

            my_print(']')

            if self.winner == 1:        # 黑棋胜利
                training_data = self.BLACK
                label_data = self.BLACK_LABEL
            else:
                training_data = self.WHITE
                label_data = self.WHITE_LABEL

            obj = [training_data[0: 128], label_data[0: 128]]

            self.data_list = [mx.ndarray.array(obj[0], config.data_device),
                              mx.ndarray.array(obj[1], config.data_device)]

            self.data = [('data', self.data_list[0])]
            self.label = [('softmax_label', self.data_list[1])]

            #print(self.data_list[1])

            #print self.data_list[0].shape, self.data_list[1].shape

            # 设置数据个数
            self.num_data = self.data_list[0].shape[0]
            #print self.data_list[0].shape
            #print("num_data", self.num_data)
            # assert self.num_data >= self.batch_size, "batch_size need to be smaller than data size."
            self.status = 1
            return




        elif self.status == 1:
            my_print(']')
            if self.winner == 0:        # 黑棋输掉
                training_data = self.BLACK
                label_data = self.BLACK_LABEL
            else:
                training_data = self.WHITE
                label_data = self.WHITE_LABEL
            obj = [training_data[0: 128], label_data[0: 128]]

            self.data_list = [mx.ndarray.array(obj[0], config.data_device),
                              mx.ndarray.array(obj[1], config.data_device)]

            self.data = [('data', self.data_list[0])]
            self.label = [('softmax_label', self.data_list[1])]

            #print(self.data_list[1])

            #print self.data_list[0].shape, self.data_list[1].shape

            # 设置数据个数
            self.num_data = self.data_list[0].shape[0]
            #print self.data_list[0].shape
            #print("num_data", self.num_data)
            # assert self.num_data >= self.batch_size, "batch_size need to be smaller than data size."
            self.status = 0
            return



            # 继续上一次的棋谱训练，训练输掉的棋谱




        # training_data = training_data.reshape(-1, 16, 19, 19)
        # training_data = np.vstack((training_data, ) * 10)    # 坑

        # label = np.load(filename_label)
        # NUM = training_data.shape[0]
        #NUM = 150
        #label_data_ = np.zeros((NUM, 361))
        #for i in range(NUM):
        #    label_data_[i][0] = 1
        #label_data_ = label_data_.reshape(-1, 19, 19)

        #label_data = label_data_.reshape(-1, 361)
        #label_data = np.argmax(label_data, axis=1)
        #print("oookkvbgyuivgyigbvyg7uyfgfgy")
        #print(type(label_data))
        #label = None
        #label_data_ = None





        # 表示加载完成

    # 负责初始化迭代器
    def __init__(self, batch_size=1, mod=None):
        super(MyDataIter, self).__init__()
        self.first_run = True
        # self.can_load_file = multiprocessing.Event()
        self.module = mod
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.BLACK = None
        self.WHITE = None
        self.BLACK_LABEL = None
        self.WHITE_LABEL = None

        self.winner = None
        self.loser = None
        """
        第一次，从状态0到1，表示赢的训练完了
        
        
        """
        self.status = 0

        # 打散加载文件的列表
        self.train_index = 0
        self.train_list = range(config.train_begin_index, config.train_end_index + 1)
        random.shuffle(self.train_list)

        self.is_train = True
        if self.is_train:  # 如果是训练数据，则开启队列和加载数据的线程
            if __name__ == '__main__':
                self.queue = multiprocessing.Queue(maxsize=1)
                self.worker = multiprocessing.Process(target=self.load_file)
                self.worker.daemon = True
                # self.worker.start()
                self.init_data()
                self.init_misc()
        else:  # 如果是测试数据，则直接加载数据
            self.load_file()
            self.init_data()
            self.init_misc()

    # 下面是一些细节函数，基本来自于MXNet源代码中迭代器的定义的复制粘贴，毋需特别关注
    def init_misc(self):
        # self.num_source = len(self.data_list)
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
        # print("ok")
        # self.load_file()

        self.cursor += self.batch_size
        # self.cursor = 0

        if self.cursor < self.num_data:
            #print self.getdata()
            #print self.getlabel()
            #while raw_input() != "c":
             #   pass
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
        #print("*************************************************")
        #print(self._getdata(self.data))
        #print("*************************************************")
        return self._getdata(self.data)

    def getlabel(self):
        # print(self.label)
        #print("*************************************************")
        #print("SELF.LAAAAABBELLLLLLLLLLLL", self.label)
        #print("*************************************************")
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