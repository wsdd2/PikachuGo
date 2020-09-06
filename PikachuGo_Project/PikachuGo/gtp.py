# coding=UTF-8

"""
计算机围棋的棋盘表示，PikachuP
2018年8月23日
"""
import numpy as np
import re
import os
import copy
import sys
import mxnet as mx
import board
import util
import zobrist
import time
import math
import random
import go_plot
import config


ROLLOUTS = config.search_times_ucb
DEPTH = config.search_depth_ucb
UCB_C = config.para_c_ucb
POLICY_WEIGHT = config.policy_weight

"""
当前全局盘面go，在函数中访问，请用global进行声明。
"""

go = board.Go()


"""
如果需要进行调试，取消下面两行的注释。
"""
# np.set_printoptions(threshold='nan')
# np.set_printoptions(precision=2, suppress=True)


"""
debug_print用来在console中进行人机交互
"""
def debug_print(args):
    print >> sys.stderr, args
    sys.stderr.flush()
    return


"""
为3个网络指定模型
"""

first_run = True
first_run_2 = True
first_run_3 = True

# 策略网络
path = config.root_dir + config.policy_network_dir
ids = config.policy_network_id
sym, arg_params, aux_params = mx.model.load_checkpoint(path, ids)
module = mx.mod.Module(symbol=sym, context=[mx.gpu()])

# 价值网络
v_path = config.root_dir + config.value_network_dir
v_ids = config.value_network_id
v_sym, v_arg_params, v_aux_params = mx.model.load_checkpoint(v_path, v_ids)
v_module = mx.mod.Module(symbol=v_sym, label_names=['tanh_label'], data_names=['data'], context=[mx.gpu()])

# 快速走子网络
f_path = config.root_dir + config.fast_rollout_dir
f_ids = config.fast_rollout_id
f_sym, f_arg_params, f_aux_params = mx.model.load_checkpoint(f_path, f_ids)
f_module = mx.mod.Module(symbol=f_sym, label_names=['soft_label'], data_names=['data'],context=[mx.gpu()])


"""
使用蒙特卡洛方法，运用fast_rollout policy network，进行预测
"""

def monte_carlo_simulate(go):
    global first_run_3
    feature = go.generate_fast()
    iter = mx.io.NDArrayIter(data=feature)
    if first_run_3:
        f_module.bind(data_shapes=iter.provide_data)
        f_module.set_params(f_arg_params, f_aux_params)
        first_run_3 = False
    pred = f_module.predict(iter).asnumpy()
    """
    快速走子网络
    排下序
    
    """
    out = np.argsort(-pred)
    go.place_stone_num(out[0][0])
    depth = DEPTH - go.round + 1
    if depth > config.serach_ucb_limit:
        depth = config.serach_ucb_limit
    # depth = random.randint(0, DEPTH // 2)
    """
    模拟走到底
    """
    while depth >= 0:
        feature = go.generate_fast()
        iter = mx.io.NDArrayIter(data=feature)
        pred = f_module.predict(iter).asnumpy()
        # rd = np.random.rand(1, 361) * 0.13
        # pred = pred + rd
        out = np.argsort(-pred)
        rd_num = random.random()
        """
        此处表示取最优/取次优的探索几率
        """
        if rd_num < 0.1:
            i = 2
        elif rd_num < 0.35:
            i = 1
        else:
            i = 0
        while go.place_stone_num(out[0][i]) and go.is_eye(out[0][i]) != 3 and i < 5:
            i += 1
        depth -= 1
    res = go.evaluate()
    # res_black, res_white = go.evaluate_2()
    # debug_print(go.board)
    # return res_black, res_white
    return res




"""

make_prediction()

gtp.py中最重要的函数




"""

def make_prediction():
    global go, first_run, first_run_2

    """
    第一步：
            先对盘面生成特征：这是一个16通道的特征。
            使用策略网络（也可能是RL网络）
            
    """
    feature = go.generate()
    """
    
            用iter喂数据给神经网络
            
    """
    iter = mx.io.NDArrayIter(data=feature)

    """
            如果是第一次运行，则绑定模型，并设置参数
    """

    if first_run:
        module.bind(data_shapes=iter.provide_data)
        module.set_params(arg_params, aux_params)
        first_run = False

    """
    t0 = time.clock()
    for i in range(320):
        feature = go.generate()
        iter = mx.io.NDArrayIter(data=feature)
        pred = module.predict(iter).asnumpy()
        pred = feature[0][2].reshape(1, 361) * pred
        out = np.argsort(-pred)
        go.place_stone_num(out[0][0])
    go_plot.go_plot(go.board)
    print time.clock() - t0
    exit(0)
    """
    """
    
            module.predict(iter)  进行预测，结果在pred中。他是一个(1, 361)的numpy数组
    
    """
    pred = module.predict(iter).asnumpy()

    """
                    由于有禁入点的缘故，用feature[0][2]点乘pred。feature[0][2]就是可入点的盘面特征
    """
    pred = feature[0][2].reshape(1, 361) * pred
    """
                    先取反  然后从小到大排   60 40 20...  -60 -40 -20   这样的话就是概率从小到大的顺序。
    """
    out = np.argsort(-pred)

    """
            X_prior    ->      X 的平均回报
    """

    if config.enable_ucb is False:
        return out[0][0]

    X_prior = []

    """
            T_prior    ->      已经探索的次数
            基于策略网络，假设一定的先验知识
            total是访问的总次数
    """
    T_prior = [config.pre_place_num] * config.search_position_num
    total = config.pre_place_num * config.search_position_num

    """
            循环做5次，具体看config.search_position_num的配置
    
    """
    for i in range(config.search_position_num):
        # 打印策略网络的落子概率
        debug_print(util.pos_to_gtppos(util.num_to_pos(out[0][i])) + '\t%.2f%%' % (pred[0][out[0][i]] * 100))
        # 复制一份
        gocopy = copy.deepcopy(go)
        # 用复制的一份盘面模拟走子！
        gocopy.place_stone_num(out[0][i])
        # 生成模拟走子后的盘面
        next_state = gocopy.generate()
        # 使用价值网络预测胜率
        iter_ = mx.io.NDArrayIter(data=next_state)
        if first_run_2:
            v_module.bind(data_shapes=iter_.provide_data)
            v_module.set_params(v_arg_params, v_aux_params)
            first_run_2 = False
        # 结果返回到result中
        result = v_module.predict(iter_).asnumpy()[0][0]
        """
            将    策略网络   与   价值网络   相结合！
            
            先验X_prior是
            
            result -> -1 要输  +1 要赢
            盘面判断的是对面的
            所以 (result + 1) / 2    ->    0->对面要输    1->对面要赢
            反过来：
                                              我要赢         我要输
                                    1 - (result + 1) / 2     0  我输了  1 我赢了
            
            
        """
        X_prior.append(config.value_weight * (1 - ((result + 1) / 2)) + POLICY_WEIGHT * pred[0][out[0][i]])

        """
        判断本方是哪一方，然后输出对手的获胜概率
        """
        if gocopy.current_player() == 0:
            side = 'BLACK: '
        else:
            side = 'WHITE: '
        debug_print(side + '%.2f%%' % ((result + 1) / 0.02))


    for i in range(config.search_position_num):

        """
        先输出探索他的 X拔 
        """
        debug_print(util.pos_to_gtppos(util.num_to_pos(out[0][i])))
        debug_print("-> {:.2f}%".format(X_prior[i] * 100))

    # 计个时
    t = time.time()
    # stat_black, stat_white = 0, 0

    # 判断现在是哪一方走棋，然后设置target值。
    """
    target值用来判断谁输谁赢
    """
    if go.current_player() == 0:
        target = 1
        _side = 'BLACK'
    else:
        target = 0
        _side = 'WHITE'

    """
    TO: 要探索多少轮。
    这个值会逐渐递增
    如果有更好的策略，在这里改进          PikachuP
    """
    TO = ROLLOUTS
    end_search_value = TO / 2 + config.pre_place_num
    for i in range(TO):
        # 选择一个位置：
        ucb_max = -1000
        ucb_id = 0
        """
            选择最值得探索的位置
        """
        for j in range(config.search_position_num):
            ucb = X_prior[j] + UCB_C * math.sqrt(math.log(total) / T_prior[j])
            if ucb > ucb_max:
                ucb_max = ucb
                ucb_id = j

        """
                    在最值得探索的位置!位置模拟落子!
        """
        go_copy = copy.deepcopy(go)
        # pre_black, pre_white = go_copy.evaluate_2()
        go_copy.place_stone_num(out[0][ucb_id])

        """
                用蒙特卡洛模拟模拟走子
                返回的是一个模拟走子的胜负结果，0表示白胜利，1表示黑胜利
        """

        res = monte_carlo_simulate(go_copy)
        prompt = ""

        """
        这里的代码算的是chens形势判断领先多少
        被我废弃了
        if _black - _white > pre_black - pre_white > 0 and target == 1:# 本方黑，黑胜利
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id] + 1) / (T_prior[ucb_id] + 1)
            T_prior[ucb_id] += 1
            prompt = "Win"
            if T_prior[ucb_id] > TO / 2:
                total += 1
                break
        elif _white - _black > pre_white - pre_white and target == 0:    # 本方白，白胜利
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id] + 1) / (T_prior[ucb_id] + 1)
            T_prior[ucb_id] += 1
            prompt = "Win"
            if T_prior[ucb_id] > TO / 2:
                total += 1
                break
        else:
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id]) / (T_prior[ucb_id] + 1)
            T_prior[ucb_id] += 1
            prompt = "Loss"
            if T_prior[ucb_id] > TO / 2:
                total += 1
                break
        total += 1
        """

        if res == 1 and target == 1: # 本方黑，黑胜利
            # 获胜概率增加了！
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id] + 1) / (T_prior[ucb_id] + 1)
            # 被访问次数增加了！
            T_prior[ucb_id] += 1
            # 提示获胜
            prompt = "Win"
            # 提早结束，提高效率
            # 原因：访问节点次数超过二分之一，不可能再有节点比这个更优了
            if T_prior[ucb_id] > end_search_value:
                total += 1
                break
        elif res == 0 and target == 0:    # 本方白，白胜利
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id] + 1) / (T_prior[ucb_id] + 1)
            T_prior[ucb_id] += 1
            prompt = "Win"
            if T_prior[ucb_id] > end_search_value:
                total += 1
                break
        else:
            X_prior[ucb_id] = (X_prior[ucb_id] * T_prior[ucb_id]) / (T_prior[ucb_id] + 1)
            T_prior[ucb_id] += 1
            prompt = "Loss"
            if T_prior[ucb_id] > end_search_value:
                total += 1
                break
        total += 1
        # 提示一下而已
        debug_print("Playout\t" + str(i) + '\t' + util.pos_to_gtppos(util.num_to_pos(out[0][ucb_id])) + "\t" + prompt)

    """
                    max_index       它是我要采取的策略！
    """
    max_index = T_prior.index(max(T_prior))

    debug_print(_side + "：{:.2f}%".format(X_prior[max_index] * 100))

    # UCB完毕，打印一下节点的访问次数
    for i in range(config.search_position_num):
        debug_print("visited: " + util.pos_to_gtppos(util.num_to_pos(out[0][i])) + ":" + str(T_prior[i]))
    # 再打印一下耗时
    debug_print("Time：" + str(time.time() - t))

    # debug_print(str(out))
    # print pred
    # print feature[0][2].reshape(1, 361)
    # return pred.argmax()

    """
    如果最优秀的评估值小于一个阈值，那么认输
    """
    if X_prior[max_index] < 0.12:
        return -1
    """
    
    overall_score_max = - 999999
    overall_score_index = -1
    for i in range(4):
        overall_score = X_prior[i] * 0.4 + pred[0][out[0][i]] * 0.6
        if overall_score > overall_score_max:
            overall_score_max = overall_score
            overall_score_index = i
    """
    return out[0][max_index]














"""
gtp协议的输入和输出
"""
def gtp_io():
    global board
    PASS_FLAG = 0
    known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove', 'quit',
                      'name', 'version', 'known_command', 'list_commands', 'protocal_version']
    while True:
        try:
            line = raw_input().strip()
        except EOFError:
            break

        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        if re.match('\d+', command[0]):
            cmdid = command[0]
            command = command[1: ]
        else:
            cmdid = ''
        ret = ''

        if command[0] == 'boardsize':
            debug_print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], 19))
            ret = None
        elif command[0] == 'clear_board':
            board = board.Go()
        elif command[0] == 'komi':
            pass
        elif command[0] == 'play':
            if command[2].upper() == 'PASS':
                go.place_stone_num(-1)
                PASS_FLAG = 1

            if command[1].upper() == 'B':# and board.current_player == BLACK:
                # print command[2]
                go.place_stone_num(util.gtppos_to_num(command[2].upper()))
            elif command[1].upper() == 'W':# and board.current_player == WHITE:
                # print command[2]
                go.place_stone_num(util.gtppos_to_num(command[2].upper()))
        elif command[0] == 'genmove':

            move = -1

            if PASS_FLAG == 1:
                PASS_FLAG = 0
                move = -1
                ret = 'pass'
            else:
                move = make_prediction()
                if move is None:
                    ret = 'pass'
                if move == -1:
                    ret = 'resign'
                else:
                    ret = util.pos_to_gtppos(util.num_to_pos(move))
            go.place_stone_num(move)

        elif command[0] == 'name':
            ret = 'PikachuP, 2018'
        elif command[0] == 'version':
            ret = '0.2'
        elif command[0] == 'list_commands':
            ret = '\n'.join(known_commands)
        elif command[0] == 'protocol_version':
            ret = '2'
        elif command[0] == 'quit':
            print '=%s \n\n' % (cmdid, ),
            exit(0)
        else:
            debug_print("Unknown Command! ")
            ret = None

        if ret is not None:
            print '=%s %s\n\n' % (cmdid, ret),
        else:
            print '?%s ???\n\n' % (cmdid, ),
        sys.stdout.flush()



def main():
    gtp_io()

if __name__=="__main__":
    main()
