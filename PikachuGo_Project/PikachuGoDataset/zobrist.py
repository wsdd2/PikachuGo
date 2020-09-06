# -*- coding: utf-8 -*-
import numpy as np

"""

Zobrist 效率非常高，每下一步棋，只需要进行一次 异或 操作，相对于对每一步棋的打分来说，
这一次异或操作带来的性能消耗可以忽略不计。Zobrist具体实现如下：

初始化一个两个 Zobrist[M][M] 的二维数组，其中M是五子棋的棋盘宽度。当然也可以是 Zobrist[M*M] 的一维数组。
设置两个是为了一个表示黑棋，一个表示白旗。
上述数组的每一个都填上一个随机数，至少保证是32位的长度（即32bit)，最好是64位。初始键值也设置一个随机数。
每下一步棋，则用当前键值异或Zobrist数组里对应位置的随机数，得到的结果即为新的键值。
如果是删除棋子（悔棋），则再异或一次即可。
"""
"""
哈希的三个状态
"""
STATE_EMPTY = 0
STATE_BLACK = 1
STATE_WHITE = 2


def get_zobrist_random():
    zob_arr = np.random.randint(0, 2**64, size=(3, 361), dtype=np.uint64)
    return zob_arr

def get_init_hash(zob_arr):
    hash = zob_arr[STATE_EMPTY][0]
    for i in range(1, 361):
        hash = np.bitwise_xor(hash, zob_arr[STATE_EMPTY][i])
    return hash



"""
hash: a 64bit unsigned integer number
zob_arr: para
state: STATE_EMPTY = 0, STATE_BLACK = 1, STATE_WHITE = 2
pos: 0-361
"""
def get_new_hash(hash, zob_arr, state, pos):
    return np.bitwise_xor(hash, zob_arr[state][pos])

"""
zob_array = get_zobrist_random()
hash = get_init_hash(zob_array)
print hash
hash = get_new_hash(hash, zob_array, STATE_EMPTY, 0)
hash = get_new_hash(hash, zob_array, STATE_BLACK, 0)
print hash
hash = get_new_hash(hash, zob_array, STATE_BLACK, 0)
hash = get_new_hash(hash, zob_array, STATE_EMPTY, 0)
print hash
"""
