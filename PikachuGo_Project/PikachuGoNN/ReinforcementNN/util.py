# -*- coding: utf-8 -*-


"""
util文件下的所有函数都被用来做棋盘坐标的转换。
"""


_column_arr = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'J':8,'K':9,'L':10,'M':11,'N':12,'O':13,'P':14,'Q':15,'R':16,'S':17,'T':18}
_column_arr_re = "ABCDEFGHJKLMNOPQRST"
num_ord_A = ord('a')

"""
将0-360这361个位置转换为坐标的表示形式
输入一个数，输出坐标
"""

def num_to_pos(num):
    return num // 19, num % 19


"""

输入一个tuple，表示坐标，输出0-360代表位置
输出-1代表位置错误
举例：
print pos_to_num((18, 18))
361
print pos_to_num((0, 19))
-1

"""
def pos_to_num(pos):
    assert isinstance(pos, tuple)
    if 0 <= pos[0] < 19 and 0 <= pos[1] < 19:
        return pos[0] * 19 + pos[1]
    return -1


"""
输入gtp坐标例如L19，输出位置(4, 10)

"""
def gtppos_to_pos(gtp):
    try:
        if 0 < int(gtp[1: ]) <= 19:
            return 19 - int(gtp[1: ]), _column_arr[gtp[0]]
        else:
            return -2, -2
    except ValueError:
        return -1, -1
    except KeyError:
        return -3, -3

def gtppos_to_num(gtp):
    return pos_to_num(gtppos_to_pos(gtp))


"""pos是一个元组，代表位置，输出了gtp坐标
print pos_to_gtppos((18, 18))
T1
"""
def pos_to_gtppos(pos):
    if 0 <= pos[0] < 19 and 0 <= pos[1] < 19:
        posy = _column_arr_re[pos[1]]
        posx = 19 - pos[0]
        return posy + str(posx)
    else:
        return "X"





"""
输入sgf格式文件的坐标，输出的是一个元组，代表位置
print sgf_to_pos('qc')
(2, 16)

"""

def sgf_to_pos(sgf):
    if sgf == 'tt' or len(sgf) < 2:
        return -1, -1
    posx = ord(sgf[1]) - num_ord_A
    posy = ord(sgf[0]) - num_ord_A
    return posx, posy



"""


输入sgf格式文件的坐标，输出的是一个元组，代表位置
print sgf_to_pos('qc')
(2, 16)

"""
def sgf_to_num(sgf):
    if sgf == 'tt' or len(sgf) < 2:
        return -1, -1
    posx = ord(sgf[1]) - num_ord_A
    posy = ord(sgf[0]) - num_ord_A
    return pos_to_num((posx, posy))

"""


print pos_to_sgf((13, 17))
rn
"""
def pos_to_sgf(pos):
    if 0 <= pos[0] < 19 and 0 <= pos[1] < 19:
        return chr(pos[1] + num_ord_A) + chr(pos[0] + num_ord_A)
    else:
        return 'tt'



















