# -*- coding: utf-8 -*-
import numpy as np
import zobrist as zb
import util
import copy
import sys
sys.setrecursionlimit(3000)

"""

PikachuGo设计了16个棋盘特征
    0 -> 本方棋子的位置
    1 -> 对方棋子的位置
    2 -> 空点
    3 -> 1气
    4 -> 2气
    5 -> 3气
    6 -> 4气
    7 -> 5气
    8 -> 上3手
    9 -> 上一手
    10 -> 极有可能成为眼位
    11 -> 填充自己眼位的走子
    12 -> 打劫规则不允许的棋子
    13 -> 本方征子不利
    14 -> 对方征子不利
    15 -> 本方是否执黑
    

board.py
含有围棋棋盘盘面的表示类Go。该类实现了以下方法：

def current_player(self)                                获得当前的行动的棋手。
def is_eye(self, pos)                                   粗略判断眼位，返回值可能是EYE_MUST/EYE_HIGH_PRO/EYE_NOT
def generate(self)                                      用于生成棋盘的特征
def generate_fast(self)                                 用于生成快速走子的盘面特征
def judge_ladder_oppo(self, key):                       模拟对方走子，判断是否征子有利/不利
def judge_ladder(self, key):                            模拟本方走子，判断是否征子有利/不利
def evaluate(self):                                     用chen's方法判断胜负。黑胜返回1，白胜利返回0。
def print_board(self):                                  调试用，打印棋盘
def is_valid_move_numpos(self, pos):                    判断是否为可入点
def place_stone_num(self, pos):                         落子，如果落子点不合法，返回False。请在落子时加以判断。

属性：
        self.board = np.zeros((19, 19), dtype=np.int8)
        self.round = 1                                  round为1的时候，黑走棋，2，白走棋
        self.group = [ { } ,  { } ]                     棋串
        self.history = [ [-1], [-1] ]                   走子历史
        self.captured = [0, 0]                          被提走的棋子数量（未使用）
        self.zob_arr = zb.get_zobrist_random()          初始盘面的Zobrist哈希值
        self.zob_history = [ ]                          历史盘面的Zobrist哈希值
        self.zob_history.append(zb.get_init_hash(zob_arr=self.zob_arr))
        self.komi = 3.75                                贴目
        self.rule = "Chinese"                           规则
        self.found = False                              判断落子是否合法时用到的属性
        self.ladder_string = []                         判断征子时用到的属性

"""

EMPTY_STONE =  0
BLACK_STONE = +1
WHITE_STONE = -1

SIDE_BLACK = 0
SIDE_WHITE = 1

"""
粗略的眼位判断
"""
EYE_MUST = 3
EYE_HIGH_PRO = 2
EYE_MAYBE = 1
EYE_NOT = 0


"""
DEPTH用于陈克训的形势分析算法
"""
DEPTH = [1, 2, 4, 8, 16, 32, 64]

# 上，右， 下， 左
dx = [0, 1, 0,-1]
dy = [-1,0, 1, 0]
# 左上，右上，左下，右下
kdx = [-1, -1, 1, 1]
kdy = [-1, 1, -1, 1]


"""
Group：棋串类，服务于Go类，用于指示当前盘面的棋串
"""
class Group(object):
    def __init__(self, id):
        self.id = id
        self.stone = set()
        self.liberty = set()

    def add_stone(self, pos):
        self.stone.add(pos)

    def add_liberty(self, pos):
        self.liberty.add(pos)

    def count_liberty(self):
        return len(self.liberty)

    def count_stone(self):
        return len(self.stone)

    def remove_liberty(self, pos):
        self.liberty.discard(pos)

    # 看是否有位于xxx的气
    def has_liberty(self, pos):
        return pos in self.liberty

    # 看是否有位于xxx的棋子
    def has_stone(self, pos):
        return pos in self.stone

    def merge_stone(self, other_group):
        assert isinstance(other_group, Group)
        self.stone = self.stone.union(other_group.stone)

    # 重新算气
    def recount_liberty(self, side, go_board):
        assert isinstance(go_board, np.ndarray)
        for s in self.stone:
            posx, posy = util.num_to_pos(s)
            for i in range(4):
                if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                    if go_board[posx + dx[i]][posy + dy[i]] == EMPTY_STONE:
                        self.add_liberty(util.pos_to_num((posx + dx[i], posy + dy[i])))



class Go(object):
    """
    获得当前的行动的棋手。
    """
    def _current_player(self):
        if self.round % 2 == 1:
            return SIDE_BLACK
        else:
            return SIDE_WHITE

    def current_player(self):
        return self._current_player()
    """
    获取当前棋手的对方棋手
    """
    def _opposite_player(self):
        if self.round % 2 == 1:
            return SIDE_WHITE
        else:
            return SIDE_BLACK

    def _current_stone(self):
        if self.round % 2 == 1:
            return BLACK_STONE
        else:
            return WHITE_STONE

    def _opposite_stone(self):
        if self.round % 2 == 1:
            return WHITE_STONE
        else:
            return BLACK_STONE

    def __init__(self):
        self.board = np.zeros((19, 19), dtype=np.int8)
        # round为1的时候，黑走棋，2，白走棋
        self.round = 1
        self.group = [ { } ,  { } ]
        self.history = [ [-1], [-1] ]
        self.captured = [0, 0]
        self.zob_arr = zb.get_zobrist_random()
        self.zob_history = [ ]
        self.zob_history.append(zb.get_init_hash(zob_arr=self.zob_arr))
        self.komi = 3.75
        self.rule = "Chinese"
        self.found = False
        self.ladder_string = []

    def is_eye(self, pos):
        current_stone = self._current_stone()
        opp_stone = self._opposite_stone()
        # 先判断旁边的四个角
        pos_x, pos_y = util.num_to_pos(pos)
        if self.board[pos_x][pos_y] != EMPTY_STONE:
            return EYE_NOT
        if pos == 0 and self.board[0][1] == self.board[1][0] == self.board[1][1] == current_stone:
            return EYE_MUST
        if pos == 18 and self.board[0][17] == self.board[1][17] == self.board[1][18] == current_stone:
            return EYE_MUST
        if pos == 342 and self.board[17][0] == self.board[17][1] == self.board[18][1] == current_stone:
            return EYE_MUST
        if pos == 360 and self.board[18][17] == self.board[17][17] == self.board[17][18] == current_stone:
            return EYE_MUST

        if pos_x == 0 and 0 < pos_y < 18:
            cnt = 0
            if self.board[pos_x][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y + 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y + 1] == opp_stone:
                cnt -= 1
            if cnt == 5:
                return EYE_MUST
            if cnt == 4:
                return EYE_HIGH_PRO
            """
        if pos_x == 18 and 0 < pos_y < 18:
            if self.board[pos_x][pos_y - 1] == self.board[pos_x][pos_y + 1] == self.board[pos_x - 1][pos_y - 1] \
                    == self.board[pos_x - 1][pos_y] == self.board[pos_x - 1][pos_y + 1] == current_stone:
                return EYE_MUST
            """
        if pos_x == 18 and 0 < pos_y < 18:
            cnt = 0
            if self.board[pos_x][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y + 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x - 1][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x - 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x - 1][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y + 1] == opp_stone:
                cnt -= 1
            if cnt == 5:
                return EYE_MUST
            if cnt == 4:
                return EYE_HIGH_PRO
        """

        if pos_y == 0 and 0 < pos_x < 18:
            if self.board[pos_x - 1][pos_y] == self.board[pos_x + 1][pos_y] == self.board[pos_x - 1][pos_y + 1] \
                    == self.board[pos_x][pos_y + 1] == self.board[pos_x + 1][pos_y + 1] == current_stone:
                return EYE_MUST
        """
        if pos_y == 0 and 0 < pos_x < 18:
            cnt = 0
            if self.board[pos_x - 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x - 1][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y + 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y + 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y + 1] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y + 1] == opp_stone:
                cnt -= 1
            if cnt == 5:
                return EYE_MUST
            if cnt == 4:
                return EYE_HIGH_PRO

        if pos_y == 18 and 0 < pos_x < 18:
            cnt = 0
            if self.board[pos_x - 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y] == opp_stone:
                cnt -= 1
            if self.board[pos_x - 1][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x - 1][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x][pos_y - 1] == opp_stone:
                cnt -= 1
            if self.board[pos_x + 1][pos_y - 1] == current_stone:
                cnt += 1
            elif self.board[pos_x + 1][pos_y - 1] == opp_stone:
                cnt -= 1
            if cnt == 5:
                return EYE_MUST
            if cnt == 4:
                return EYE_HIGH_PRO



        if 0 < pos_x < 18 and 0 < pos_y < 18:
            cnt_beside = 0
            cnt_not_beside = 0
            score = 0.0
            for i in range(4):
                if self.board[pos_x + dx[i]][pos_y + dy[i]] == current_stone:
                    cnt_beside += 1
                    score += 2.5
                elif self.board[pos_x + dx[i]][pos_y + dy[i]] == opp_stone:
                    score -= 4
            for i in range(4):
                if self.board[pos_x + kdx[i]][pos_y + kdy[i]] == current_stone:
                    cnt_not_beside += 1
                    score += 1
                elif self.board[pos_x + kdx[i]][pos_y + kdy[i]] == opp_stone:
                    score -= 2
            if cnt_beside == 4 and cnt_not_beside >= 3:
                return EYE_MUST
            if score > 8:
                return EYE_HIGH_PRO

        return EYE_NOT

    # 用来生成棋盘的特征
    """
    0->本方棋子的位置
    1->对方棋子的位置
    2->空点
    3->1气
    4->2气
    5->3气
    6->4气
    7->5气
    8->上3手
    9->上一手
    10->极有可能成为眼位
    11->填充自己眼位的走子
    12->打劫规则不允许的棋子
    13->本方征子不利
    14->对方征子不利
    15->本方是否执黑
    """
    def generate(self):
        features = np.zeros((1, 16, 19, 19), dtype=np.int8)
        current_stone = self._current_stone()
        opposite_stone = self._opposite_stone()


        """本方棋子的位置，和对方棋子的位置2/16
        """
        features[0][0] = (self.board == current_stone) + 0
        features[0][1] = (self.board == opposite_stone) + 0
        for i in range(361):
            px, py = util.num_to_pos(i)
            if self.is_valid_move_numpos(i) == 1:
                """空点的位置3/16
                """
                features[0][2][px][py] = 1
            elif self.is_valid_move_numpos(i) == -1:
                """如果是打劫，那么+打劫位置4/16
                """
                features[0][12][px][py] = 1
            eye = self.is_eye(i)
            if eye == 2:
                """对于本方来说，极有可能成为眼位的空点5/16
                """
                features[0][10][px][py] = 1
            elif eye == 3:
                """已经是眼位的点6/16
                """
                features[0][11][px][py] = 1

        """5个气位的特征，外加两个征子的特征 7,8,9,10,11,12,13/16
        """
        for key, value in self.group[self._current_player()].items():
            if value.count_liberty() == 1:
                """1气要做征子判断！对于本方的1气棋子来说，
                先自己走，然后让对方走
                """
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    if self.judge_ladder(key) == 1:
                        features[0][13][x][y] = 1
                    features[0][3][x][y] = 1
            elif value.count_liberty() == 2:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][4][x][y] = 1
            elif value.count_liberty() == 3:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][5][x][y] = 1
            elif value.count_liberty() == 4:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][6][x][y] = 1
            elif value.count_liberty() >= 5:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][7][x][y] = 1
        for key, value in self.group[self._opposite_player()].items():
            if value.count_liberty() == 1:
                """对方的1气也做征子判断，但是本方先让一手，然后看征子情况
                """
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    if self.judge_ladder_oppo(key) == 1:
                        features[0][14][x][y] = 1
                    features[0][3][x][y] = 1
            elif value.count_liberty() == 2:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][4][x][y] = 1
            elif value.count_liberty() == 3:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][5][x][y] = 1
            elif value.count_liberty() == 4:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][6][x][y] = 1
            elif value.count_liberty() >= 5:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][7][x][y] = 1

        if len(self.history[self._opposite_player()]) > 0 and self.history[self._opposite_player()][-1] >= 0:
            x, y = util.num_to_pos(self.history[self._opposite_player()][-1])
            features[0][9][x][y] = 1
            features[0][8][x][y] = 1

        if len(self.history[self._opposite_player()]) > 1 and self.history[self._opposite_player()][-2] >= 0:
            x, y = util.num_to_pos(self.history[self._opposite_player()][-2])
            features[0][8][x][y] = 1

        if len(self.history[self._current_player()]) > 0 and self.history[self._current_player()][-1] >= 0:
            x, y = util.num_to_pos(self.history[self._current_player()][-1])
            features[0][8][x][y] = 1

        if self._current_player() == SIDE_BLACK:
            features[0][15] = np.ones((19, 19), dtype=np.int8)
        return features


    def generate_fast(self):
        features = np.zeros((1, 8, 19, 19), dtype=np.int8)
        current_stone = self._current_stone()
        opposite_stone = self._opposite_stone()


        """本方棋子的位置，和对方棋子的位置2/16
        """
        features[0][0] = (self.board == current_stone) + 0
        features[0][1] = (self.board == opposite_stone) + 0
        """5个气位的特征，外加两个征子的特征 7,8,9,10,11,12,13/16
        """
        for key, value in self.group[self._current_player()].items():
            if value.count_liberty() == 1:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][2][x][y] = 1
            elif value.count_liberty() == 2:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][3][x][y] = 1
            elif value.count_liberty() == 3:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][4][x][y] = 1
            elif value.count_liberty() == 4:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][5][x][y] = 1
            elif value.count_liberty() >= 5:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][6][x][y] = 1
        for key, value in self.group[self._opposite_player()].items():
            if value.count_liberty() == 1:

                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][2][x][y] = 1
            elif value.count_liberty() == 2:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][3][x][y] = 1
            elif value.count_liberty() == 3:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][4][x][y] = 1
            elif value.count_liberty() == 4:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][5][x][y] = 1
            elif value.count_liberty() >= 5:
                for st in value.stone:
                    x, y = util.num_to_pos(st)
                    features[0][6][x][y] = 1

        if len(self.history[self._opposite_player()]) > 0 and self.history[self._opposite_player()][-1] >= 0:
            x, y = util.num_to_pos(self.history[self._opposite_player()][-1])
            features[0][7][x][y] = 1

        return features

    # judge_ladder就是模拟对方走子，判断是否征子有利/不利，然后返回回起始位置
    def judge_ladder_oppo(self, key):
        # 自己copy一份，然后判断征子，模拟走子。
        x = copy.deepcopy(self)
        x.place_stone_num(-1)
        return x.judge_ladder(key)

    def judge_ladder(self, key):
        self_copy = copy.deepcopy(self)
        pos = -1
        opp = set()
        for st in self_copy.group[self_copy._current_player()][key].stone:
            posx, posy = util.num_to_pos(st)
            pos = st
            for i in range(4):
                if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                    if self_copy.board[posx + dx[i]][posy + dy[i]] == self_copy._opposite_stone():
                        opp.add(util.pos_to_num((posx + dx[i], posy + dy[i])))
        """
        
        for st in opp:
            for key, value in self_copy.group[self_copy._opposite_player()].items():
                if value.has_stone(st):
                    self_copy.ladder_string.append(key)
                    break
        """
        self_copy.ladder_string = list(opp)
        return self_copy._judge_ladder(pos)

    """
    chen's模型：
    对于三线以内的影响补偿函数
    """
    def _get_compensation(self, x, y):
        coef_on_x, coef_on_y = 1.0, 1.0
        if x == 2 or x == 16:
            coef_on_x = 1.2
        elif x == 1 or x == 17:
            coef_on_x = 1.5
        elif x == 0 or x == 18:
            coef_on_x = 2.0

        if y == 2 or y == 16:
            coef_on_y = 1.2
        elif y == 1 or y == 17:
            coef_on_y = 1.5
        elif y == 0 or y == 18:
            coef_on_y = 2.0

        return coef_on_x * coef_on_y

    # judge_ladder就是模拟走子，判断是否征子有利/不利，然后返回回起始位置
    def _judge_ladder(self, pos):
        # 自己copy一份，然后判断征子，模拟走子。
        x = copy.deepcopy(self)
        return x.is_ladder(pos)

    # 陈克训的影响模型的递归实现函数
    def _influence_chen(self, arr, x, y, depth):
        if depth < 0:
            return
        if self.board[x][y] == 0 and arr[x][y] < DEPTH[depth]:
            arr[x][y] = DEPTH[depth] * self._get_compensation(x, y)
            for i in range(4):
                if 0 <= x + dx[i] < 19 and 0 <= y + dy[i] < 19:
                    self._influence_chen( arr, x + dx[i], y + dy[i], depth - 1)

    # 判断比之前好多少
    def evaluate_2(self):
        overall_judge = np.zeros((19, 19))
        for x in range(19):
            for y in range(19):
                arr = None
                if self.board[x][y] != EMPTY_STONE:
                    arr = np.zeros((19, 19), dtype=np.int16)
                    for i in range(4):
                        if 0 <= x + dx[i] < 19 and 0 <= y + dy[i] < 19:
                            self._influence_chen(arr, x + dx[i], y + dy[i], 6)
                if self.board[x][y] == BLACK_STONE:
                    overall_judge += arr
                elif self.board[x][y] == WHITE_STONE:
                    overall_judge += - arr

        cnt_black = 0
        cnt_white = 0
        for i in range(19):
            for j in range(19):
                if self.board[i][j] == BLACK_STONE:
                    cnt_black += 256
                elif self.board[i][j] == WHITE_STONE:
                    cnt_white += 256
                else:
                    if overall_judge[i][j] > + 128:
                        cnt_black += 128
                    elif overall_judge[i][j] > 0:
                        cnt_black += overall_judge[i][j]
                    elif overall_judge[i][j] < -128:
                        cnt_white += 128
                    elif overall_judge[i][j] < 0:
                        cnt_white += - overall_judge[i][j]
        return cnt_black, cnt_white

    # 判断胜负, 黑胜返回1，白胜利返回0
    def evaluate(self):
        overall_judge = np.zeros((19, 19))
        for x in range(19):
            for y in range(19):
                arr = None
                if self.board[x][y] != EMPTY_STONE:
                    arr = np.zeros((19, 19), dtype=np.int16)
                    for i in range(4):
                        if 0 <= x + dx[i] < 19 and 0 <= y + dy[i] < 19:
                            self._influence_chen(arr, x + dx[i], y + dy[i], 6)
                if self.board[x][y] == BLACK_STONE:
                    overall_judge += arr
                elif self.board[x][y] == WHITE_STONE:
                    overall_judge += - arr

        cnt_black = 0
        cnt_white = 0
        for i in range(19):
            for j in range(19):
                if self.board[i][j] == BLACK_STONE:
                    cnt_black += 1
                elif self.board[i][j] == WHITE_STONE:
                    cnt_white += 1
                else:
                    if overall_judge[i][j] > 0:
                        cnt_black += 1
                    elif overall_judge[i][j] < 0:
                        cnt_white += 1
        if cnt_black - self.komi > cnt_white:
            return 1
        else:
            return 0

        """
        for i in range(19):
            for j in range(19):
                if overall_judge[i][j] > 128:
                    overall_judge[i][j] = 128
                elif overall_judge[i][j] < -128:
                    overall_judge[i][j] = -128
        """

        # go_plot.go_plot(self.board)
        # go_plot.go_plot_plus2(overall_judge)
        # exit(-12346)



    def is_ladder(self, pos):
        # 先在气位落子，判断是否有2气
        place_pos = -1
        # key就是轮数，value是Gruop棋串
        # 第一个for循环表示：存在pos的棋串是不是只有一口气，如果是的话，记录这口气的位置
        for key, value in self.group[self._current_player()].items():
            if value.has_stone(pos) and value.count_liberty() == 1:
                for place in value.liberty:
                    if self.place_stone_num(place) is True:
                        place_pos = place
                    else:
                        return 0

        # 模拟走子
        # 如果走完子之后，本棋串的气大于2，那么表示可以逃出，返回0，表示征子失败
        if self.group[self._opposite_player()][self.round - 1].count_liberty() > 2:
            return 0
        # 如果棋串的气小于二，表示征子成功，返回1
        if self.group[self._opposite_player()][self.round - 1].count_liberty() < 2:
            return 1
        # 如果气正好等于2，那么对手有两个气位可以走，复制一份。
        if self.group[self._opposite_player()][self.round - 1].count_liberty() == 2:
            ids = set()
            for ladder_stone in self.ladder_string:
                for key, value in self.group[self._current_player()].items():
                    if value.has_stone(ladder_stone):
                        ids.add(key)
            for id_ in ids:
                if self.group[self._current_player()][id_].count_liberty() == 1:
                    return 0
            lst = []
            for lp in self.group[self._opposite_player()][self.round - 1].liberty:
                lst.append(lp)
            g2 = copy.deepcopy(self)
            # 走第一个气位
            if self.place_stone_num(lst[0]) == False:
                return 0
            self.ladder_string.append(lst[0])
            x1 = self._judge_ladder(place_pos)
            if x1 == 1:
                return 1
            # 走第二个气位
            if g2.place_stone_num(lst[1]) == False:
                return 0
            g2.ladder_string.append(lst[1])
            x2 = g2._judge_ladder(place_pos)

            return max(x1, x2)


    def print_board(self):
        for i in range(19):
            for j in range(19):
                if self.board[i][j] == EMPTY_STONE:
                    print '.',
                elif self.board[i][j] == BLACK_STONE:
                    print 'x',
                else:
                    print 'o',
            print

    def _dfs(self, board, posx, posy, current_stone):
        if self.found is True:
            return
        for i in range(4):
            if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                if board[posx + dx[i]][posy + dy[i]] == EMPTY_STONE:
                    self.found = True
                    return
        board[posx][posy] = -5
        for i in range(4):
            if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                if board[posx + dx[i]][posy + dy[i]] == current_stone:
                    self._dfs(board, posx + dx[i], posy + dy[i], current_stone)

    """    
    def monte_is_valid_move(self, pos):
        land_posx, land_posy = util.sgf_to_pos(pos)
        board_copy = copy.deepcopy(self.board)
        # 最先判断落子位置是不是空点，如果是不是空点，那么一定是不可以下的。
        if self.board[land_posx][land_posy] != EMPTY_STONE:
            return False
        if self._current_player() == SIDE_BLACK:
            c_zb_state = zb.STATE_BLACK
            o_zb_state = zb.STATE_WHITE
            current_stone = BLACK_STONE
            opposite_stone = WHITE_STONE
        else:
            c_zb_state = zb.STATE_WHITE
            o_zb_state = zb.STATE_BLACK
            current_stone = WHITE_STONE
            opposite_stone = BLACK_STONE

        # 判断它的周围是不是有空点，如果有空点，那么一定是可以下的。
        for i in range(4):
            if 0 <= land_posx + dx[i] < 19 and 0 <= land_posy + dy[i] < 19:
                if self.board[land_posx + dx[i]][land_posy + dy[i]] == EMPTY_STONE:
                    return True
        # 判断吃子的话，先算一下zobrist哈希
        num_pos = util.pos_to_num((land_posx, land_posy))
        hash = self.zob_history[-1]

        # 判断是否吃子了，如果吃子了，那么应该是可以下的，但是要判断打劫，用zb哈希
        has_eat = False
        for key, value in self.group[self._opposite_player()].items():
            if value.count_liberty() == 1:
                if value.has_liberty(util.pos_to_num((land_posx, land_posy))):
                    has_eat = True
                    for s in self.group[self._opposite_player()][key].stone:
                        px, py = util.num_to_pos(s)
                        board_copy[px][py] = EMPTY_STONE
                        # 取消对方子的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, o_zb_state, s)
                        # 赋予空点的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, s)
        board_copy[land_posx][land_posy] = current_stone
        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, num_pos)
        hash = zb.get_new_hash(hash, self.zob_arr, c_zb_state, num_pos)

        if has_eat and len(self.zob_history) > 2:
            if self.zob_history[-2] == hash:
                # print 'ko rule!'
                return False
        if has_eat:
            return True

        # 如果没有吃子，那么可能就是往对方里面填子，那么用dfs遍历一下，看看这个棋串周围是否有空点
        # pos_x, pos_y是坐标！
        self.found = False
        self._dfs(board_copy, land_posx, land_posy, current_stone)
        if self.found is False:
            self.found = False
            return False
        return True
    """
    def is_valid_move(self, pos):

        land_posx, land_posy = util.sgf_to_pos(pos)
        board_copy = copy.deepcopy(self.board)
        # 最先判断落子位置是不是空点，如果是不是空点，那么一定是不可以下的。
        if self.board[land_posx][land_posy] != EMPTY_STONE:
            return False
        # 判断它的周围是不是有空点，如果有空点，那么一定是可以下的。
        for i in range(4):
            if 0 <= land_posx + dx[i] < 19 and 0 <= land_posy + dy[i] < 19:
                if self.board[land_posx + dx[i]][land_posy + dy[i]] == EMPTY_STONE:
                    return True
        # 判断吃子的话，先算一下zobrist哈希
        num_pos = util.pos_to_num((land_posx, land_posy))
        hash = self.zob_history[-1]
        if self._current_player() == SIDE_BLACK:
            c_zb_state = zb.STATE_BLACK
            o_zb_state = zb.STATE_WHITE
            current_stone = BLACK_STONE
            opposite_stone = WHITE_STONE
        else:
            c_zb_state = zb.STATE_WHITE
            o_zb_state = zb.STATE_BLACK
            current_stone = WHITE_STONE
            opposite_stone = BLACK_STONE
        # 判断是否吃子了，如果吃子了，那么应该是可以下的，但是要判断打劫，用zb哈希
        has_eat = False
        for key, value in self.group[self._opposite_player()].items():
            if value.count_liberty() == 1:
                if value.has_liberty(util.pos_to_num((land_posx, land_posy))):
                    has_eat = True
                    for s in self.group[self._opposite_player()][key].stone:
                        px, py = util.num_to_pos(s)
                        board_copy[px][py] = EMPTY_STONE
                        # 取消对方子的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, o_zb_state, s)
                        # 赋予空点的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, s)
        board_copy[land_posx][land_posy] = current_stone
        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, num_pos)
        hash = zb.get_new_hash(hash, self.zob_arr, c_zb_state, num_pos)

        if has_eat and len(self.zob_history) > 2:
            if self.zob_history[-2] == hash:
                # print 'ko rule!'
                return False
        if has_eat:
            return True

        # 如果没有吃子，那么可能就是往对方里面填子，那么用dfs遍历一下，看看这个棋串周围是否有空点
        # pos_x, pos_y是坐标！
        self.found = False
        self._dfs(board_copy, land_posx, land_posy, current_stone)
        if self.found is False:
            self.found = False
            return False
        return True

    def is_valid_move_numpos(self, pos):
        land_posx, land_posy = util.num_to_pos(pos)
        board_copy = copy.deepcopy(self.board)
        # 最先判断落子位置是不是空点，如果是不是空点，那么一定是不可以下的。
        if self.board[land_posx][land_posy] != EMPTY_STONE:
            return 0
        # 判断它的周围是不是有空点，如果有空点，那么一定是可以下的。
        for i in range(4):
            if 0 <= land_posx + dx[i] < 19 and 0 <= land_posy + dy[i] < 19:
                if self.board[land_posx + dx[i]][land_posy + dy[i]] == EMPTY_STONE:
                    return 1
        # 判断吃子的话，先算一下zobrist哈希
        num_pos = util.pos_to_num((land_posx, land_posy))
        hash = self.zob_history[-1]
        if self._current_player() == SIDE_BLACK:
            c_zb_state = zb.STATE_BLACK
            o_zb_state = zb.STATE_WHITE
            current_stone = BLACK_STONE
            opposite_stone = WHITE_STONE
        else:
            c_zb_state = zb.STATE_WHITE
            o_zb_state = zb.STATE_BLACK
            current_stone = WHITE_STONE
            opposite_stone = BLACK_STONE
        # 判断是否吃子了，如果吃子了，那么应该是可以下的，但是要判断打劫，用zb哈希
        has_eat = False
        for key, value in self.group[self._opposite_player()].items():
            if value.count_liberty() == 1:
                if value.has_liberty(util.pos_to_num((land_posx, land_posy))):
                    has_eat = True
                    for s in self.group[self._opposite_player()][key].stone:
                        px, py = util.num_to_pos(s)
                        board_copy[px][py] = EMPTY_STONE
                        # 取消对方子的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, o_zb_state, s)
                        # 赋予空点的状态
                        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, s)
        board_copy[land_posx][land_posy] = current_stone
        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, num_pos)
        hash = zb.get_new_hash(hash, self.zob_arr, c_zb_state, num_pos)

        if has_eat and len(self.zob_history) > 2:
            if self.zob_history[-2] == hash:
                # print 'ko rule!'
                return -1
        if has_eat:
            return 1

        # 如果没有吃子，那么可能就是往对方里面填子，那么用dfs遍历一下，看看这个棋串周围是否有空点
        # pos_x, pos_y是坐标！
        self.found = False
        self._dfs(board_copy, land_posx, land_posy, current_stone)
        if self.found is False:
            self.found = False
            return 0
        return 1

    def place_stone_num(self, pos):
        land_posx, land_posy = util.num_to_pos(pos)
        current_player = self._current_player()
        opposite_player = self._opposite_player()
        num_pos = util.pos_to_num((land_posx, land_posy))
        current_stone, opposite_stone = None, None
        # 如果是停一手，则做如下操作
        """
            mistake
            !!!!
            停一手忘记更新棋盘了！
        """
        if land_posx < 0:
            self.history[current_player].append(pos)
            self.zob_history.append(self.zob_history[-1])
            """
            2019年1月7日01:54:54修补该bug
            """
            """
            
            if current_player == SIDE_BLACK:
                current_stone = BLACK_STONE
                opposite_stone = WHITE_STONE
            else:
                current_stone = WHITE_STONE
                opposite_stone = BLACK_STONE
            """
            self.round += 1

            return
        # 先判断是否合法

        if self.is_valid_move(util.pos_to_sgf((land_posx, land_posy))) is False:
            # print 'Invalid!'
            return False
        # 如果合法，则做这些操作：
        """
        1. 减去对方棋串的气
        2. 将新的子加入本方的棋串中
            2.1 遍历本方所有棋串的气，如果有重合，记录备用
            2.2 将这些有重合的都加入这个新的棋串中去
            2.3 删除那些重合加入的棋串
            2.4 重新计算新串的气
        3. 对对面所有没气了的棋串：如果对面没有没气了的棋串，并且本方的气是0，说明自尽，那么对本方这块棋的周围，
            3.1 对于每个没气了的棋串的每个子，看上下左右（注意边界）
            3.2 上下左右如果有本方的棋子，那么加入一个列表备用
            3.3 删除这个棋串
        4. 重新生成棋盘
        5. 记录的这些黑子，判断属于哪些棋块，对这些棋块重新算气
        """
        # 合法了的话，棋盘更新
        if current_player == SIDE_BLACK:
            self.board[land_posx][land_posy] = BLACK_STONE
            current_stone = BLACK_STONE
            opposite_stone = WHITE_STONE
        else:
            self.board[land_posx][land_posy] = WHITE_STONE
            current_stone = WHITE_STONE
            opposite_stone = BLACK_STONE
        # 1. 减去对方棋串的气，并用dead列表记录哪些死透了的棋子
        dead = []
        for key, value in self.group[opposite_player].items():
            assert isinstance(value, Group)
            value.remove_liberty(num_pos)
            if value.count_liberty() == 0:
                dead.append(key)

        # 遍历本方所有棋串的气，如果有重合，记录备用在merge里面
        merge = []
        for key, value in self.group[current_player].items():
            assert isinstance(value, Group)
            if value.has_liberty(num_pos):
                merge.append(key)
        # 2.2 新建一个棋串，将这些有重合的都加入这个新的棋串中去
        self.group[current_player][self.round] = Group(self.round)
        self.group[current_player][self.round].add_stone(num_pos)
        # 2.3 删除那些重合加入的棋串
        for ids in merge:
            self.group[current_player][self.round].merge_stone(self.group[current_player][ids])
            self.group[current_player].pop(ids)
        # 2.4 重新计算新串的气(有一个不必要的参数side)
        self.group[current_player][self.round].recount_liberty(current_player, self.board)

        # 3 对面所有没气了的棋串，在dead里面，删掉这些子，但是记录一下他们
        record_dead_stone = set()
        for d in dead:
            for st in self.group[opposite_player][d].stone:
                record_dead_stone.add(st)
                posx, posy = util.num_to_pos(st)
                # 在这里清除了对方的死子
                self.board[posx][posy] = EMPTY_STONE
        # 由于记录了死子的位置，所以看一下它的上下左右，有没有本方的子
        # 有的话，记录本方这些子的棋串号
        recount_liberty_group = set()
        for dead_pos in record_dead_stone:
            posx, posy = util.num_to_pos(dead_pos)
            for i in range(4):
                if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                    if self.board[posx + dx[i]][posy + dy[i]] == current_stone:
                        for key, value in self.group[current_player].items():
                            if value.has_stone(util.pos_to_num((posx + dx[i], posy + dy[i]))):
                                recount_liberty_group.add(key)
        # 删掉这些棋串
        for del_ids in dead:
            self.group[opposite_player].pop(del_ids)
        # 对需要重新算气的本方棋串重新算气
        for rec in recount_liberty_group:
            self.group[current_player][rec].recount_liberty(current_player, self.board)

        if current_player == SIDE_BLACK:
            c_zb_state = zb.STATE_BLACK
            o_zb_state = zb.STATE_WHITE
        else:
            c_zb_state = zb.STATE_WHITE
            o_zb_state = zb.STATE_BLACK

        hash = self.zob_history[-1]
        # 取消本方落子的那个空点
        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, util.pos_to_num((land_posx, land_posy)))
        # 本方落子在那个空点
        hash = zb.get_new_hash(hash, self.zob_arr, c_zb_state, util.pos_to_num((land_posx, land_posy)))

        for ds in record_dead_stone:
            hash = zb.get_new_hash(hash, self.zob_arr, o_zb_state, ds)
            hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, ds)

        self.zob_history.append(hash)
        self.history[current_player].append(pos)
        # self.print_board()
        self.round += 1
        return True


    def place_stone(self, pos):
        land_posx, land_posy = util.sgf_to_pos(pos)
        num = util.pos_to_num((land_posx, land_posy))
        self.place_stone_num(num)
        """
        current_player = self._current_player()
        opposite_player = self._opposite_player()
        num_pos = util.pos_to_num((land_posx, land_posy))
        current_stone, opposite_stone = None, None
        # 如果是停一手，则做如下操作
        if land_posx < 0:
            self.history[current_player].append('tt')
            self.zob_history.append(self.zob_history[-1])
            self.round += 1
            return
        # 先判断是否合法

        if self.is_valid_move(pos) is False:
            # print 'Invalid!'
            return
        # 如果合法，则做这些操作：
        """
        """
        1. 减去对方棋串的气
        2. 将新的子加入本方的棋串中
            2.1 遍历本方所有棋串的气，如果有重合，记录备用
            2.2 将这些有重合的都加入这个新的棋串中去
            2.3 删除那些重合加入的棋串
            2.4 重新计算新串的气
        3. 对对面所有没气了的棋串：如果对面没有没气了的棋串，并且本方的气是0，说明自尽，那么对本方这块棋的周围，
            3.1 对于每个没气了的棋串的每个子，看上下左右（注意边界）
            3.2 上下左右如果有本方的棋子，那么加入一个列表备用
            3.3 删除这个棋串
        4. 重新生成棋盘
        5. 记录的这些黑子，判断属于哪些棋块，对这些棋块重新算气
        """
        """
        # 合法了的话，棋盘更新
        if current_player == SIDE_BLACK:
            self.board[land_posx][land_posy] = BLACK_STONE
            current_stone = BLACK_STONE
            opposite_stone = WHITE_STONE
        else:
            self.board[land_posx][land_posy] = WHITE_STONE
            current_stone = WHITE_STONE
            opposite_stone = BLACK_STONE
        # 1. 减去对方棋串的气，并用dead列表记录哪些死透了的棋子
        dead = []
        for key, value in self.group[opposite_player].items():
            assert isinstance(value, Group)
            value.remove_liberty(num_pos)
            if value.count_liberty() == 0:
                dead.append(key)

        # 遍历本方所有棋串的气，如果有重合，记录备用在merge里面
        merge = []
        for key, value in self.group[current_player].items():
            assert isinstance(value, Group)
            if value.has_liberty(num_pos):
                merge.append(key)
        # 2.2 新建一个棋串，将这些有重合的都加入这个新的棋串中去
        self.group[current_player][self.round] = Group(self.round)
        self.group[current_player][self.round].add_stone(num_pos)
        # 2.3 删除那些重合加入的棋串
        for ids in merge:
            self.group[current_player][self.round].merge_stone(self.group[current_player][ids])
            self.group[current_player].pop(ids)
        # 2.4 重新计算新串的气(有一个不必要的参数side)
        self.group[current_player][self.round].recount_liberty(current_player, self.board)

        # 3 对面所有没气了的棋串，在dead里面，删掉这些子，但是记录一下他们
        record_dead_stone = set()
        for d in dead:
            for st in self.group[opposite_player][d].stone:
                record_dead_stone.add(st)
                posx, posy = util.num_to_pos(st)
                # 在这里清除了对方的死子
                self.board[posx][posy] = EMPTY_STONE
        # 由于记录了死子的位置，所以看一下它的上下左右，有没有本方的子
        # 有的话，记录本方这些子的棋串号
        recount_liberty_group = set()
        for dead_pos in record_dead_stone:
            posx, posy = util.num_to_pos(dead_pos)
            for i in range(4):
                if 0 <= posx + dx[i] < 19 and 0 <= posy + dy[i] < 19:
                    if self.board[posx + dx[i]][posy + dy[i]] == current_stone:
                        for key, value in self.group[current_player].items():
                            if value.has_stone(util.pos_to_num((posx + dx[i], posy + dy[i]))):
                                recount_liberty_group.add(key)
        # 删掉这些棋串
        for del_ids in dead:
            self.group[opposite_player].pop(del_ids)
        # 对需要重新算气的本方棋串重新算气
        for rec in recount_liberty_group:
            self.group[current_player][rec].recount_liberty(current_player, self.board)

        if current_player == SIDE_BLACK:
            c_zb_state = zb.STATE_BLACK
            o_zb_state = zb.STATE_WHITE
        else:
            c_zb_state = zb.STATE_WHITE
            o_zb_state = zb.STATE_BLACK

        hash = self.zob_history[-1]
        # 取消本方落子的那个空点
        hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, util.pos_to_num((land_posx, land_posy)))
        # 本方落子在那个空点
        hash = zb.get_new_hash(hash, self.zob_arr, c_zb_state, util.pos_to_num((land_posx, land_posy)))

        for ds in record_dead_stone:
            hash = zb.get_new_hash(hash, self.zob_arr, o_zb_state, ds)
            hash = zb.get_new_hash(hash, self.zob_arr, zb.STATE_EMPTY, ds)

        self.zob_history.append(hash)
        self.history[current_player].append(pos)
        # self.print_board()
        self.round += 1
        """
