# -*- coding: utf-8 -*-
import numpy as np


def apply_random_symmetry_without_label(train):
    assert isinstance(train, np.ndarray)
    n_train = train.shape[0]
    n_dim = train.shape[1]
    # print 'dim: ', n_dim
    for i in range(n_train):
        action = np.random.randint(1, 9)  # 生成1-8的随机数
        for j in range(n_dim):
            if action == 1:
                pass
            elif action == 2:
                train[i][j] = np.rot90(train[i][j], k=1)
            elif action == 3:
                train[i][j] = np.rot90(train[i][j], k=2)
            elif action == 4:
                train[i][j] = np.rot90(train[i][j], k=3)
            elif action == 5:
                train[i][j] = np.fliplr(train[i][j])
            elif action == 6:
                train[i][j] = np.fliplr(np.rot90(train[i][j], k=1))
            elif action == 7:
                train[i][j] = np.fliplr(np.rot90(train[i][j], k=2))
            else:
                train[i][j] = np.fliplr(np.rot90(train[i][j], k=3))