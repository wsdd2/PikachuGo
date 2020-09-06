#-*-coding:utf-8-*-
import numpy as np


def apply_random_symmetry(train, label):
    assert isinstance(train, np.ndarray)
    assert isinstance(label, np.ndarray)
    n_train = train.shape[0]
    n_dim = train.shape[1]
    # print 'dim: ', n_dim
    for i in range(n_train):
        action = np.random.randint(1, 9)  # 生成1-8的随机数
        if action == 1:
            pass
        elif action == 2:
            label[i] = np.rot90(label[i], k=1)
        elif action == 3:
            label[i] = np.rot90(label[i], k=2)
        elif action == 4:
            label[i] = np.rot90(label[i], k=3)
        elif action == 5:
            label[i] = np.fliplr(label[i])
        elif action == 6:
            label[i] = np.fliplr(np.rot90(label[i], k=1))
        elif action == 7:
            label[i] = np.fliplr(np.rot90(label[i], k=2))
        else:
            label[i] = np.fliplr(np.rot90(label[i], k=3))

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
