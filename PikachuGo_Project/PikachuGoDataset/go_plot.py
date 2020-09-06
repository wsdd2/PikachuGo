# -*- coding: utf-8 -*-

from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
fig = plt.figure()
import numpy as np
"""


ax = fig.add_subplot(111)

cir1 = Circle(xy = (0.0, 1.0), radius=0.49,color='black', alpha=1) #第一个参数为圆心坐标，第二个为半径 #第三个为透明度（0-1）
cir2 = Circle(xy = (1.0, 2.0), radius=0.49,color='#cfcfcf', alpha=1) #第一个参数为圆心坐标，第二个为半径 #第三个为透明度（0-1）
ax.add_patch(cir1)
ax.add_patch(cir2)
plt.axis('scaled')
plt.axis('equal')

plt.show()
"""

def go_plot(arr):
    assert isinstance(arr, np.ndarray)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置x主坐标间隔 1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 设置x从坐标间隔 0.1
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置y主坐标间隔 1
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 设置y从坐标间隔 0.1
    cir1 = Circle(xy=(-1, 20), radius=0.48, color='white', alpha=1)
    cir2 = Circle(xy=(19, 0), radius=0.48, color='white', alpha=1)
    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.grid(True)
    for i in range(19):
        for j in range(19):
            if arr[i][j] == +1:
                cir = Circle(xy = (1.0 * j, -1.0 * i + 19), radius=0.48, alpha=1)
                ax.add_patch(cir)
            elif arr[i][j] == -1:
                cir = Circle(xy = (1.0 * j, - 1.0 * i + 19), radius=0.48, color='#d5d5d5', alpha=1)
                ax.add_patch(cir)
    plt.axis('equal')
    plt.xlim((0, 19.5))
    plt.ylim((0, 19.5))
    for i in range(0, 19):
        plt.axvline(i, linewidth=0.2, linestyle = '-')
        plt.axhline(i+1, linewidth=0.2, linestyle = '-')
    plt.axis('off')
    plt.show()

def go_plot_plus(arr, save=False):
    assert isinstance(arr, np.ndarray)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置x主坐标间隔 1
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 设置x从坐标间隔 0.1
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))  # 设置y主坐标间隔 1
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))  # 设置y从坐标间隔 0.1
    cir1 = Circle(xy=(-1, 20), radius=0.48, color='white', alpha=1)
    cir2 = Circle(xy=(19, 0), radius=0.48, color='white', alpha=1)
    ax.add_patch(cir1)
    ax.add_patch(cir2)
    ax.grid(True)
    for i in range(19):
        for j in range(19):
            if arr[i][j] == +1:
                cir = Circle(xy = (1.0 * j, -1.0 * i + 19), radius=0.48, alpha=1)
                ax.add_patch(cir)
            elif arr[i][j] == -1:
                cir = Circle(xy = (1.0 * j, - 1.0 * i + 19), radius=0.48, color='#d5d5d5', alpha=1)
                ax.add_patch(cir)
            elif arr[i][j] > 1:
                cir = Circle(xy=(1.0 * j, - 1.0 * i + 19), radius=0.48, color='red', alpha=arr[i][j] / 4.0)
                ax.add_patch(cir)
    plt.axis('equal')
    plt.xlim((0, 19.5))
    plt.ylim((0, 19.5))
    for i in range(0, 19):
        plt.axvline(i, linewidth=0.2, linestyle = '-')
        plt.axhline(i+1, linewidth=0.2, linestyle = '-')
    plt.axis('off')
    plt.show()
