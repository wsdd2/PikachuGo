# -*- coding: utf-8 -*-
import os
import re

"""
STEP1--------------------------------------------


这是棋谱处理的第一步操作
删除对我而言不好操作的棋谱。
1.  让子
2.  结果不明的

PikachuP
-------------------------------------------------

在这里设置棋谱的根目录！
"""
"""
---------------------------------参数设置-----------------------------------
"""

root_dir = "E:/PikachuGoDataSample/0_original_data/"

"""
---------------------------------参数设置-----------------------------------
"""


regex_1 = "AB\["
regex_2 = "HA\[\d*\]"
regex_result = "RE\[[WB]\+[RT0-9.]*\]"


list = os.listdir(root_dir)
r1 = re.compile(regex_1)
r2 = re.compile(regex_2)
r3 = re.compile(regex_result)
cnt = 0
rm_list = []
"""
如果匹配到了有
1.  让子
2.  结果不明的
删除它
"""
for file in list:
    fp = open(root_dir + file)
    content = fp.read()
    x = r3.findall(content)
    if len(x) > 0:
        x1 = r1.findall(content)
        x2 = r2.findall(content)
        if len(x1) + len(x2) > 0:
            print file
            rm_list.append(file)
            cnt += 1
        else:
            pass
    else:
        print file
        rm_list.append(file)
        cnt += 1


"""
从rm_list中移除所有的棋谱
在操作系统层面删除！
"""
for rm in rm_list:
    os.remove(root_dir + rm)

print("PikachuP: Processed. Found %d sgf(s), Delete %d sgf(s)" % (len(list), cnt))