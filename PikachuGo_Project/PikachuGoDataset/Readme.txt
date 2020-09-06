本文件夹下的文件用于数据集的制作
围棋采用sgf文件格式，移除符合以下条件的棋谱

让子棋（含有AB[...]）
结果不明的
如果贴目不明确，那么按中国规则，添加3.75的贴目

1_del_handicap.py
2_sgf_process.py
3_data_maker.py
4_policy_drop.py
5_value_drop.py
6_shuffle.py
7_generate_ultimate.py