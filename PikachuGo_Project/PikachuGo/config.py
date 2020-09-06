# coding=UTF-8
"""
PikachuGo的配置项
"""

"""
PikachuGo的根目录
"""
root_dir            = "E:/PikachuGo/"
"""
策略网络的位置
"""
policy_network_dir  = "model_policy/model"
"""
策略网络id
"""
policy_network_id   = 13620
"""
价值网络的位置
"""
value_network_dir   = "model_value/model"
"""
价值网络id
"""
value_network_id   = 9360
"""
快速走子网络的位置
"""
fast_rollout_dir    = "model_fast/model"
"""
快速走子网络id
"""
fast_rollout_id    = 4980
"""
是否使用UCB进行搜索
"""
enable_ucb          = False
"""
UCB的搜索次数
"""
search_times_ucb    = 50
"""
UCB的最大搜索上限
"""
serach_ucb_limit    = 50
"""
UCB的搜索深度
"""
search_depth_ucb    = 200
"""
UCB 最大搜索的深度
"""
search_depth_ucb_max= 400
"""
UCB的C值
"""
para_c_ucb          = 0.8
"""
策略网络所占权重
"""
policy_weight       = 0.65
"""
价值网络所占权重
"""
value_weight        = 0.35
"""
使用快速走子策略时，展开的落子点个数
"""
search_position_num = 5
"""
假设先验的走子次数
"""
pre_place_num = 8
"""
使用快速走子网络时，添加的噪声
"""
fast_rollout_noise  = 0.13