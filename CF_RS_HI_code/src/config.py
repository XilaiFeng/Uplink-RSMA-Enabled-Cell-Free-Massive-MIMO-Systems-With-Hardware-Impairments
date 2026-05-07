'''
@Project ：DQN_IRL
@File    ：config.py
@Author  ：Ruichen
@Date    ：2021/7/28 9:18
'''

import os
import time
import random
import scipy
from scipy import special
import scipy.io as scio
import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
import torch.nn as nn
from torch.nn import functional as F

from collections import deque

random_seed = 2022

K = 5  # UE number
L = 10  # AP number
J = 2 # sub_message number
N = 2  # antenna number
kappa = 0.9

nbrOfRealizations = 1000

tau_p = 3


# 最大功率（和功率最大）
Pmax = 200  # dBm   28  38  43   48

# 动作维度和策略
# 动作包括：order
# action_policy = "continuous"
# action_policy = "hard_descrate"
action_policy = "tanh_descrate"

action_num = K * J + K * J


# 状态维度
# np.hstack([order, p, rate, reward])
state_num = K * J + K * J + K * J + 1

max_episode_steps = 10 ** 3
decay_steps = 10000

# p_data = scio.loadmat('result_update/p.mat')
# p = p_data['p']
