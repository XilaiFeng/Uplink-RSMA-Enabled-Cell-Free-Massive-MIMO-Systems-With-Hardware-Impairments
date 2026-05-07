import os
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def mean_data(data, n):
    l = len(data)
    result = []
    for i in range(1, l//n):
        result.append(np.mean(data[(i-1)*n: i*n]))
    return np.array(result)


# path = r"result/给定方向/N20_Pmax19.95262314968879_Nt8_EH0.001rmin4.0_lr3e-4_unit256_step15.0x10^6"

# path3 = r"resultPES_PPO1.5x10^6"
# path2 = r"resultPPO1.5x10^6"
# path1 = r"resultIPES_PPO1.3x10^6"
# path4 = r"resultIIPES_PPO1.3x10^6"
# path5 = r"resultIIPES_PPO_batch_2561.3x10^6"
# path6 = r"resultIIPES_PPO_batch_dynamic1.3x10^6"
# path7 = r'resultIIPES_PPO_batch_5121.3x10^6'
# path8 = r'resultPPO_batchsize_1281.3x10^6'
path = r'resultPPO_vae_ppo1.5x10^6'
path_att_ppo = r'resultPPO_vae_ppo1.5x10^6'

date = "20250227-2152"

# EES = scio.loadmat(path + "/EE_" + date + ".mat")["EE"].squeeze()
# reward = scio.loadmat(path + "/reward_" + date + ".mat")["reward"].squeeze()
# val_max_reward = np.max(reward)
# print(val_max_reward)
# SSE_ppo_L_30_data = scio.loadmat(path3 + "/reward_20250420-1628.mat")["reward"].squeeze()
# SSE_ppo_L_20_data = scio.loadmat(path2 + "/reward_20250420-1718.mat")["reward"].squeeze()
# SSE_ppo_L_10_data = scio.loadmat(path1 + "/reward_20250420-2040.mat")["reward"].squeeze()
# SSE_ppo_L_40_data = scio.loadmat(path4 + "/reward_20250420-2120.mat")["reward"].squeeze()
# SSE_ppo_L_50_data = scio.loadmat(path5 + "/reward_20250420-2211.mat")["reward"].squeeze()
# SSE_ppo_L_60_data = scio.loadmat(path6 + "/reward_20250421-1002.mat")["reward"].squeeze()
# SSE_ppo_L_70_data = scio.loadmat(path7 + "/reward_20250522-2156.mat")["reward"].squeeze()
# SSE_ppo_L_80_data = scio.loadmat(path8 + "/reward_20250523-2050.mat")["reward"].squeeze()
ppo = scio.loadmat(path_att_ppo + "/reward_20250716-1255.mat")["reward"].squeeze()
vae_ppo = scio.loadmat(path_att_ppo + "/reward_20250716-0454.mat")["reward"].squeeze()

max_r = max(ppo)
print(max_r)
max_r = max(vae_ppo)
print(max_r)

# rate_u = scio.loadmat(path + "/rate_u_" + date + ".mat")["rate_u"].squeeze()

# EHS = scio.loadmat(path + "/EH_" + date + ".mat")["EH"].squeeze()
# rates = scio.loadmat(path + "/rates_" + date + ".mat")["rate"].squeeze()
# EHS1 = EHS[:, 0]
# EHS2 = EHS[:, 1]
# rates1 = rates[:, 0]
# rates2 = rates[:, 1]

# 设置滑动窗口大小
window_size = 3000

# rate_e = np.sum(rate_e, axis=1)
# rate_u = np.sum(rate_u, axis=1)
# time_per_episode_ppo = 4135.37
# time_per_episode_pes_ppo = 3224.39
# #
# total_episodes = 1299000
# episode_numbers = np.arange(1, total_episodes + 1)
# time_points_ppo = episode_numbers * time_per_episode_ppo
# time_points_pes_ppo = episode_numbers * time_per_episode_pes_ppo

# 计算滑动窗口平均值
# smoothed_rewards = np.convolve(reward, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e = np.convolve(rate_e, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_u = np.convolve(SSE_ppo_L_30_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e = np.convolve(SSE_ppo_L_20_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e2 = np.convolve(SSE_ppo_L_10_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e3 = np.convolve(SSE_ppo_L_40_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e4 = np.convolve(SSE_ppo_L_50_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e5 = np.convolve(SSE_ppo_L_60_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e6 = np.convolve(SSE_ppo_L_70_data, np.ones(window_size)/window_size, mode='valid')
# smoothed_rate_e7 = np.convolve(SSE_ppo_L_80_data, np.ones(window_size)/window_size, mode='valid')
smoothed_ppo = np.convolve(ppo, np.ones(window_size)/window_size, mode='valid')
smoothed_vae_ppo = np.convolve(vae_ppo, np.ones(window_size)/window_size, mode='valid')

plt.rcParams["font.sans-serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
# 绘制曲线
plt.figure()
plt.grid()
# plt.plot(x, sum_rate_e, label='eMBB', color='b', linestyle='--', marker='o')
# plt.plot(x, sum_rate_u, label='URLLC', color='r', linestyle='-', marker='>')
# # plt.plot(val_max_reward, label='Utility Function')
# plt.xlabel('Weight')
# plt.ylabel('Sum Rate (bit/s/Hz)')
# plt.legend()
# plt.show()
# plt.plot(smoothed_rate_u, label='PPO', color='b', linestyle='--')
# plt.plot(smoothed_rate_e, label='eMBB rate')
# plt.plot(smoothed_ppo[:1299000], label='PPO', color='k', linestyle='-')
# plt.plot(smoothed_rate_e2, label='PPO', color='g', linestyle='--')
# plt.plot(smoothed_rate_e3, label='PPO', color='m', linestyle='--')
# plt.plot(smoothed_pes_ppo[:1299000], label='PES-PPO', color='r', linestyle='--')
# # plt.plot(smoothed_rate_e6[:1299000], label='PES-PPO', color='r', linestyle='--')
# plt.plot(smoothed_pes_ppo2[:1299000], label='PES-PPO', color='g', linestyle='--')
# plt.plot(smoothed_res_ppo[:1299000], label='ResNet-PPO', color='r', linestyle='--')
plt.plot(smoothed_ppo[:1299000], label='PPO', color='b', linestyle='--')
plt.plot(smoothed_vae_ppo[:1299000], label='VAE-PPO', color='k', linestyle='--')

# plt.plot(smoothed_rate_e5, label='PPO', color='r', linestyle='--')
# plt.xlim([0, 1200000])
# plt.plot(smoothed_rate_u, label='URLLC rate')
plt.xlabel('Episode')
plt.ylabel('Reward')
# plt.title('Reward with Sliding Window Smoothing')
plt.legend()
plt.show()

# path1 = r"result/state不变对比"
#
# EES1 = scio.loadmat(path1 + "/EE_20220616-1933.mat")["EE"].squeeze()
# reward1 = scio.loadmat(path1 + "/reward_20220616-1933.mat")["reward"].squeeze()
# EHS1 = scio.loadmat(path1 + "/EH_20220616-1933.mat")["EH"].squeeze()
# rates1 = scio.loadmat(path1 + "/rates_20220616-1933.mat")["rate"].squeeze()


length = 10000

# EES = mean_data(EES, length)
# reward = mean_data(reward, length)
# EHS1 = mean_data(EHS1, length)
# EHS2 = mean_data(EHS2, length)
# rates1 = mean_data(rates1, length)
# rates2 = mean_data(rates2, length)

# s = "v4"
# EE_path = "result\\1\\EE" + s + ".mat"
# scio.savemat(EE_path, {'EE': EES})
# reward_path = "result\\1\\reward" + s +".mat"
# scio.savemat(reward_path, {'reward': reward})
# rates1_path = "result\\1\\rates1" + s + ".mat"
# scio.savemat(rates1_path, {'rates': rates1})
# rates2_path = "result\\1\\rates2" + s + ".mat"
# scio.savemat(rates2_path, {'rates': rates2})
# EHS1_path = "result\\1\\EHS1" + s + ".mat"
# scio.savemat(EHS1_path, {'EH': EHS1})
# EHS2_path = "result\\1\\EHS2" + s + ".mat"
# scio.savemat(EHS2_path, {'EH': EHS2})


# x = list(range(1, (len(reward)+1)))

# EES1 = mean_data(EES1, length)
# EHS1 = mean_data(EHS1, length)
# reward1 = mean_data(reward1, length)
# rates1 = mean_data(rates1, length)


#
# plt.plot(x, EES, label="state变化")
# plt.plot(x, EES1, label="state不变")
# plt.xlabel("Episode")
# plt.ylabel("EE")
# plt.legend()
# plt.show()
#
# plt.plot(x, rates, label="state变化")
# plt.plot(x, rates1, label="state不变")
# plt.xlabel("Episode")
# plt.ylabel("rate")
# plt.legend()
# plt.show()
#
# plt.plot(x, EHS, label="state变化")
# plt.plot(x, EHS1, label="state不变")
# plt.xlabel("Episode")
# plt.ylabel("EH")
# plt.legend()
# plt.show()
#
# plt.plot(x, reward, label="state变化")
# plt.plot(x, reward1, label="state不变")
# plt.xlabel("Episode")
# plt.ylabel("reward")
# plt.legend()
# plt.show()


# plt.subplot(1, 3, 1)
# plt.plot(x, EES, label="EE")
# plt.plot(x, reward, label="reward")
# plt.xlabel("Epsiode")
# plt.ylabel("reward")
# # plt.ylim(-1, 4)
# plt.legend()
# plt.show()

# plt.subplot(1, 3, 2)
# plt.plot(x, EHS1, label="User 1")
# plt.plot(x, EHS2, label="User 2")
# plt.xlabel("Epsiode")
# plt.ylabel("EH")
# plt.legend()
# plt.show()

# # plt.subplot(1, 3, 3)
# plt.plot(x, rates1, label="User 1")
# plt.plot(x, rates2, label="User 2")
# plt.xlabel("Epsiode")
# plt.ylabel("rate")
# plt.legend()

plt.show()
