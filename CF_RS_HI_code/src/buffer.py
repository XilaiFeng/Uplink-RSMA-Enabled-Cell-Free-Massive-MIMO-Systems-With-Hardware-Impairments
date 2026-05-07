import os
import numpy as np
import torch
from src.config import *


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        # # TODO 动作
        # if power_num == 1:
        #     self.actions[self._p] = float(action)
        # else:
        self.actions[self._p].copy_(torch.from_numpy(action)) # 离散
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 2), dtype=torch.float, device=device)  # 修改為2列，分別存儲order和power的對數概率
        self.next_states = torch.empty(
            (self.total_size, state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p].copy_(torch.from_numpy(log_pi))  # 直接複製整個log_pi數組
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

# class RolloutBuffer:
#     def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
#         self._n = 0
#         self._p = 0
#         self.mix = mix
#         self.buffer_size = buffer_size
#         self.total_size = 1 * buffer_size
#         self.device = device
#
#         # 主数据存储
#         self.states = torch.empty((self.total_size, state_shape), dtype=torch.float, device=device)
#         self.actions = torch.empty((self.total_size, action_shape), dtype=torch.float, device=device)
#         self.rewards = torch.empty((self.total_size, 1), dtype=torch.float, device=device)
#         self.dones = torch.empty((self.total_size, 1), dtype=torch.float, device=device)
#         self.log_pis = torch.empty((self.total_size, 1), dtype=torch.float, device=device)
#         self.next_states = torch.empty((self.total_size, state_shape), dtype=torch.float, device=device)
#
#         # 优先级相关
#         self.priorities = torch.ones(self.total_size, dtype=torch.float, device=device)  # 初始优先级为1
#         self.max_priority = 1.0  # 跟踪最大优先级
#
#         self.extract_counts = torch.zeros(buffer_size, dtype=torch.int32, device=device)
#
#     def append(self, state, action, reward, done, log_pi, next_state):
#         # 存储数据
#         self.states[self._p].copy_(torch.from_numpy(state))
#         self.actions[self._p].copy_(torch.from_numpy(action))
#         self.rewards[self._p] = float(reward)
#         self.dones[self._p] = float(done)
#         self.log_pis[self._p] = float(log_pi)
#         self.next_states[self._p].copy_(torch.from_numpy(next_state))
#
#         # 设置新样本优先级为当前最大值
#         self.priorities[self._p] = self.max_priority
#
#         self._p = (self._p + 1) % self.total_size
#         self._n = min(self._n + 1, self.total_size)
#
#         self.explore_eps = 0.3
#
#
#     def get(self):
#         """保持原有接口兼容性"""
#         assert self._p % self.buffer_size == 0
#         start = (self._p - self.buffer_size) % self.total_size
#         idxes = slice(start, start + self.buffer_size)
#         return (self.states[idxes], self.actions[idxes], self.rewards[idxes],
#                 self.dones[idxes], self.log_pis[idxes], self.next_states[idxes])
#
#     def sample(self, batch_size, alpha=0.6, beta=0.4, gamma=0.3, explore_eps=0.1):
#         """基于rank的优先级采样"""
#         valid_size = min(self._n, self.total_size)
#         #
#         # 计算rank概率
#         sorted_priorities, sorted_indices = torch.sort(self.priorities[:valid_size], descending=True)
#         gae_ranks = torch.arange(1, valid_size + 1, device=self.device)[sorted_indices.argsort()]
#
#         _, sorted_count_idx = torch.sort(self.extract_counts[:valid_size], descending=False)
#         count_ranks = torch.arange(1, valid_size + 1, device=self.device)[sorted_count_idx.argsort()]
#
#         ranks = gae_ranks + gamma * count_ranks
#
#         probs = 1.0 / (ranks.float() ** alpha)
#         probs /= probs.sum()
#
#         # 加入混合探索（方案2）
#         uniform_probs = torch.ones_like(probs) / valid_size
#         mixed_probs = (1 - explore_eps) * probs + explore_eps * uniform_probs
#         mixed_probs /= mixed_probs.sum()
#
#         # 采样
#         sampled_indices = torch.multinomial(mixed_probs, batch_size, replacement=True)
#         self.extract_counts[sampled_indices] += 1
#
#         # 计算重要性权重
#         weights = (valid_size * probs[sampled_indices]) ** (-beta)
#         weights /= weights.max() + 1e-8  # 防止除零
#
#         # # 1. 基于GAE的Rank计算
#         # _, sorted_gae_idx = torch.sort(self.priorities[:valid_size], descending=True)
#         # gae_ranks = torch.arange(valid_size, device=self.device)[sorted_gae_idx.argsort()]  # [0~N-1]
#         #
#         # # 2. 采样次数惩罚项（反向Rank）
#         # _, sorted_count_idx = torch.sort(self.extract_counts[:valid_size], descending=True)
#         # count_penalty = torch.arange(valid_size, device=self.device)[sorted_count_idx.argsort()]  # 被采样次数越多，惩罚越高
#         #
#         # # 3. 组合优先级（可调权重）
#         # combined_priority = (1.0 * gae_ranks) - (0.3 * count_penalty)  # 调整系数控制影响
#         #
#         # # 4. 加入随机探索项
#         # random_noise = torch.rand_like(combined_priority.float()) * self.explore_eps * valid_size
#         # combined_priority += random_noise
#         #
#         # # 5. 转换为概率分布
#         # probs = torch.softmax(combined_priority, dim=0)
#
#         # # 采样并更新次数
#         # sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
#         # self.extract_counts[sampled_indices] += 1
#         #
#         # # 重要性权重校正
#         # weights = (valid_size * probs[sampled_indices]) ** (-beta)
#         # weights /= weights.max() + 1e-8
#
#         return (
#             self.states[sampled_indices],
#             self.actions[sampled_indices],
#             self.rewards[sampled_indices],
#             self.dones[sampled_indices],
#             self.log_pis[sampled_indices],
#             self.next_states[sampled_indices],
#             sampled_indices,
#             weights
#         )
#
#     def update_priorities(self, indices, new_priorities):
#         """更新优先级（需在PPO更新后调用）"""
#         new_priorities = torch.abs(new_priorities.squeeze()) + 1e-6  # 防止零优先级
#         self.priorities[indices] = new_priorities
#         self.max_priority = max(self.max_priority, new_priorities.max().item())
