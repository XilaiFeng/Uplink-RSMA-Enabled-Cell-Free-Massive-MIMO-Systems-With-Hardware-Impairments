import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from algo.base import Algorithm
from src import env
from src.buffer import RolloutBuffer
from network import StateIndependentPolicy, StateFunction
from network import exponential_decay, natural_exp_decay
from src.config import *


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.,
                 rollout_length=2048, mix_buffer=20, lr_actor=5e-4,
                 lr_critic=5e-4, units_actor=(256, 256), units_critic=(256, 256),
                 epoch_ppo=20, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # random.seed(random_seed)
        # np.random.seed(random_seed)
        # torch.manual_seed(random_seed)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        return step % self.rollout_length == 0

    #改动作
    def step(self, env, state, t, step):
        # t += 1
        # order_part = np.zeros((config.K * config.J, 1))
        order = np.zeros((K * J, 1))
        p = np.zeros((K * J, 1))
        action, log_pi = self.explore(state)
        # action:[order]，长度为 K * J
        # todo 从 action中解析处order动作
        order_data = action[: K * J] + 1
        p_data = 0.5 * Pmax * (action[K * J:]+1)

        for i in reversed(range(K * J)):
            index = np.argmax(order_data)
            order[index] = i
            order_data[index] = 1e-9

        order = order.reshape([K, J])

        p = p_data.reshape([K, J])

        next_state, reward, SINR, done = env.Step(order, p)

        mask = False if t == env._max_episode_steps else done

        # for i in range(M):
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        t += 1

        if done:
            t = 0
            next_state = env.reset()

        return next_state, reward, SINR, order

    # airl重写了此方法，因此不会使用该函数更新，而是使用判别器模块计算的reward进行更新
    def update(self, writer):
        self.learning_steps += 1
        # for _ in range(self.epoch_ppo):
            # states, actions, rewards, dones, log_pis, next_states = \
            #     self.buffer.sample(batch_size=128)
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        # TODO ppo训练逻辑
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        lr = exponential_decay(self.optim_critic, self.lr_critic, self.learning_steps, decay_steps=config.decay_steps,
                               decay_rate=0.1)
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        # if self.learning_steps_ppo % self.epoch_ppo == 0:
        #     writer.add_scalar(
        #         'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        lr = exponential_decay(self.optim_actor, self.lr_actor, self.learning_steps, decay_steps=config.decay_steps,
                               decay_rate=0.1)
        log_pis = self.actor.evaluate_log_pi(states, actions)
        # entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        # (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        # if self.learning_steps_ppo % self.epoch_ppo == 0:
        #     writer.add_scalar(
        #         'loss/actor', loss_actor.item(), self.learning_steps)
        #     writer.add_scalar(
        #         'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir):
        pass
