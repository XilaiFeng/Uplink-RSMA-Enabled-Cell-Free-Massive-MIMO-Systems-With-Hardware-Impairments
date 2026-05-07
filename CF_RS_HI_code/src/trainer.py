import os
from time import time, sleep
from datetime import timedelta

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from src.config import *
import time

class Trainer:
    def __init__(self, env, env_test, algo, log_dir=None, seed=0, num_steps=5*10**3,
                 eval_interval=10**3, num_eval_episodes=5):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        # self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        # self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = None
        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        # self.model_dir = os.path.join(log_dir, 'model')
        # if not os.path.exists(self.model_dir):
        #     os.makedirs(self.model_dir)


        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

    def train(self):
        # Time to start training.
        # self.start_time = time()
        # Episode's timestep.
        t = 0
        rewards = []
        orders = []
        ps = []
        SSEs = []
        # Initialize the environment.
        state = self.env.reset()
        start_time = time.time()

        for step in range(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            # l = min(len(EHs), 1000)
            # EHs_temp = EHs[-l:]
            state, reward, SINR, order = self.algo.step(self.env, state, t, step)
            SSE = np.sum(np.real(np.log2(1 + SINR)))
            rewards.append(reward)
            SSEs.append(SSE)
            # EHs.append(np.mean(EH))
            orders.append(order)
            ps.append(order)
            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)

            num = 1000

            if step % num == 0:
                end_time = time.time()
                print('eposide:', step / 10,
                      '    reward: %.3g' % np.mean(rewards[-num:]),
                      '    SSE: %.3g' % np.mean(SSEs[-num:]),
                      '    Time:', end_time - start_time)
                start_time = end_time


            # # Evaluate regularly.
            # if step % self.eval_interval == 0:
            #     self.evaluate(step)
        # self.algo.save_models(
        #     os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        # sleep(10)

        return rewards, SSEs, orders, ps

    def evaluate(self, step):
        mean_return = 0.0

        for episode in range(self.num_eval_episodes):
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            while (not done):
                action = self.algo.exploit(state)
                state, reward, done = self.env_test.step(action, episode%10)
                episode_return += reward

            mean_return += episode_return / self.num_eval_episodes

        self.writer.add_scalar('return/test', mean_return, step)
        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
