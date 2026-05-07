
import os
import argparse
import sys
import time
from datetime import datetime
import torch
import random
from src.env import CF_RS_HI
from algo.ppo import PPO
from src.trainer import Trainer
from src.config import *
import scipy.io as scio
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def run(args):

    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = CF_RS_HI()
    env_test = CF_RS_HI()

    algo = PPO(
        state_shape=env.observation_shape,
        action_shape=env.action_shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        # lr_actor=1e-4,
        # lr_critic=1e-4
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     'logs', args.env_id, 'ppo', f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        # log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    rewards, SE, order, ps = trainer.train()
    val_max_reward = np.max(rewards)
    # idx_max_reward = np.argmax(rewards, axis=0)
    # order = order[idx_max_reward]
    # print(order)
    # print(val_max_reward)

    path = "/result" + "PPO_vae_ppo" + str(args.num_steps/(10**6)) + "x10^6"
    if not os.path.isdir(os.getcwd() + path):
        os.makedirs(os.getcwd() + path)
    s = ""
    reward_path = os.getcwd() + path + "\\reward_" + str(time) + s +".mat"
    scio.savemat(reward_path, {'reward': rewards})
    # rate_u_path = os.getcwd() + path + "\\rate_u_" + str(time) + s + ".mat"
    # scio.savemat(rate_u_path, {'rate_u': rate_u})
    # rate_e_path = os.getcwd() + path + "\\rate_e_" + str(time) + s + ".mat"
    # scio.savemat(rate_e_path, {'rate_u': rate_e})
    # order_path = os.getcwd() + path + "\\order_" + str(time) + s + ".mat"
    # scio.savemat(order_path, {'order': order})

    return val_max_reward
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--num_steps', type=int, default=50 * 10**4)
    p.add_argument('--eval_interval', type=int, default=10 ** 4)
    p.add_argument('--env_id', type=str, default='IRS')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=random_seed)
    args = p.parse_args()
    # log_print = open('result/log/' + str(Pmax) + '_' + str(datetime.now().strftime("%Y%m%d-%H%M")) + '.txt', 'w')
    # sys.stdout = log_print
    # sys.stderr = log_print
    print(time.asctime(time.localtime(time.time())))
    val_max_reward= run(args)
    print(time.asctime(time.localtime(time.time())))
    print(val_max_reward)
