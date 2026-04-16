"""
@Project     ：TAL_2024
@File        ：train_tdmpc2.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/9
@Last        : 2024/11/9
@Description : 
"""
import os

import torch
import hydra
from termcolor import cprint
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.utils.buffer import ReplayBuffer
from src.baselines.tdmpc2.tdmpc2 import TDMPC2
from src.baselines.tdmpc2.common.parser import parse_cfg

global agent_cfg


@hydra.main(version_base=None, config_name='tdmpc2_config.yaml', config_path='.')
def get_agent_cfg(cfg: dict):
    cfg = parse_cfg(cfg)
    # cfg.multitask = True
    # cfg.tasks = 512
    cfg.obs_shape = {'state': (1024,)}  # * obs(512) + task(512)
    cfg.latent_dim = 512
    cfg.action_dim = 111
    cfg.task_dim = 1024
    cfg.episode_length = 20
    cfg.seed_steps = 1000
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    cfg.action_name_dim = 11
    cfg.action_obj_dim = 36
    cfg.action_state_dim = 28
    global agent_cfg
    agent_cfg = cfg


def main():
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    cprint('Set random seed = {}'.format(rnd_seed), 'green')

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_epochs = 350
    config = EnvironmentConfig(args)

    global agent_cfg
    get_agent_cfg()
    agent = TDMPC2(agent_cfg, config=config)

    graphs_dir = 'data/2/home/'
    train_data_path = 'data/2/train_dataset.pkl'
    buffer = ReplayBuffer(config, graphs_dir, train_data_path)

    checkpoint_dir = 'checkpoint/tdmpc2/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_freq = 10
    start_epoch = 0
    update_model, update_pi = True, True
    for epoch_num in range(start_epoch, config.NUM_EPOCHS):
        total_loss = 0
        consistency_loss = 0
        reward_loss = 0
        value_loss = 0
        pi_loss = 0
        # if epoch_num % 3 == 0:
        #     update_model, update_pi = True, False
        # else:
        #     update_model, update_pi = False, True
        for i in range(len(buffer)):
            data = buffer.sample(i)
            train_metrics = agent.update(data, update_model=update_model, update_pi=update_pi)
            total_loss += train_metrics['total_loss']
            consistency_loss += train_metrics['consistency_loss']
            reward_loss += train_metrics['reward_loss']
            value_loss += train_metrics['value_loss']
            pi_loss += train_metrics['pi_loss']
        # report_str = 'Epoch {}: total_loss: {} consistency_loss: {} reward_loss: {} value_loss: {}'
        # print(report_str.format(epoch_num, total_loss, consistency_loss, reward_loss, value_loss))
        # print('Epoch {}: pi_loss: {}'.format(epoch_num, pi_loss))
        print('Epoch {}: total_loss: {} | pi_loss: {}'.format(epoch_num, total_loss, pi_loss))

        if epoch_num % save_freq == 0:
            save_path = os.path.join(checkpoint_dir, 'tdmpcc2_epoch_{}.pth'.format(epoch_num))
            agent.save(save_path)


if __name__ == '__main__':
    main()
