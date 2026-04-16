import torch
import random
import numpy as np
from collections import OrderedDict


def setup_seed(seed, torch_deterministic=True):
    """Setup seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def discount_cumsum(x, gamma):
    """
    https://github.com/NJU-RL/Meta-DT/blob/main/decision_transformer/dataset.py
    """
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def convert_data_to_trajectories(data, args):
    """
    https://github.com/NJU-RL/Meta-DT/blob/main/decision_transformer/dataset.py
    """
    trajectories = []
    start_ind = 0
    if args.env_name == 'PointRobot-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'HalfCheetahVel-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'HalfCheetahDir-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'AntDir-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'WalkerRandParams-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'HopperRandParams-v0':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break
    elif args.env_name == 'Reach':
        for ind, terminal in enumerate(data['terminals']):
            if (ind + 1) % args.max_episode_steps == 0:
                traj = OrderedDict()
                for key, value in data.items():
                    traj[key] = value[start_ind: ind + 1]
                trajectories.append(traj)
                start_ind = ind + 1
            if len(trajectories) == args.max_train_eposides:
                break

    # print(f'Convert {ind} transitions to {(len(trajectories))} trajectories.')
    return trajectories
