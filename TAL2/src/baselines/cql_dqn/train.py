import wandb
import torch
import pickle
import random
import argparse
import numpy as np
from collections import deque
from utils import save
from agent import CQLAgent
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.baselines.cql_dqn.buffer import ReplayBuffer


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--run_name', type=str, default='CQL-DQN',
                        help='Run name, default: CQL-DQN')
    parser.add_argument('--env', type=str, default='CartPole-v0',
                        help='Gym environment name, default: CartPole-v0')
    parser.add_argument('--episodes', type=int, default=300,
                        help='Number of episodes, default: 200')
    parser.add_argument('--buffer_size', type=int, default=100_000,
                        help='Maximal training data size, default: 100_000')
    parser.add_argument('--seed', type=int, default=0, help='Seed, default: 1')
    parser.add_argument('--min_eps', type=float, default=0.01, help='Minimal Epsilon, default: 4')
    parser.add_argument('--eps_frames', type=int, default=1e3,
                        help='Number of steps for annealing the epsilon value to the min epsilon.')
    parser.add_argument('--log_video', type=int, default=0,
                        help='Log agent behaviour to wanbd when set to 1, default: 0')
    parser.add_argument('--save_every', type=int, default=100,
                        help='Saves the network every x epochs, default: 25')

    args = parser.parse_args()
    return args


def train(config_current):
    np.random.seed(config_current.seed)
    random.seed(config_current.seed)
    torch.manual_seed(config_current.seed)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'APN'
    args.num_epochs = 200
    config = EnvironmentConfig(args)

    # env = gym.make(config.env)
    # env.seed(config.seed)
    # env.action_space.seed(config.seed)

    eps = 1.
    d_eps = 1 - config_current.min_eps
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    with open('checkpoints/home/action_list_train_dataset.pkl') as f:
        action_set = pickle.load(f)

    graphs_dir = 'new_dataset/home/'
    train_data_path = 'new_dataset/train_dataset.pkl'
    buffer = ReplayBuffer(config, graphs_dir, train_data_path)

    with wandb.init(project='CQL', name=config_current.run_name, config=config_current):
        agent = CQLAgent(config, action_set)
        wandb.watch(agent.network, log='gradients', log_freq=10)

        for i in range(1, config_current.episodes + 1):
            # state = env.reset()
            episode_steps = 0
            rewards = 0
            for iter_num, experiences in enumerate(buffer):
                (graphSeq, goal2vec, _, actionSeq, action2vec, _, _, rewards, dones) = experiences
                # action = agent.get_action(state, epsilon=eps)
                steps += 1
                # next_state, reward, done, _ = env.step(action[0])
                # buffer.add(state, action, reward, next_state, done)
                loss, cql_loss, bellmann_error = agent.learn(experiences)
                # state = next_state
                # rewards += reward
                episode_steps += 1
                eps = max(1 - ((steps * d_eps) / config_current.eps_frames),
                          config_current.min_eps)
                # if done:
                #     break

            average10.append(rewards)
            total_steps += episode_steps
            print('Episode: {} | Reward: {} | Q Loss: {} | Steps: {}'.format(i, rewards, loss,
                                                                             steps, ))

            wandb.log({'Reward': rewards,
                       'Average10': np.mean(average10),
                       'Total steps': total_steps,
                       'Q Loss': loss,
                       'CQL Loss': cql_loss,
                       'Bellmann error': bellmann_error,
                       'Steps': steps,
                       'Epsilon': eps,
                       'Episode': i,
                       'Buffer size': buffer.__len__()})

            if i % config_current.save_every == 0:
                save(config, args, wandb, 'CQL-DQN', agent.network, None, eps, None, ep=0)


if __name__ == '__main__':
    config_current = get_config()
    train(config_current)
