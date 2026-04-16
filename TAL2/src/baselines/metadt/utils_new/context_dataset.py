"""
@Project     ：TAL_2024
@File        ：context_dataset.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/20
@Last        : 2024/11/20
@Description : 
"""
import os
import torch
import pickle
from tqdm import tqdm
from typing import Literal
from termcolor import cprint
from src.utils.buffer import ReplayBuffer


class ContextDataset(ReplayBuffer):

    def __init__(
            self,
            config,
            graphs_dir,
            node_sequences_path,
            gamma=0.1,
            DATA_NUM=None,
            task_OBJ_VEC=False,
            DATA_ARGUMENT=False,
            reward_type: Literal["regression", "classification"] = 'regression',
            reward_codebook_path: str = './checkpoints/metadt/reward_codebook.pkl'
    ):
        """
        Params:
            - reward_type:
                - 'regression' -> 0, 1, 100
                - 'classification' -> [1, 0, 0], [0, 1, 0], [0, 0, 1]
        """
        super().__init__(config, graphs_dir, node_sequences_path, gamma, DATA_NUM, task_OBJ_VEC,
                         DATA_ARGUMENT)
        self.reward_type = reward_type
        self.reward_codebook_path = reward_codebook_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if reward_type == 'classification':
            self.load_reward_codebook()

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.terminals = []
        self.tasks = []

        self.states_segment = []
        self.actions_segment = []
        self.rewards_segment = []

        self.next_states_segment = []
        self.next_actions_segment = []
        self.next_rewards_segment = []

        for i in tqdm(range(super().__len__())):
            (states, task, _, _, actions, _, start_node, rewards, dones) = super().__getitem__(i)
            if self.reward_type == 'classification':
                rewards = self.embedding_rewards(rewards)

            for idx in range(1, len(states) - 1):
                self.states.append(states[idx])
                self.actions.append(actions[idx])
                self.rewards.append(rewards[idx])
                self.next_states.append(states[idx + 1])
                self.terminals.append(dones[idx])
                self.tasks.append(task)

                start_idx = max(0, idx - config.context_len)
                assert len(states[start_idx: idx]) <= config.context_len
                self.states_segment.append(states[start_idx: idx])
                self.actions_segment.append(torch.stack(actions[start_idx: idx]))
                self.rewards_segment.append(torch.cat(rewards[start_idx: idx]))
                self.next_states_segment.append(states[start_idx + 1: idx + 1])
                self.next_actions_segment.append(actions[start_idx + 1: idx + 1])
                self.next_rewards_segment.append(torch.stack(rewards[start_idx + 1: idx + 1]))
        cprint('[ContextDataset] data num: {}'.format(len(self.states)), 'green')

    def load_reward_codebook(self):
        if os.path.exists(self.reward_codebook_path):
            with open(self.reward_codebook_path, 'rb') as f:
                self.reward_codebook = pickle.load(f)
            print('Reward codebook loaded from {}.'.format(self.reward_codebook_path))
        else:
            os.makedirs('./checkpoints/metadt/', exist_ok=True)
            self.reward_codebook = self.calculate_reward_value_num()
            with open(self.reward_codebook_path, 'wb') as f:
                pickle.dump(self.reward_codebook, f)
            print('Reward codebook calculation completed.')
        print('Reward codebook: {}'.format(self.reward_codebook))

    def calculate_reward_value_num(self):
        """Count the number of reward values."""
        reward_codebook = []
        for i in range(super().__len__()):
            (_, _, _, _, _, _, _, rewards, _) = super().__getitem__(i)
            for r in rewards:
                if r not in reward_codebook:
                    reward_codebook.append(r)
        reward_codebook.sort()
        return reward_codebook

    def embedding_rewards(self, rewards):
        """Reward embedding."""
        rewards_embed = []
        for r in rewards:
            idx = self.reward_codebook.index(r)
            r_embed = torch.zeros((1, len(self.reward_codebook))).to(self.device)
            r_embed[0, idx] = 1
            rewards_embed.append(r_embed)
        return rewards_embed

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return (
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.next_states[index],
            self.tasks[index]
        ), (
            self.states_segment[index],
            self.actions_segment[index],
            self.rewards_segment[index],
        ), (
            self.next_states_segment[index],
            self.next_actions_segment[index],
            self.next_rewards_segment[index],
        )
