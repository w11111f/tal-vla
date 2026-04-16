"""
@Project     ：TAL_2024
@File        ：metadt_dataset.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/21
@Last        : 2024/12/2
@Description : 
"""
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from typing import Literal
from termcolor import cprint
from src.utils.buffer import ReplayBuffer


def discount_cumsum(x, gamma):
    discount_cumsum = torch.zeros((len(x)))
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(len(x) - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


class MetaDTDataset(ReplayBuffer):

    def __init__(
            self,
            config,
            context_encoder,
            graphs_dir,
            node_sequences_path,
            gamma=0.1,
            DATA_NUM=None,
            task_OBJ_VEC=False,
            DATA_ARGUMENT=False,
            reward_type: Literal["regression", "classification"] = 'regression',
            reward_codebook_path: str = './checkpoints/metadt/reward_codebook.pkl',
            return_scale=100,
    ):
        super().__init__(config, graphs_dir, node_sequences_path, gamma, DATA_NUM, task_OBJ_VEC,
                         DATA_ARGUMENT)
        self.config = config
        self.context_encoder = context_encoder
        self.graph_dir = graphs_dir
        self.node_sequences_path = node_sequences_path
        self.reward_type = reward_type
        self.reward_codebook_path = reward_codebook_path
        self.return_scale = return_scale
        self.horizon = config.context_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if reward_type == 'classification':
            self.load_reward_codebook()

        self.tasks = []
        self.contexts = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.rtgs = []
        self.timesteps = []
        self.masks = []

        for i in tqdm(range(super().__len__())):
            (states, task, _, _, actions, _, start_node, rewards, dones) = super().__getitem__(i)
            if self.reward_type == 'classification':
                rewards_tensor = self.embedding_rewards(rewards)
            else:
                rewards_tensor = rewards

            contexts = self.calculate_context(states, actions, rewards_tensor, task)

            for idx in range(1, len(states)):
                start_idx = max(0, idx - self.horizon)
                assert len(states[start_idx: idx]) <= self.horizon
                state_seg = states[start_idx: idx]
                action_seg = actions[start_idx: idx]
                reward_seg = rewards_tensor[start_idx: idx]
                context = torch.stack(contexts[start_idx: idx])
                # with torch.no_grad():
                #     context = context_encoder(state_seg, action_seg, reward_seg, task)

                tlen = len(state_seg)
                dtlen = self.horizon - tlen
                rtg_seg = discount_cumsum(rewards[start_idx:], gamma=1.)[:tlen].reshape(-1, 1)
                rtg_seg /= self.return_scale
                assert rtg_seg.shape[0] == len(state_seg), \
                    'rtg: {} != states: {}'.format(rtg_seg.shape[0], len(state_seg))
                timestep_seg = torch.arange(start_idx, start_idx + tlen).reshape(-1)
                mask_seg = torch.cat([torch.zeros(dtlen), torch.ones(tlen)], dim=0).unsqueeze(0)

                self.tasks.append(task)
                self.contexts.append(context.to(self.device))  # * [1, 64]
                self.states.append(state_seg)
                self.actions.append(torch.stack(action_seg).to(self.device))
                self.rewards.append(reward_seg)
                self.rtgs.append(rtg_seg.to(self.device))
                self.timesteps.append(timestep_seg.to(self.device))
                self.masks.append(mask_seg.to(self.device))

        cprint('[ContextDataset] data num: {}'.format(len(self.states)), 'green')

    def calculate_context(self, states, actions, rewards_tensor, task):
        contexts = []
        for idx in range(1, len(states)):
            start_idx = max(0, idx - self.horizon)
            assert len(states[start_idx: idx]) <= self.horizon
            state_seg = states[start_idx: idx]
            action_seg = torch.stack(actions[start_idx: idx])
            reward_seg = torch.cat(rewards_tensor[start_idx: idx])
            with torch.no_grad():
                context = self.context_encoder(state_seg, action_seg, reward_seg, task)
                contexts.append(context)
        return contexts

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
            self.tasks[index],
            self.contexts[index],
            self.states[index],
            self.actions[index],
            self.rewards[index],
            self.rtgs[index],
            self.timesteps[index],
            self.masks[index],

            # self.prompt_state[index],
            # self.prompt_action[index],
            # self.prompt_reward[index],
            # self.prompt_rtg[index],
            # self.prompt_tsp[index],
            # self.prompt_mask[index],
        )
