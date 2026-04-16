"""
@Project     ：TAL_2024
@File        ：context_dataset_tal.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/20
@Last        : 2024/11/20
@Description : 
"""
import os
import torch
import torch.nn as nn
from termcolor import cprint

from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.baselines.metadt.utils_model import save_context_checkpoint
from src.baselines.metadt.utils_new.context_dataset import ContextDataset
from src.baselines.metadt.utils_new.utils import load_context_model


def main():
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    cprint('Set random seed = {}'.format(rnd_seed), 'green')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)
    args.num_epochs = 150
    config.context_len = 3
    reward_type = 'classification'

    # * ---------------------------------------------------------------
    # * Dataset.
    graphs_dir = 'data/1/home/'
    train_data_path = 'data/1/train_dataset.pkl'
    train_dataset = ContextDataset(config, graphs_dir, train_data_path, reward_type=reward_type)

    # * ---------------------------------------------------------------
    # * Model.
    context_encoder, reward_decoder = load_context_model(config, device, reward_type)

    save_model_path = f'checkpoints/metadt/context/'
    os.makedirs(save_model_path, exist_ok=True)
    optimizer = torch.optim.Adam(
        [*context_encoder.parameters(), *reward_decoder.parameters()], lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.65)
    if reward_type == 'regression':
        criterion = nn.MSELoss()
    elif reward_type == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    # * ---------------------------------------------------------------
    # * Training.
    best_loss = float('inf')
    save_freq = 10
    optimizer.zero_grad()
    batch_size = 32
    for epoch in range(args.num_epochs):
        # * Model training.
        # batch_size = 1 if epoch < 200 else 4
        total_loss = 0
        for step, (transition, segment, next_segment) in enumerate(train_dataset):
            state, action, reward, next_state, task = transition
            state_segment, action_segment, reward_segment = segment
            context = context_encoder(state_segment, action_segment, reward_segment, task)
            reward_predict = reward_decoder(state, action, next_state, task, context)
            loss = criterion(reward_predict, reward)
            total_loss += loss.item()
            # * Backward.
            loss /= batch_size
            loss.backward()
            if step % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]['lr']
        print('Epoch {} lr {:.5f} total loss: {}'.format(epoch, current_lr, total_loss))
        scheduler.step()

        if (epoch + 1) % save_freq == 0:
            model_name = f'{save_model_path}/context_models_{epoch + 1}.pt'
            save_context_checkpoint(context_encoder, reward_decoder, model_name)


if __name__ == '__main__':
    main()
