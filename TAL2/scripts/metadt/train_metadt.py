"""
@Project     ：TAL_2024
@File        ：train_metadt.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/21
@Last        : 2024/11/21
@Description : 
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import cprint

from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig

from src.baselines.metadt.utils_model import load_context_checkpoint
from src.baselines.metadt.utils_new.metadt_dataset import MetaDTDataset
from src.baselines.metadt.utils_new.metadt_model import MetaDecisionTransformer
from src.baselines.metadt.utils_new.utils import load_context_model, save_metadt_checkpoint


def main():
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    cprint('Set random seed = {}'.format(rnd_seed), 'green')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)
    args.num_epochs = 500
    config.context_len = 3
    reward_type = 'classification'
    save_model_path = f'checkpoints/metadt/metadt/dataset_1/'
    os.makedirs(save_model_path, exist_ok=True)

    # * ---------------------------------------------------------------
    # * Model.
    context_checkpoint_path = './checkpoints/metadt/context/context_models_dataset_1.pt'
    context_encoder, reward_decoder = load_context_model(config, device, reward_type, False)
    context_encoder, reward_decoder = load_context_checkpoint(context_encoder, reward_decoder,
                                                              context_checkpoint_path)
    metadt = MetaDecisionTransformer(config, device).to(device).train()
    optimizer = optim.AdamW(metadt.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()

    # * ---------------------------------------------------------------
    # * Dataset.
    print('Dataset: 1')
    graphs_dir = 'data/1/home/'
    train_data_path = 'data/1/train_dataset.pkl'
    train_dataset = MetaDTDataset(
        config, context_encoder, graphs_dir, train_data_path, reward_type=reward_type
    )

    # * ---------------------------------------------------------------
    # * Training.
    act_dim = 111
    action_name_dim = 11
    object_dim = 36
    state_dim = 28
    action_cursor = 0
    object_cursor_1 = action_name_dim  # * 11 -> 11+36
    object_cursor_2 = action_name_dim + object_dim  # * 11+36 -> 11+36+36
    state_cursor = action_name_dim + object_dim * 2  # * 11+36+36 -> 11+36+36+28
    save_freq = 10
    optimizer.zero_grad()
    batch_size = 32
    for epoch in range(args.num_epochs):
        # * Model training.
        total_loss = 0
        for step, data in enumerate(train_dataset):
            task, contexts, states, actions, rewards, rtg, timesteps, mask = data
            action_name, object_1, object_2, state = metadt(
                task, contexts, states, actions, rtg, timesteps, mask, None
            )
            action_name = action_name.reshape(-1, action_name_dim)[mask.reshape(-1) > 0]
            object_1 = object_1.reshape(-1, object_dim)[mask.reshape(-1) > 0]
            object_2 = object_2.reshape(-1, object_dim)[mask.reshape(-1) > 0]
            state = state.reshape(-1, state_dim)[mask.reshape(-1) > 0]

            action_name_target = actions[:, action_cursor:object_cursor_1]
            object_1_target = actions[:, object_cursor_1:object_cursor_2]
            object_2_target = actions[:, object_cursor_2:state_cursor]
            state_target = actions[:, state_cursor:]

            loss = criterion(action_name, action_name_target) \
                   + criterion(object_1, object_1_target) \
                   + criterion(object_2, object_2_target) \
                   + criterion(state, state_target)

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
            model_name = f'{save_model_path}/metadt_models_{epoch + 1}.pt'
            save_metadt_checkpoint(metadt, model_name)


if __name__ == '__main__':
    main()
