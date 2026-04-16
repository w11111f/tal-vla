import os
import json
import pickle
import torch
import numpy as np
from collections import OrderedDict

from src.baselines.metadt.envs.envs import AntDirEnv
from src.baselines.metadt.context.model import RNNContextEncoder, RewardDecoder
from src.baselines.metadt.context.dataset import ContextDataset
from src.baselines.metadt.meta_dt.model import MetaDecisionTransformer
from src.baselines.metadt.meta_dt.dataset import MetaDT_Dataset, append_context_to_data
from src.baselines.metadt.utils import convert_data_to_trajectories


def save_context_checkpoint(context_encoder, reward_decoder, checkpoint_path):
    torch.save(
        {
            'context_encoder': context_encoder.state_dict(),
            'reward_decoder': reward_decoder.state_dict(),
            # 'state_decoder': state_decoder.state_dict()
        },
        checkpoint_path
    )


def load_context_checkpoint(context_encoder, reward_model, checkpoint_path):
    context_encoder.load_state_dict(torch.load(checkpoint_path)['context_encoder'])
    # dynamic_decoder.load_state_dict(torch.load(load_path)['state_decoder'])
    reward_model.load_state_dict(torch.load(checkpoint_path)['reward_decoder'])
    print('[LOAD] context checkpoint from {}'.format(checkpoint_path))
    return context_encoder, reward_model


def load_model(args, state_dim, action_dim, device, context_checkpoint_path=None,
               train_context=False):
    meta_dt = MetaDecisionTransformer(
        state_dim=state_dim,
        act_dim=action_dim,
        max_length=args.dt_horizon,
        max_ep_len=args.max_episode_steps,
        context_dim=args.context_dim,
        hidden_size=args.dt_embed_dim,
        n_layer=args.dt_n_layer,
        n_head=args.dt_n_head,
        n_inner=4 * args.dt_embed_dim,
        activation_function=args.dt_activation_function,
        n_positions=1024,
        resid_pdrop=args.dt_dropout,
        attn_pdrop=args.dt_dropout,
    ).to(device) if not train_context else None

    context_dim = args.context_dim  # * 16
    hidden_dim = args.context_hidden_dim  # * 128
    context_encoder = RNNContextEncoder(state_dim, action_dim, context_dim, hidden_dim).to(device)
    reward_decoder = RewardDecoder(state_dim, action_dim, context_dim, hidden_dim).to(device)
    # state_decoder = StateDecoder(state_dim, action_dim, context_dim, hidden_dim).to(device)
    if context_checkpoint_path is None:
        d = f'./metadt_saves/{args.env_name}/context/{args.data_quality}/horizon{args.context_horizon}/'
        context_checkpoint_path = d + 'context_models_best.pt'
        if os.path.exists(context_checkpoint_path):
            try:
                load_context_checkpoint(context_encoder, reward_decoder, context_checkpoint_path)
                print('Load context encoder from {}'.format(context_checkpoint_path))
            except:
                print('Checkpoint size mismatch, fail to load.')

    if not train_context:
        for name, param in context_encoder.named_parameters():
            param.requires_grad = False
        for name, param in reward_decoder.named_parameters():
            param.requires_grad = False

    return meta_dt, context_encoder, reward_decoder


def create_env_AntDir(args):
    # * Make env, multi-task setting.
    with open(f'./metadt_datasets/{args.env_name}/{args.data_quality}/task_goals.pkl', 'rb') as file:
        velocities = pickle.load(file)
    env = AntDirEnv(tasks=velocities)
    # env.seed(args.seed)
    # env.action_space.seed(args.seed)
    # * Load the task information.
    with open(f'metadt_datasets/{args.env_name}/{args.data_quality}/task_info.json', 'r') as f:
        task_info = json.load(f)
    return env, task_info


def load_metadt_dataset(args, context_encoder, dynamic_decoder, device):
    prompt_trajectories_list = []
    for ind in range(args.num_tasks):
        file_path = f'metadt_datasets/{args.env_name}/{args.data_quality}/dataset_task_prompt{ind}.pkl'
        with open(file_path, "rb") as f:
            prompt_trajectories = pickle.load(f)
        prompt_trajectories_list.append(prompt_trajectories)

    train_trajectories = []
    for task_id in np.arange(args.num_train_tasks):
        train_data = OrderedDict()
        keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'masks']
        # keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'masks']
        for key in keys:
            train_data[key] = []
        file_path = f'metadt_datasets/{args.env_name}/{args.data_quality}/dataset_task_{task_id}.pkl'
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for key, values in data.items():
            train_data[key].append(values)
        for key, values in train_data.items():
            train_data[key] = np.concatenate(values, axis=0)
        train_data = append_context_to_data(train_data, context_encoder,
                                            horizon=args.context_horizon, device=device, args=args)
        train_trajectories_per = convert_data_to_trajectories(train_data, args)
        for trajectory in train_trajectories_per:
            train_trajectories.append(trajectory)

    dataset = MetaDT_Dataset(
        train_trajectories,
        args.dt_horizon,
        args.max_episode_steps,
        args.dt_return_scale,
        device,
        prompt_trajectories_list=prompt_trajectories_list,
        args=args,
        world_model=[context_encoder, dynamic_decoder]
    )
    return dataset


def load_context_dataset(args, device):
    # * Preparing the training and testing datasets.
    train_data, test_data = OrderedDict(), OrderedDict()
    keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'masks']
    for key in keys:
        train_data[key] = []
        test_data[key] = []
    data_path = f'./metadt_datasets/{args.env_name}/{args.data_quality}'
    for task_id in range(args.num_tasks):
        with open(f'{data_path}/dataset_task_{task_id}.pkl', "rb") as f:
            data = pickle.load(f)
        for key, values in data.items():
            if task_id < args.num_train_tasks:
                train_data[key].append(values)
            else:
                test_data[key].append(values)
    for key, values in train_data.items():
        train_data[key] = np.concatenate(train_data[key], axis=0)
        test_data[key] = np.concatenate(test_data[key], axis=0)

    train_dataset = ContextDataset(train_data, horizon=args.context_horizon, device=device)
    test_dataset = ContextDataset(test_data, horizon=args.context_horizon, device=device)
    return train_dataset, test_dataset
