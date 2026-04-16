import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.baselines.metadt.configs import args_ant_dir
from src.baselines.metadt.utils import discount_cumsum, setup_seed
from src.baselines.metadt.utils_model import load_model, create_env_AntDir, load_metadt_dataset
from src.baselines.metadt.meta_dt.trainer import MetaDT_Trainer
from src.baselines.metadt.meta_dt.dataset import append_error_to_trajectory
from src.baselines.metadt.meta_dt.evaluation import meta_evaluate_episode_rtg


def eval_task(args, env, global_step, task_ids, task_info, trajectories_buffer, world_model, model,
              context_encoder, context_horizon, state_mean, state_std, state_dim, action_dim,
              max_len, max_ep_len, scale, device):
    model.eval()
    avg_epi_return = 0.0
    avg_max_return_offline = 0.0
    for task_id in task_ids:
        env.reset_task(task_id)
        target_ret = task_info[f'task {task_id}']['return_scale'][1]
        if (global_step <= args.warm_train) or (args.zero_shot):
            prompt = None
        else:
            total_rewards = [sum(traj['rewards']) for traj in trajectories_buffer[task_id]]
            top_indices = sorted(range(len(total_rewards)), key=lambda i: total_rewards[i],
                                 reverse=True)[:3]
            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
            traj = [trajectories_buffer[task_id][i] for i in top_indices]
            traj = random.choice(traj)
            traj = append_error_to_trajectory(world_model, device, context_horizon, traj, args,
                                              state_mean, state_std)
            indices = np.arange(context_horizon, args.max_ep_len - max_len + 1)
            world_model_error = [traj['errors'][sj: sj + args.max_ep_len].sum() for sj in indices]
            error_probs = np.array(world_model_error) / np.sum(world_model_error)
            selected_index = np.random.choice(indices, p=error_probs)
            si = selected_index

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, action_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(
                traj['rewards'][si:], gamma=1.
            )[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)
            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            mid_dim = max_len - tlen
            s[-1] = np.concatenate([np.zeros((1, mid_dim, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, mid_dim, action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, mid_dim, 1)), r[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, mid_dim, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, mid_dim)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, mid_dim)), np.ones((1, tlen))], axis=1))
            s = torch.from_numpy(np.concatenate(s, axis=0)).to(device).float()
            a = torch.from_numpy(np.concatenate(a, axis=0)).to(device).float()
            r = torch.from_numpy(np.concatenate(r, axis=0)).to(device).float()

            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(device).float()
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(device).long()
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device)
            rtg = rtg[:, :-1, :]
            rtg = rtg.reshape((1, -1, rtg.shape[-1]))
            prompt = s, a, r, rtg, timesteps, mask

        epi_return, epi_length, traj_per = meta_evaluate_episode_rtg(
            env,
            state_dim,
            action_dim,
            model,
            context_encoder,
            max_episode_steps=args.max_episode_steps,
            scale=args.dt_return_scale,
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            target_return=target_ret / args.dt_return_scale,
            horizon=args.context_horizon,
            num_eval_episodes=args.num_eval_episodes,
            prompt=prompt,
            args=args,
            epoch=global_step,
        )
        trajectories_buffer[task_id].append(traj_per)
        avg_epi_return += epi_return
        avg_max_return_offline += target_ret
        print('Evaluate on the {}-th task, target return {:.2f}, received return {:.2f}'.format(
            task_id, target_ret, epi_return))
    avg_epi_return /= len(task_ids)
    avg_max_return_offline /= len(task_ids)
    return avg_epi_return, avg_max_return_offline


def main():
    args = args_ant_dir.get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = f'runs/{args.env_name}/{args.zero_shot}/{args.data_quality}/{args.seed}/horizon{args.context_horizon}'
    os.makedirs(results_dir, exist_ok=True)
    setup_seed(args.seed)
    np.set_printoptions(precision=3, suppress=True)
    writer = SummaryWriter(results_dir)

    # * Env and task.
    env, task_info = create_env_AntDir(args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # * Models.
    meta_dt, context_encoder, dynamic_decoder = load_model(args, state_dim, action_dim, device)
    world_model = [context_encoder, dynamic_decoder]
    optimizer = optim.AdamW(meta_dt.parameters(), lr=args.dt_lr, weight_decay=args.dt_weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / args.meta_dt_warmup_steps, 1)
    )
    agent = MetaDT_Trainer(meta_dt, optimizer)

    # * Dataset.
    eval_train_task_ids = np.arange(5)
    test_task_ids = np.arange(args.num_train_tasks, args.num_tasks)
    train_dataset = load_metadt_dataset(args, context_encoder, dynamic_decoder, device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.meta_dt_batch_size, shuffle=True)
    state_mean, state_std = train_dataset.state_mean, train_dataset.state_std

    global_step = 0
    max_len = args.prompt_length
    max_ep_len = args.max_ep_len
    scale = args.scale
    trajectories_buffer = [[] for _ in range(args.num_tasks)]
    while global_step <= args.max_step:
        print(f'\n==========  {global_step} ==========')
        for step, batch in tqdm(enumerate(train_dataloader)):
            (states, contexts, actions, rewards, dones, rtg, timesteps, masks, prompt_states,
             prompt_actions, prompt_rewards, prompt_returns_to_go, prompt_timesteps,
             prompt_attention_mask) = batch
            prompt_returns_to_go = prompt_returns_to_go[:, :-1, :]
            prompts = (prompt_states, prompt_actions, prompt_rewards, prompt_returns_to_go,
                       prompt_timesteps, prompt_attention_mask)
            if (global_step <= args.warm_train) or (args.zero_shot):
                train_loss = agent.train_step(states, contexts, actions, rewards, dones, rtg,
                                              timesteps, masks, None)
                scheduler.step()
            else:
                train_loss = agent.train_step(states, contexts, actions, rewards, dones, rtg,
                                              timesteps, masks, prompts)
                scheduler.step()
            global_step += 1
            writer.add_scalar('train/loss', train_loss, global_step)

            if global_step % args.eval_step == 0:
                # * Evaluate on five tranining tasks.
                print(f'\nIterations {global_step} Evaluate on 5 training tasks...')
                avg_epi_return, avg_max_return_offline = eval_task(
                    args, env, global_step, eval_train_task_ids, task_info, trajectories_buffer,
                    world_model, meta_dt, context_encoder, args.context_horizon, state_mean,
                    state_std, state_dim, action_dim, max_len, max_ep_len, scale, device
                )
                writer.add_scalars(f'return/train tasks',
                                   {'MetaDT': avg_epi_return, 'Oracle': avg_max_return_offline},
                                   global_step)
                print(f'\nAverage received return {avg_epi_return:.2f}, '
                      f'average max return {avg_max_return_offline:.2f}')

                # * Evaluate on five test tasks.
                print(f'\nIterations {global_step} Evaluate on 5 test tasks...')
                avg_epi_return, avg_max_return_offline = eval_task(
                    args, env, global_step, test_task_ids, task_info, trajectories_buffer,
                    world_model, meta_dt, context_encoder, args.context_horizon, state_mean,
                    state_std, state_dim, action_dim, max_len, max_ep_len, scale, device
                )
                writer.add_scalars(f'return/test tasks',
                                   {'MetaDT': avg_epi_return, 'Oracle': avg_max_return_offline},
                                   global_step)
                print(f'\nAverage received return {avg_epi_return:.2f}, '
                      f'average max return {avg_max_return_offline:.2f}')


if __name__ == '__main__':
    main()
