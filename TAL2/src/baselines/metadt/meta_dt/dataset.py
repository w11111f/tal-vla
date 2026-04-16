import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def append_context_to_data(data, context_encoder, horizon=4, device='cpu', args=None):
    context_encoder.eval()
    context_encoder = context_encoder.to(device)

    states = data['observations']
    actions = data['actions']
    rewards = data['rewards'].reshape(-1, 1)
    terminals = data['terminals'].reshape(-1, 1)
    next_states = data['next_observations']

    states_segment, actions_segment, rewards_segment, next_states_segment = [], [], [], []

    initial_state_idx = 0
    for idx in range(states.shape[0]):
        start_idx = max(0, idx - horizon, initial_state_idx)

        if initial_state_idx == idx:
            state_seg = np.zeros((horizon, states.shape[1]))
            action_seg = np.zeros((horizon, actions.shape[1]))
            reward_seg = np.zeros((horizon, rewards.shape[1]))
            next_states_seg = np.zeros((horizon, next_states.shape[1]))
        else:
            state_seg = states[start_idx: idx]
            action_seg = actions[start_idx: idx]
            reward_seg = rewards[start_idx: idx]
            next_states_seg = next_states[start_idx: idx]

        length_gap = horizon - state_seg.shape[0]
        states_segment.append(np.pad(state_seg, ((length_gap, 0), (0, 0))))
        actions_segment.append(np.pad(action_seg, ((length_gap, 0), (0, 0))))
        rewards_segment.append(np.pad(reward_seg, ((length_gap, 0), (0, 0))))
        next_states_segment.append(np.pad(next_states_seg, ((length_gap, 0), (0, 0))))

        if terminals[idx]:
            initial_state_idx = idx + 1

    # size: (num_samples, seq_len, dim)
    states_segment = torch.from_numpy(np.stack(states_segment, axis=0)).float().to(device)
    actions_segment = torch.from_numpy(np.stack(actions_segment, axis=0)).float().to(device)
    rewards_segment = torch.from_numpy(np.stack(rewards_segment, axis=0)).float().to(device)
    next_states_segment = torch.from_numpy(np.stack(next_states_segment, axis=0)).float().to(
        device)

    with torch.no_grad():
        contexts = context_encoder(states_segment.transpose(0, 1), actions_segment.transpose(0, 1),
                                   rewards_segment.transpose(0, 1))

    data['contexts'] = contexts.detach().cpu().numpy()

    return data


def append_error_to_trajectory(world_model,device,context_horizon, traj,args,mean,std):
        if args.env_name=='ML10':
            context_encoder, state_decoder,reward_decoder = world_model
            context_encoder.eval();state_decoder.eval(),reward_decoder.eval()
            context_encoder.to(device),state_decoder.to(device),reward_decoder.to(device)
        else:

            context_encoder, dynamics_decoder = world_model
            context_encoder.eval(); dynamics_decoder.eval()
            context_encoder.to(device); dynamics_decoder.to(device)
    
        states = traj['observations']
        actions = traj['actions']
        rewards = traj['rewards'].reshape(-1, 1)
        next_states = traj['next_observations']

        states_segment, actions_segment, rewards_segment,next_states_segment = [], [], [],[]
        for ind in range(states.shape[0]):
            start_ind = max(0, ind-context_horizon)

            if ind == 0:
                state_seg = np.zeros((context_horizon, states.shape[1]))
                action_seg = np.zeros((context_horizon, actions.shape[1]))
                reward_seg = np.zeros((context_horizon, rewards.shape[1]))
                next_state_seg = np.zeros((context_horizon, next_states.shape[1]))

            else:
                state_seg = states[start_ind : ind]
                action_seg = actions[start_ind : ind]
                reward_seg = rewards[start_ind : ind]
                next_state_seg = next_states[start_ind : ind]

            tlen = state_seg.shape[0]
            state_seg = np.concatenate([np.zeros((context_horizon-tlen, state_seg.shape[1])), state_seg], axis=0)
            action_seg = np.concatenate([np.zeros((context_horizon-tlen, action_seg.shape[1])), action_seg], axis=0)
            reward_seg = np.concatenate([np.zeros((context_horizon-tlen, reward_seg.shape[1])), reward_seg], axis=0)
            next_state_seg = np.concatenate([np.zeros((context_horizon-tlen, next_state_seg.shape[1])), next_state_seg], axis=0)
            states_segment.append(state_seg)
            actions_segment.append(action_seg)
            rewards_segment.append(reward_seg)
            next_states_segment.append(next_state_seg)

        # size: (seq_len, context_horizon, dim)
        states_segment = torch.from_numpy(np.stack(states_segment, axis=0)).float().to(device)
        actions_segment = torch.from_numpy(np.stack(actions_segment, axis=0)).float().to(device)
        rewards_segment = torch.from_numpy(np.stack(rewards_segment, axis=0)).float().to(device)
        next_states_segment = torch.from_numpy(np.stack(next_states_segment, axis=0)).float().to(device)
        # size: (seq_len, dim)
        states = torch.from_numpy(traj['observations']).float().to(device)
        actions = torch.from_numpy(traj['actions']).float().to(device)
        next_states = torch.from_numpy(traj['next_observations']).float().to(device)
        rewards = torch.from_numpy(traj['rewards'].reshape(-1,1)).float().to(device)
        with torch.no_grad():
            if( (args.env_name=='WalkerRandParams-v0')or(args.env_name == 'HopperRandParams-v0')):
                # contexts = context_encoder(states_segment.transpose(0,1), actions_segment.transpose(0,1), rewards_segment.transpose(0,1),next_states_segment.transpose(0,1))
                contexts = context_encoder(states_segment.transpose(0,1), actions_segment.transpose(0,1), rewards_segment.transpose(0,1))
                states_predict = dynamics_decoder(states, actions, rewards,next_states, contexts).detach().cpu().numpy()
                
                traj['errors'] = abs(states_predict - ((traj['next_observations']-mean)/std))
            elif args.env_name=='ML10':
                contexts = context_encoder(states_segment.transpose(0,1), actions_segment.transpose(0,1), rewards_segment.transpose(0,1))
                states_predict = state_decoder(states, actions, rewards,next_states, contexts).detach().cpu().numpy()
                reward_predict = reward_decoder(states, actions, next_states, contexts).detach().cpu().numpy()
                traj['errors'] = abs(reward_predict - traj['rewards'].reshape(-1,1)) +abs(states_predict - traj['next_observations'])

            else:
                contexts = context_encoder(states_segment.transpose(0,1), actions_segment.transpose(0,1), rewards_segment.transpose(0,1))
                reward_predict = dynamics_decoder(states, actions, next_states, contexts).detach().cpu().numpy()
                traj['errors'] = abs(reward_predict - traj['rewards'].reshape(-1,1))

        return traj


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


class MetaDT_Dataset(Dataset):
    def __init__(self, trajectories, horizon, max_episode_steps, return_scale, device,prompt_trajectories_list,args,world_model):

        self.trajectories = trajectories
        self.horizon = horizon 
        self.max_episode_steps = max_episode_steps
        self.return_scale = return_scale
        self.device = device 
        self.prompt_trajectories_list =prompt_trajectories_list
        self.args=args
        self.world_model = world_model
        self.context_horizon = args.context_horizon

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(self.state_mean)
        print(self.state_std)

        num_timesteps = sum(traj_lens)

        self.return_min = np.min(returns)
        self.return_max = np.max(returns)
        self.return_avg = np.average(returns)
        print(f'Dataset info: {len(trajectories)} trajectories, {num_timesteps} transitions, returns [{returns.min()}, {returns.max()}]')

        print('Preparing the training data for MetaDT...')
        self.parse_trajectory_segment()
        print(f'Size of training data: {self.states.size(0)}')

    
    def parse_trajectory_segment(self):
        states, actions, rewards, dones, rtg, timesteps, masks, contexts = [], [], [], [], [], [], [], []
        prompt_state,prompt_action,prompt_rewards,prompt_dones,prompt_rtg,prompt_timesteps,prompt_masks= [], [], [], [], [], [], []
        state_dim = self.args.state_dim
        act_dim = self.args.act_dim
        max_len =self.args.prompt_length
        max_ep_len,scale =self.args.max_ep_len,self.args.scale
        state_mean,state_std = self.state_mean,self.state_std
        device = self.args.device
        print(f'Segmenting a total of {len(self.trajectories)} trajectories...')
        for num, traj in tqdm(enumerate(self.trajectories)):
            ids = num // self.args.total_epi
            prompt_trajectories_list = self.prompt_trajectories_list[ids]
            p_i=random.randint(0, 2)
            promt_traj = prompt_trajectories_list[p_i]
            promt_traj = append_error_to_trajectory(self.world_model,self.device,self.context_horizon,promt_traj,self.args,self.state_mean,self.state_std)
            indices = np.arange(self.context_horizon, max_ep_len- max_len + 1)
            world_model_error = [promt_traj['errors'][sj : sj+max_len].sum() for sj in indices]
            error_probs = np.array(world_model_error) / np.sum(world_model_error)
            selected_index = np.random.choice(indices, p=error_probs)
            p_start = selected_index

            for si in range(traj['rewards'].shape[0] - 1):
                prompt_state.append(promt_traj['observations'][p_start:p_start + max_len].reshape(1, -1, state_dim))
                prompt_action.append(promt_traj['actions'][p_start:p_start + max_len].reshape(1, -1, act_dim))
                prompt_rewards.append(promt_traj['rewards'][p_start:p_start + max_len].reshape(1, -1, 1))
                prompt_timesteps.append(np.arange(p_start, p_start + prompt_state[-1].shape[1]).reshape(1, -1))
                prompt_rtg.append(discount_cumsum(promt_traj['rewards'][p_start:], gamma=1.)[:prompt_state[-1].shape[1] + 1].reshape(1, -1, 1))
                if prompt_rtg[-1].shape[1] <= prompt_state[-1].shape[1]:
                    prompt_rtg[-1] = np.concatenate([prompt_rtg[-1], np.zeros((1, 1, 1))], axis=1)
                # padding and state + reward normalization
                tlen = prompt_state[-1].shape[1]
                
                prompt_state[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), prompt_state[-1]], axis=1)
                # if not variant['no_state_normalize']:
                prompt_state[-1] = (prompt_state[-1] - state_mean) / state_std
                prompt_action[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., prompt_action[-1]], axis=1)
                prompt_rewards[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), prompt_rewards[-1]], axis=1)
                prompt_rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), prompt_rtg[-1]], axis=1) / scale
                prompt_timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), prompt_timesteps[-1]], axis=1)
                prompt_masks.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
                

                ###################################################################################################################


                # get sequences from dataset
                state_seg = traj['observations'][si : si+self.horizon]
                action_seg = traj['actions'][si : si+self.horizon]
                reward_seg = traj['rewards'][si : si+self.horizon].reshape(-1, 1)
                context_seg = traj['contexts'][si : si+self.horizon]
                
                if 'terminals' in traj:
                    done_seg = traj['terminals'][si : si+self.horizon].reshape(-1)
                else:
                    done_seg = traj['dones'][si : si+self.horizon].reshape(-1)

                timestep_seg = np.arange(si, si+state_seg.shape[0]).reshape(-1)
                timestep_seg[timestep_seg >= self.max_episode_steps] = self.max_episode_steps - 1  # padding cutoff

                rtg_seg = discount_cumsum(traj['rewards'][si:], gamma=1.)[:state_seg.shape[0] + 1].reshape(-1, 1)
                if rtg_seg.shape[0] <= state_seg.shape[0]:
                    rtg_seg = np.concatenate([rtg_seg, np.zeros((1, 1))], axis=0)

                # padding and state + reward normalization
                tlen = state_seg.shape[0]
                state_seg = np.concatenate([np.zeros((self.horizon - tlen, state_seg.shape[1])), state_seg], axis=0)
                state_seg = (state_seg - self.state_mean) / self.state_std
                context_seg = np.concatenate([np.zeros((self.horizon - tlen, context_seg.shape[1])), context_seg], axis=0)

                action_seg = np.concatenate([np.ones((self.horizon - tlen, action_seg.shape[1])) * -10., action_seg], axis=0)
                reward_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), reward_seg], axis=0)
                done_seg = np.concatenate([np.ones((self.horizon - tlen)) * 2, done_seg], axis=0)
                rtg_seg = np.concatenate([np.zeros((self.horizon - tlen, 1)), rtg_seg], axis=0) / self.return_scale
                timestep_seg = np.concatenate([np.zeros((self.horizon - tlen)), timestep_seg], axis=0)
                mask_seg = np.concatenate([np.zeros((self.horizon - tlen)), np.ones((tlen))], axis=0)

                states.append(state_seg)
                contexts.append(context_seg)
                actions.append(action_seg)
                rewards.append(reward_seg)
                dones.append(done_seg)
                rtg.append(rtg_seg)
                timesteps.append(timestep_seg)
                masks.append(mask_seg)
        prompt_state = torch.from_numpy(np.concatenate(prompt_state , axis=0)).to(dtype=torch.float32, device=device)
        prompt_action = torch.from_numpy(np.concatenate(prompt_action, axis=0)).to(dtype=torch.float32, device=device)
        prompt_rewards = torch.from_numpy(np.concatenate(prompt_rewards, axis=0)).to(dtype=torch.float32, device=device)

        prompt_rtg = torch.from_numpy(np.concatenate(prompt_rtg, axis=0)).to(dtype=torch.float32, device=device)
        prompt_timesteps = torch.from_numpy(np.concatenate(prompt_timesteps, axis=0)).to(dtype=torch.long, device=device)
        prompt_masks= torch.from_numpy(np.concatenate(prompt_masks, axis=0)).to(device=device)

        self.prompt_state,self.prompt_action,self.prompt_reward=prompt_state,prompt_action,prompt_rewards,
        self.prompt_done,self.prompt_rtg,self.prompt_tsp,self.prompt_mask = prompt_dones,prompt_rtg,prompt_timesteps,prompt_masks

        self.states = torch.from_numpy(np.stack(states, axis=0)).to(dtype=torch.float32, device=self.device)
        self.contexts = torch.from_numpy(np.stack(contexts, axis=0)).to(dtype=torch.float32, device=self.device)
        self.actions = torch.from_numpy(np.stack(actions, axis=0)).to(dtype=torch.float32, device=self.device)
        self.rewards = torch.from_numpy(np.stack(rewards, axis=0)).to(dtype=torch.float32, device=self.device)
        self.dones = torch.from_numpy(np.stack(dones, axis=0)).to(dtype=torch.long, device=self.device)
        self.rtg = torch.from_numpy(np.stack(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        self.timesteps = torch.from_numpy(np.stack(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        self.masks = torch.from_numpy(np.stack(masks, axis=0)).to(dtype=torch.float32, device=self.device)


    def __getitem__(self, index):
        return (
            self.states[index], 
            self.contexts[index],
            self.actions[index], 
            self.rewards[index], 
            self.dones[index], 
            self.rtg[index], 
            self.timesteps[index], 
            self.masks[index],
            self.prompt_state[index],
            self.prompt_action[index],
            self.prompt_reward[index],
            # self.prompt_done[index],
            self.prompt_rtg[index],
            self.prompt_tsp[index],
            self.prompt_mask[index],
        )


    def __len__(self):
        assert self.states.size(0) == self.contexts.size(0)
        return self.states.size(0)
















