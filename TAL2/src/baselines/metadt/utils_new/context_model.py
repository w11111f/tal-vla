"""
@Project     ：TAL_2024
@File        ：context_model.py
@Author      ：Xianqi-Zhang
@Date        ：2024/11/20
@Last        : 2024/11/28
@Description : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.modules import GraphFeatureExtractor, make_layers, SimNorm, MLP, load_resnet18


def weights_init_(m):
    """Initialize Policy weights."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class RNNContextEncoderReg(nn.Module):
    def __init__(
            self,
            config,
            device,
            state_dim=1024,
            action_dim=111,  # * 11 + 36 * 2 + 28
            context_dim=256,
            context_hidden_dim=512
    ):
        super().__init__()
        self.config = config
        self.device = device
        # activation = SimNorm()
        activation = nn.ReLU()

        self._graph_emb = GraphFeatureExtractor(config)  # * 1024
        self._task_emb = GraphFeatureExtractor(config)
        self.state_encoder = make_layers(state_dim, [512], context_dim, activation)
        self.action_encoder = make_layers(action_dim, [128], context_dim, activation)
        self.reward_encoder = nn.Sequential(nn.Linear(1, context_dim), activation)
        self.task_encoder = make_layers(state_dim, [512], context_dim, activation)
        self.layers = make_layers(
            context_dim * 4, [context_dim * 2, context_dim], context_dim, activation
        )
        self.gru = nn.GRU(input_size=context_dim, hidden_size=context_hidden_dim, num_layers=3)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.context_output = nn.Linear(context_hidden_dim, context_dim)
        # self.apply(weights_init_)

    # def forward(self, states, actions, rewards):
    #     """
    #     Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
    #     """
    #     states_embedding = torch.cat([self._graph_emb(state) for state in states])
    #     states_embedding = self.state_encoder(states_embedding)
    #     actions_embedding = self.action_encoder(torch.stack(actions))
    #     rewards_embedding = self.reward_encoder(
    #         torch.tensor(rewards).to(self.device).unsqueeze(0).reshape(-1, 1)
    #     )
    #     if len(states) < self.config.context_len:
    #         diff_dim = self.config.context_len - len(states)
    #         s_p = [diff_dim] + list(states_embedding.shape[1:])
    #         a_p = [diff_dim] + list(actions_embedding.shape[1:])
    #         r_p = [diff_dim] + list(rewards_embedding.shape[1:])
    #         states_embedding = torch.cat([states_embedding, torch.zeros(s_p).to(self.device)])
    #         actions_embedding = torch.cat([actions_embedding, torch.zeros(a_p).to(self.device)])
    #         rewards_embedding = torch.cat([rewards_embedding, torch.zeros(r_p).to(self.device)])
    #     h = torch.cat((states_embedding, actions_embedding, rewards_embedding), dim=-1)
    #
    #     # * gru_output: [seq_len * batch_size * hidden_dim]
    #     gru_output, _ = self.gru(h)
    #     contexts = self.context_output(gru_output[-1])  # * 128
    #     return contexts

    def forward(self, states, actions, rewards, task):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        """
        states_embedding = torch.cat([self._graph_emb(state) for state in states])
        states_embedding = self.state_encoder(states_embedding)
        actions_embedding = self.action_encoder(torch.stack(actions))
        rewards_embedding = self.reward_encoder(
            torch.tensor(rewards).to(self.device).unsqueeze(0).reshape(-1, 1)
        )
        task_embedding = self.task_encoder(self._task_emb(task))
        task_diff_dim = self.config.context_len - task_embedding.shape[0]
        t_p = [task_diff_dim] + list(task_embedding.shape[1:])
        task_embedding = torch.cat([task_embedding, torch.zeros(t_p).to(self.device)])
        if len(states) < self.config.context_len:
            diff_dim = self.config.context_len - len(states)
            s_p = [diff_dim] + list(states_embedding.shape[1:])
            a_p = [diff_dim] + list(actions_embedding.shape[1:])
            r_p = [diff_dim] + list(rewards_embedding.shape[1:])
            states_embedding = torch.cat([states_embedding, torch.zeros(s_p).to(self.device)])
            actions_embedding = torch.cat([actions_embedding, torch.zeros(a_p).to(self.device)])
            rewards_embedding = torch.cat([rewards_embedding, torch.zeros(r_p).to(self.device)])
        h = torch.cat(
            (states_embedding, actions_embedding, rewards_embedding, task_embedding), dim=-1
        )
        h = self.layers(h)
        # * gru_output: [seq_len * batch_size * hidden_dim]
        gru_output, _ = self.gru(h)
        contexts = self.context_output(gru_output[-1])  # * 128
        return contexts


class RewardDecoderReg(nn.Module):

    def __init__(
            self,
            config,
            device,
            state_dim=1024,
            action_dim=111,  # * 11 + 36 * 2 + 28
            context_dim=256,
            context_hidden_dim=512
    ):
        super().__init__()
        self.config = config
        self.device = device
        # activation = SimNorm()
        activation = nn.ReLU()

        self._graph_emb = GraphFeatureExtractor(config)  # * 1024
        self._task_emb = GraphFeatureExtractor(config)
        self.state_encoder = make_layers(state_dim, [512], context_dim, activation)
        self.action_encoder = make_layers(action_dim, [128], context_dim, activation)
        self.task_encoder = make_layers(state_dim, [512], context_dim, activation)
        self.layers1 = make_layers(
            context_dim * 5, [context_dim * 3, context_dim * 2], context_hidden_dim, activation
        )
        self.layers2 = make_layers(context_hidden_dim, [context_hidden_dim, context_hidden_dim], 1)
        # self.apply(weights_init_)

    def forward(self, state, action, next_state, task, context):
        # extract features for states, actions
        s_feature = self.state_encoder(self._graph_emb(state))
        a_feature = self.action_encoder(action.unsqueeze(0))
        n_feature = self.state_encoder(self._graph_emb(next_state))
        t_feature = self.task_encoder(self._task_emb(task))

        h = torch.cat((s_feature, a_feature, n_feature, t_feature, context.unsqueeze(0)), dim=-1)
        reward_predict = self.layers2(self.layers1(h))
        return reward_predict


class RNNContextEncoderCls(nn.Module):
    def __init__(
            self,
            config,
            device,
            reward_dim=3,
            state_dim=1024,
            action_dim=111,  # * 11 + 36 * 2 + 28
            context_dim=64,
            context_hidden_dim=256
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.act = nn.ReLU()  # * SimNorm()

        self._graph_emb = GraphFeatureExtractor(config)  # * 1024
        self._task_emb = GraphFeatureExtractor(config)
        self.action_encoder = nn.Linear(action_dim, 36)
        self.reward_encoder = nn.Linear(reward_dim, 32)

        self.layers = load_resnet18(in_channels=self.config.context_len, out_channels=context_dim)
        self.gru = nn.GRU(input_size=context_dim, hidden_size=context_hidden_dim, num_layers=3)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        self.context_output = nn.Linear(context_hidden_dim, context_dim)
        self.apply(weights_init_)

    def forward(self, states, actions, rewards, task):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        """
        states_embedding = torch.cat([self._graph_emb(state) for state in states])
        actions_embedding = self.action_encoder(actions)
        rewards_embedding = self.reward_encoder(rewards)
        task_embedding = self._task_emb(task)
        task_diff_dim = self.config.context_len - task_embedding.shape[0]
        t_p = [task_diff_dim] + list(task_embedding.shape[1:])
        task_embedding = torch.cat([task_embedding, torch.zeros(t_p).to(self.device)])
        if len(states) < self.config.context_len:
            diff_dim = self.config.context_len - len(states)
            s_p = [diff_dim] + list(states_embedding.shape[1:])
            a_p = [diff_dim] + list(actions_embedding.shape[1:])
            r_p = [diff_dim] + list(rewards_embedding.shape[1:])
            states_embedding = torch.cat([states_embedding, torch.zeros(s_p).to(self.device)])
            actions_embedding = torch.cat([actions_embedding, torch.zeros(a_p).to(self.device)])
            rewards_embedding = torch.cat([rewards_embedding, torch.zeros(r_p).to(self.device)])
        h = torch.cat(
            (states_embedding, actions_embedding, rewards_embedding, task_embedding), dim=-1
        ).reshape(1, -1, 46, 46)
        h = self.layers(self.act(h))
        # * gru_output: [seq_len * batch_size * hidden_dim]
        gru_output, _ = self.gru(h)
        contexts = self.context_output(gru_output[-1])  # * 128
        return contexts


class RewardDecoderCls(nn.Module):

    def __init__(
            self,
            config,
            device,
            reward_dim=3,
            state_dim=1024,
            action_dim=111,  # * 11 + 36 * 2 + 28
            context_dim=64,
            context_hidden_dim=256,
            action_hidden_dim=36,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.act = nn.ReLU()  # * SimNorm()

        self._graph_emb = GraphFeatureExtractor(config)  # * 1024
        self._task_emb = GraphFeatureExtractor(config)
        self.action_encoder = nn.Linear(action_dim, action_hidden_dim)
        self.context_encoder = nn.Linear(context_dim, context_hidden_dim)
        # self.fc = nn.Linear(state_dim * 3 + context_hidden_dim + action_hidden_dim, 1 * 58 * 58)
        self.layers = load_resnet18(in_channels=1, out_channels=reward_dim)
        self.apply(weights_init_)

    def forward(self, state, action, next_state, task, context):
        # extract features for states, actions
        s_feature = self._graph_emb(state)
        a_feature = self.action_encoder(action.unsqueeze(0))
        n_feature = self._graph_emb(next_state)
        t_feature = self._task_emb(task)
        c_feature = self.context_encoder(context.unsqueeze(0))
        h = torch.cat((s_feature, a_feature, n_feature, t_feature, c_feature), dim=-1)
        # h = self.fc(self.act(h)).reshape(1, 1, 58, 58)
        h = h.reshape(1, 1, 58, 58)
        reward_predict = self.layers(self.act(h))
        return reward_predict

# class RewardDecoderCls(nn.Module):
#
#     def __init__(
#             self,
#             config,
#             device,
#             reward_dim=3,
#             state_dim=1024,
#             action_dim=111,  # * 11 + 36 * 2 + 28
#             action_hidden_dim=36,
#             context_dim=64,
#             context_hidden_dim=256
#     ):
#         super().__init__()
#         self.config = config
#         self.device = device
#         self.act = nn.ReLU()  # * SimNorm()
#
#         self._graph_emb_1 = GraphFeatureExtractor(config)  # * 1024
#         self._graph_emb_2 = GraphFeatureExtractor(config)
#         self._task_emb = GraphFeatureExtractor(config)
#         self.state_encoder_1 = nn.Linear(state_dim, context_hidden_dim)
#         self.state_encoder_2 = nn.Linear(state_dim, context_hidden_dim)
#         self.task_encoder = nn.Linear(state_dim, context_hidden_dim)
#         self.action_encoder = nn.Linear(action_dim, action_hidden_dim)
#         self.layers = make_layers(
#             context_hidden_dim * 3 + action_hidden_dim + context_dim, [context_hidden_dim * 2],
#             context_hidden_dim,
#             self.act
#         )
#         self.fc = nn.Linear(context_hidden_dim, reward_dim)
#         # self.apply(weights_init_)
#
#     def forward(self, state, action, next_state, task, context):
#         # extract features for states, actions
#         s_feature = self.state_encoder_1(self._graph_emb_1(state))
#         n_feature = self.state_encoder_2(self._graph_emb_2(next_state))
#         t_feature = self.task_encoder(self._task_emb(task))
#         a_feature = self.action_encoder(action.unsqueeze(0))
#         h = torch.cat((s_feature, a_feature, n_feature, t_feature, context.unsqueeze(0)), dim=-1)
#         reward_predict = self.fc(self.layers(self.act(h)))
#         # reward_predict = reward_predict.softmax(dim=1)
#         return reward_predict
