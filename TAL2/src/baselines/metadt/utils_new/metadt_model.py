"""
@Project     ：TAL_2024
@File        ：metadt_model.py
@Author      ：Xianqi-Zhang
@Date        ：2024/12/2
@Last        : 2024/12/4
@Description : 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from src.modules.modules import GraphFeatureExtractor, load_resnet18
from src.baselines.metadt.meta_dt.trajectory_gpt2 import GPT2Model


def weights_init_(m):
    """Initialize Policy weights."""
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0, std=0.5)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MetaDecisionTransformer(nn.Module):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    Code based on: https://github.com/NJU-RL/Meta-DT/blob/main/meta_dt/model.py
    """

    def __init__(
            self,
            config,
            device,
            hidden_size=1024,
            max_ep_len=4096,
            state_dim=1024,
            action_dim=111,  # * 11 + 36 * 2 + 28
            context_dim=64
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_name_dim = 11
        self.action_object_dim = 36
        self.action_state_dim = 28
        self.act = nn.ReLU()  # * SimNorm()

        gpt2_config = transformers.GPT2Config(
            vocab_size=1,  # * Doesn't matter -- we don't use the vocab.
            n_embd=hidden_size,
            n_layer=6,
            n_head=4,
            n_inner=4 * hidden_size,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )
        self.transformer = GPT2Model(gpt2_config)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self._task_emb = GraphFeatureExtractor(config, output_dim=hidden_size, layer_type='conv')
        self._graph_emb = GraphFeatureExtractor(config, output_dim=hidden_size, layer_type='conv')
        self.embed_states = nn.Conv1d(state_dim + context_dim, hidden_size, kernel_size=1)
        self.embed_action = nn.Conv1d(action_dim, hidden_size, kernel_size=1)
        self.embed_return = nn.Conv1d(1, hidden_size, kernel_size=1)

        # * Note: we don't predict states or returns for the paper.
        self.pred_return = nn.Linear(hidden_size, 1)
        self.pred_state = nn.Linear(hidden_size, state_dim)
        self.l_dim = 42
        # self.pred_action = nn.Linear(hidden_size, action_dim)
        # self.layers = nn.Linear(hidden_size, self.layer_dim * self.layer_dim)
        self.layers = nn.Conv1d(hidden_size, self.l_dim * self.l_dim, kernel_size=1)
        self.pred_action = load_resnet18(1, hidden_size)

        # * Action name: 11, object name: 36, state: 28
        self.pred_action_name = nn.Conv1d(hidden_size, self.action_name_dim, kernel_size=1)
        self.pred_object_1 = nn.Conv1d(hidden_size, self.action_object_dim, kernel_size=1)
        self.pred_object_2 = nn.Conv1d(hidden_size, self.action_object_dim, kernel_size=1)
        self.pred_state = nn.Conv1d(hidden_size, self.action_state_dim, kernel_size=1)
        self.apply(weights_init_)

    def forward(self, task, contexts, states, actions, returns_to_go, timesteps,
                attention_mask=None, prompt=None):
        seq_length = self.config.context_len
        horizon = 3 * self.config.context_len  # * GPT-2 input horizon.
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length), dtype=torch.long)

        states_embedding = torch.cat([self._graph_emb(state) for state in states])
        states_embedding = torch.cat((states_embedding, contexts), dim=1)
        states_embedding = self.embed_states(states_embedding.unsqueeze(0).permute(0, 2, 1))
        actions_embedding = self.embed_action(actions.unsqueeze(0).permute(0, 2, 1))
        returns_embedding = self.embed_return(returns_to_go.unsqueeze(0).permute(0, 2, 1))
        states_embedding = states_embedding.permute(0, 2, 1).squeeze(0)
        actions_embedding = actions_embedding.permute(0, 2, 1).squeeze(0)
        returns_embedding = returns_embedding.permute(0, 2, 1).squeeze(0)
        time_embeddings = self.embed_timestep(timesteps)

        # * Time embeddings are treated similar to positional embeddings.
        states_embedding = states_embedding + time_embeddings
        actions_embedding = actions_embedding + time_embeddings
        returns_embedding = returns_embedding + time_embeddings

        task_embedding = self._task_emb(task)
        task_diff_dim = self.config.context_len - task_embedding.shape[0]
        t_p = [task_diff_dim] + list(task_embedding.shape[1:])
        task_embedding = torch.cat([torch.zeros(t_p).to(self.device), task_embedding]).unsqueeze(0)
        if len(states) < self.config.context_len:
            diff_dim = self.config.context_len - len(states)
            s_p = [diff_dim] + list(states_embedding.shape[1:])
            a_p = [diff_dim] + list(actions_embedding.shape[1:])
            r_p = [diff_dim] + list(returns_embedding.shape[1:])
            states_embedding = torch.cat([torch.zeros(s_p).to(self.device), states_embedding])
            actions_embedding = torch.cat([torch.zeros(a_p).to(self.device), actions_embedding])
            returns_embedding = torch.cat([torch.zeros(r_p).to(self.device), returns_embedding])
        h = torch.stack(
            (returns_embedding, states_embedding, actions_embedding), dim=1
        ).unsqueeze(0).permute(0, 2, 1, 3).reshape(1, horizon, self.hidden_size)

        # * ----------------------------------------------------------
        # * GPT-2 inputs.
        stacked_inputs = torch.cat((task_embedding, h), dim=1)
        # * GPT-2 mask.
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(1, horizon)
        p_attention_mask = torch.ones((1, task_embedding.shape[1])).to(self.device)
        stacked_attention_mask = torch.cat((p_attention_mask, stacked_attention_mask), dim=1)
        # * GPT-2 forward.
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )

        # * ----------------------------------------------------------
        # h = self.layers(self.act(h))
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(1, -1, 3, self.hidden_size).permute(0, 2, 1, 3)
        # * return_preds [1, *, 1], state_preds [1, *, 512], action_preds [1, *, 111]
        # return_preds = self.pred_return(x[:, 2])[:, -seq_length:, :]  # * given state and action
        # state_preds = self.pred_state(x[:, 2])[:, -seq_length:, :]  # * given state and action

        b, c = x[:, 1].shape[0], x[:, 1].shape[1]  # * given state
        # * For nn.Linear().
        # action_pred = self.layers(x[:, 1]).reshape(c, 1, self.layer_dim, self.layer_dim)
        # * For nn.Conv1d().
        action_pred = self.layers(x[:, 1].permute(0, 2, 1)).reshape(c, 1, self.l_dim, self.l_dim)
        action_feature = self.pred_action(action_pred).unsqueeze(0).permute(0, 2, 1)

        action_name = self.pred_action_name(action_feature)
        object_1 = self.pred_object_1(action_feature)
        object_2 = self.pred_object_2(action_feature)
        state = self.pred_state(action_feature)

        action_name = action_name.reshape(b, c, self.action_name_dim)[:, -seq_length:, :]
        object_1 = object_1.reshape(b, c, self.action_object_dim)[:, -seq_length:, :]
        object_2 = object_2.reshape(b, c, self.action_object_dim)[:, -seq_length:, :]
        state = state.reshape(b, c, self.action_state_dim)[:, -seq_length:, :]

        return action_name, object_1, object_2, state
