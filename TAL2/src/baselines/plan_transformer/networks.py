import torch
import torch.nn as nn
import torch.nn.functional as F
from src.baselines.plan_transformer.layers import Block
from src.tal.layers import GatedHeteroRGCNLayer


class PlanTransformer(nn.Module):

    def __init__(self, config, in_feats, n_objects, n_hidden, n_states, n_layers, etypes, activation, dropout):
        super().__init__()
        self.name = 'PlanTransformer'

        self.config = config
        self.n_hidden = n_hidden  # * 128
        self.n_objects = n_objects
        self.n_states = n_states

        self.activation = nn.PReLU()
        self.layers = nn.ModuleList()
        self.layers.append(GatedHeteroRGCNLayer(config, in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers.append(GatedHeteroRGCNLayer(config, n_hidden, n_hidden, etypes, activation=activation))
        # * reset_parameters
        # * Only reset layer_1
        for layer in self.layers:
            layer.reset_parameters()

        self.embed = nn.Sequential(nn.Linear(config.PRETRAINED_VECTOR_SIZE, n_hidden),
                                   self.activation,
                                   nn.Linear(n_hidden, n_hidden))
        # * ------------------------------------------------------------
        if self.config.goal_representation_type == 2:
            self.pre_embed = nn.Sequential(
                nn.Linear(config.PRETRAINED_VECTOR_SIZE * 3 + config.N_STATES, config.PRETRAINED_VECTOR_SIZE),
                self.activation,
                nn.Linear(config.PRETRAINED_VECTOR_SIZE, config.PRETRAINED_VECTOR_SIZE)
            )
        # * ------------------------------------------------------------
        self.scene_fc_0 = nn.Linear(n_hidden * 4, n_hidden * 4)
        self.scene_fc_1 = nn.Linear(n_hidden * 2, n_hidden * 4)
        self.scene_fc_2 = nn.Linear(self.n_objects, 1)

        self.fc1 = nn.Linear(n_hidden * 4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(config.possibleActions))

        self.p1_object = nn.Linear(n_hidden * 4 + len(config.possibleActions), n_hidden)
        self.p2_object = nn.Linear(n_hidden, n_hidden)
        self.p3_object = nn.Linear(n_hidden, config.num_objects)

        self.q1_object = nn.Linear(n_hidden * 4 + len(config.possibleActions), n_hidden)
        self.q2_object = nn.Linear(n_hidden, n_hidden)
        self.q3_object = nn.Linear(n_hidden, config.num_objects)

        self.q1_state = nn.Linear(n_hidden * 4 + len(config.possibleActions), n_hidden)
        self.q2_state = nn.Linear(n_hidden, n_hidden)
        self.q3_state = nn.Linear(n_hidden, n_states)

        self.metric_1 = nn.Linear(in_feats, n_hidden)
        self.metric_2 = nn.Linear(n_hidden, n_hidden)

        # * ------------------------------------------------------------
        self.state_dim = self.n_hidden * 4  # * [1, 512]
        self.max_timestep = 100
        self.act_dim = 111  # * action: [action_name, args_1, args_2, state]
        # * Trajectory segment length input to model. (K in decision transformer. )
        self.context_len = self.config.context_len
        self.n_heads = 1
        self.dropout_p = 0.1
        self.n_blocks = 3

        # * transformer blocks
        input_seq_len = 2 * self.context_len + 1  # * Only prompt + state and action.
        blocks = [Block(self.n_hidden * 4, input_seq_len, self.n_heads, self.dropout_p) for _ in range(self.n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # * projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(self.n_hidden * 4)
        self.embed_timestep = nn.Embedding(self.max_timestep, self.n_hidden * 4)
        self.embed_state = torch.nn.Linear(self.state_dim, self.n_hidden * 4)

        # * Embedding actions.
        self.embed_action = nn.Linear(self.act_dim, self.n_hidden * 4)

        # * prediction heads
        self.predict_prompt = nn.Linear(self.n_hidden * 4, self.state_dim)
        self.predict_state = nn.Linear(self.n_hidden * 4, self.state_dim)
        self.predict_action = nn.Linear(self.n_hidden * 4, self.state_dim)

        # self.pos_embeddings = self.get_positional_embeddings()

    # def get_positional_embeddings(self, N=None, d_model=None) -> torch.Tensor:
    #     """
    #     B: batch size
    #     N: sequence length
    #     d_model: token dimension
    #     """
    #     if N is None:
    #         N = self.context_len
    #     if d_model is None:
    #         d_model = self.state_dim
    #     result = torch.ones(N, d_model)
    #     pos = torch.arange(N).unsqueeze(1)
    #     i = torch.arange(0, d_model, 2)
    #     div = 10000 ** (i / d_model)
    #     term = pos / div
    #     result[:, 0::2] = torch.sin(term)
    #     result[:, 1::2] = torch.cos(term)
    #     return result.to(self.config.device)

    def preprocess_graph(self, g):
        h_embed = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            h_embed = layer(g, h_embed)
        metric_part = g.ndata['feat']
        metric_part = self.activation(self.metric_1(metric_part))
        metric_part = self.activation(self.metric_2(metric_part))
        h_embed = torch.cat([h_embed, metric_part], dim=1)  # * [36, 256]

        scene_embedding = self.scene_fc_1(h_embed)  # * [36, 512]
        final_to_decode = self.scene_fc_2(scene_embedding.t())  # * [512, 1]
        final_to_decode = final_to_decode.t()  # * [1, 512]

        final_to_decode = final_to_decode.unsqueeze(0)  # * [1, 1, 512]

        return final_to_decode

    def forward(self, time_steps, prompt_state, states, actions):
        # * prompt_state is goal_state.
        # * ------------------------------------------------------------
        # * Preprocessing prompt_state and states.
        if len(actions) == 0:
            actions = [torch.zeros((self.act_dim), device=self.config.device)]

        prompt_embed = self.preprocess_graph(prompt_state)  # * [1, 1, 512]
        tmp_state_embeds = []
        for tmp_state in states:
            tmp_state_embed = self.preprocess_graph(tmp_state)
            tmp_state_embeds.append(tmp_state_embed)
        states_embed = torch.cat(tmp_state_embeds, dim=1)  # * [1, N, 512]  States: (Bs x  max_seq_length x state_dim)
        actions_embed = torch.stack(actions).unsqueeze(0)  # * [1, N, 111]

        # * ------------------------------------------------------------
        # * Use delta feature.
        states_embed = torch.abs(states_embed - prompt_embed)

        # * ------------------------------------------------------------
        # * Padding: for trajectory length less than self.context_len.
        # * State padding.
        state_traj_len = len(states)
        if state_traj_len < self.context_len:
            state_padding_len = self.context_len - state_traj_len
            states_padding_shape = (list(states_embed.shape[0:1]) + [state_padding_len] + list(states_embed.shape[2:]))
            states_padding = torch.zeros(states_padding_shape, dtype=states_embed.dtype, device=self.config.device)
            states_embed = torch.cat([states_embed, states_padding], dim=1)

        # * Action padding.
        action_traj_len = len(actions)
        if action_traj_len < self.context_len:
            act_padding_len = self.context_len - action_traj_len
            actions_padding_shape = (list(actions_embed.shape[0:1]) + [act_padding_len] + list(actions_embed.shape[2:]))
            actions_padding = torch.zeros(actions_padding_shape, dtype=actions_embed.dtype, device=self.config.device)
            actions_embed = torch.cat([actions_embed, actions_padding], dim=1)

        # * ------------------------------------------------------------
        time_embeddings = self.embed_timestep(time_steps)  # * [5, 128]
        time_embeddings = time_embeddings.unsqueeze(0)  # * [1, 5, 128]
        # time_embeddings = self.pos_embeddings

        # * time embeddings are treated similar to positional embeddings
        prompt_embeddings = self.embed_state(prompt_embed)  # * [1, 1, 128]
        state_embeddings = self.embed_state(states_embed) + time_embeddings  # * [1, self.context_len, 128]
        action_embeddings = self.embed_action(actions_embed) + time_embeddings

        B, T, _ = state_embeddings.shape  # * [1, 5, xxx]

        # * stack states and actions and reshape sequence as (prompt, d_s_0, a_0, d_s_1, a_1, d_s_2, a_2 ...)
        # h = torch.stack((prompt_embeddings, state_embeddings, action_embeddings), dim=1)  # * [1, 2 * T, 5, 128]
        h = torch.stack((state_embeddings, action_embeddings), dim=1)  # * [1, 2 * T + 1, 5, 128]
        h = h.permute(0, 2, 1, 3).reshape(B, 2 * T, self.n_hidden * 4)  # * [1, 10, 128]
        h = torch.cat([h, prompt_embeddings], dim=1)  # * [1, 11, 128]

        h = self.embed_ln(h)
        # * transformer and prediction
        h = self.transformer(h)  # * [1, 11, 128]
        # print('h.shape: {}'.format(h.shape))

        # * get h reshaped such that its size = (B x 3 x T x h_dim) and
        # * h[:, 0, t] is conditioned on the input sequence [prompt, d_s_0, a_0 ...
        # * h[:, 1, t] is conditioned on the input sequence [prompt, d_s_0, a_0 ... d_s_t
        # * h[:, 2, t] is conditioned on the input sequence [prompt, d_s_0, a_0 ... d_s_t, a_t
        # * that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # * each conditioned on all previous time steps plus
        # * the 3 input variables at that timestep (prompt, s_t, a_t) in sequence.
        # h = h.reshape(B, T, 3, self.n_hidden).permute(0, 2, 1, 3)  # * [1, 3, 5, 128]
        h = h[:, 1:, :]
        h = h.reshape(B, T, 2, self.n_hidden * 4).permute(0, 2, 1, 3)  # * [1, 2, 5, 128]
        # print(h[:, 1].shape)  # * [1, 5, 128]

        # # * ----------------------------------------------------------
        # * 02.
        # * Pred action.
        action_decode = self.predict_action(h[:, 1])
        action_decode = action_decode.squeeze(0)  # * [5, 512] Refer to final_to_decode in APN.

        action = self.activation(self.fc1(action_decode))
        action = self.activation(self.fc2(action))
        action = self.fc3(action)
        action = F.softmax(action, dim=1)
        pred_action_values = list(action[0])
        ind_max_action = pred_action_values.index(max(pred_action_values))
        one_hot_action = [0] * len(pred_action_values)
        one_hot_action[ind_max_action] = 1
        one_hot_action = torch.tensor(one_hot_action, dtype=torch.float32).view(1, -1)
        one_hot_action = one_hot_action.repeat(action_decode.shape[0], 1)
        if self.config.device is not None:
            one_hot_action = one_hot_action.to(self.config.device)

        # * Predicting the first argument of the action
        action_decode_with_one_hot = torch.cat([action_decode, one_hot_action], 1)  # * [5, 523]: [5, 512] + [5, 11]

        pred1_object = self.activation(self.p1_object(action_decode_with_one_hot))
        pred1_object = self.activation(self.p2_object(pred1_object))
        pred1_object = self.p3_object(pred1_object)
        # pred1_output = torch.sigmoid(pred1_object)
        pred1_output = F.softmax(pred1_object, dim=-1)

        # * Predicting the second argument of the action
        pred2_object = self.activation(self.q1_object(action_decode_with_one_hot))
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = self.q3_object(pred2_object)
        # pred2_object = torch.sigmoid(pred2_object)
        pred2_object = F.softmax(pred2_object, dim=-1)

        pred2_state = self.activation(self.q1_state(action_decode_with_one_hot))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = self.q3_state(pred2_state)
        # pred2_state = torch.sigmoid(pred2_state)
        pred2_state = F.softmax(pred2_state, dim=-1)

        # * action: [5, 11] | pred1_output: [5, 36] | pred2_object: [5, 36] | pred2_state: [5, 28]
        predicted_actions = torch.cat((action, pred1_output, pred2_object, pred2_state), 1)
        predicted_actions = predicted_actions.unsqueeze(0)  # * [1, 5, 111]
        # print('666' * 10)
        # print(predicted_actions.shape)

        # return prompt_preds, state_preds, action_preds
        return predicted_actions