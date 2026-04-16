"""
@File             : networks.py
@Project          : M3
@Time             : 2022/7/23 12:53
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/7/23 12:53     
@Version          : 1.0  
@Desciption       : None   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tal.layers import GatedHeteroRGCNLayer


class DDQN(nn.Module):
    """
    Model:
        Input: Start_State, Goal_State
        Output: Action
    Same model architecture as APN.
    Without final softmax and sigmoid activation.
    """

    def __init__(self, config, in_feats, n_objects, n_hidden, n_states, n_layers, etypes,
                 activation, dropout):
        super().__init__()
        self.name = 'DDQN'
        self.config = config
        self.n_hidden = n_hidden  # * 128
        self.n_objects = n_objects
        self.n_states = n_states
        self.activation = nn.PReLU()
        self.layers_1 = nn.ModuleList()
        self.layers_1.append(
            GatedHeteroRGCNLayer(config, in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers_1.append(
                GatedHeteroRGCNLayer(config, n_hidden, n_hidden, etypes, activation=activation))
        # * reset_parameters
        # * Only reset layer_1
        for layer in self.layers_1:
            layer.reset_parameters()

        self.layers_2 = nn.ModuleList()
        self.layers_2.append(
            GatedHeteroRGCNLayer(config, in_feats, n_hidden, etypes, activation=activation))
        for i in range(n_layers - 1):
            self.layers_2.append(
                GatedHeteroRGCNLayer(config, n_hidden, n_hidden, etypes, activation=activation))

        self.embed = nn.Sequential(nn.Linear(config.PRETRAINED_VECTOR_SIZE, n_hidden),
                                   self.activation,
                                   nn.Linear(n_hidden, n_hidden))
        if self.config.goal_representation_type == 2:
            self.pre_embed = nn.Sequential(
                nn.Linear(config.PRETRAINED_VECTOR_SIZE * 3 + config.N_STATES,
                          config.PRETRAINED_VECTOR_SIZE),
                self.activation,
                nn.Linear(config.PRETRAINED_VECTOR_SIZE, config.PRETRAINED_VECTOR_SIZE)
            )
        self.scene_fc_0 = nn.Linear(n_hidden * 4, n_hidden * 4)
        self.scene_fc_1 = nn.Linear(n_hidden * 2, n_hidden * 4)
        self.scene_fc_2 = nn.Linear(self.n_objects, 1)

        self.fc1 = nn.Linear(n_hidden * 4, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, len(config.possibleActions))

        self.p1_object = nn.Linear(n_hidden * 5 + len(config.possibleActions), n_hidden)
        self.p2_object = nn.Linear(n_hidden, n_hidden)
        self.p3_object = nn.Linear(n_hidden, 1)

        self.q1_object = nn.Linear(n_hidden * 5 + len(config.possibleActions) + 1, n_hidden)
        self.q2_object = nn.Linear(n_hidden, n_hidden)
        self.q3_object = nn.Linear(n_hidden, 1)

        self.q1_state = nn.Linear(n_hidden * 4 + len(config.possibleActions), n_hidden)
        self.q2_state = nn.Linear(n_hidden, n_hidden)
        self.q3_state = nn.Linear(n_hidden, n_states)

        self.metric_1 = nn.Linear(in_feats, n_hidden)
        self.metric_2 = nn.Linear(n_hidden, n_hidden)

        l = []
        for obj in config.all_objects:
            l.append(config.object2vec[obj])
        if self.config.device is not None:
            self.object_vec = torch.tensor(l, dtype=torch.float32, device=self.config.device)
        else:
            self.object_vec = torch.tensor(l, dtype=torch.float32)

    def forward(self, g, goalVec):
        # * g: Start state.
        # * goalVec: Final state.
        h_start = g.ndata['feat']
        for i, layer in enumerate(self.layers_1):
            h_start = layer(g, h_start)
        metric_part_start = g.ndata['feat']
        metric_part_start = self.activation(self.metric_1(metric_part_start))
        metric_part_start = self.activation(self.metric_2(metric_part_start))
        h_start = torch.cat([h_start, metric_part_start], dim=1)

        # * Use same layer as h_start.
        h_final = goalVec.ndata['feat']
        for i, layer in enumerate(self.layers_1):
            h_final = layer(goalVec, h_final)
        metric_part_final = goalVec.ndata['feat']
        metric_part_final = self.activation(self.metric_1(metric_part_final))
        metric_part_final = self.activation(self.metric_2(metric_part_final))
        h_final = torch.cat([h_final, metric_part_final], dim=1)

        # * Delta feature
        # scene_embedding = h_final - h_start
        scene_embedding = torch.abs(h_final - h_start)
        scene_embedding = self.scene_fc_1(scene_embedding)  # * [36, 512]

        final_to_decode = self.scene_fc_2(scene_embedding.t())  # * [512, 1]
        final_to_decode = final_to_decode.t()

        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.fc3(action)
        # action = F.softmax(action, dim=1)
        pred_action_values = list(action[0])
        ind_max_action = pred_action_values.index(max(pred_action_values))
        one_hot_action = [0] * len(pred_action_values)
        one_hot_action[ind_max_action] = 1
        one_hot_action = torch.tensor(one_hot_action, dtype=torch.float32).view(1, -1)
        if self.config.device is not None:
            one_hot_action = one_hot_action.to(self.config.device)

        # * Predicting the first argument of the action
        pred1_input = torch.cat([final_to_decode, one_hot_action], 1)
        pred1_object = self.activation(self.p1_object(
            torch.cat([pred1_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1),
                       self.activation(self.embed(self.object_vec))], 1)))
        pred1_object = self.activation(self.p2_object(pred1_object))
        pred1_object = self.p3_object(pred1_object)
        # pred1_output = torch.sigmoid(pred1_object)
        pred1_output = pred1_object

        # * Predicting the second argument of the action
        pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
        pred2_object = self.activation(self.q1_object(
            torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1),
                       self.activation(self.embed(self.object_vec)),
                       pred1_output.view(self.n_objects, 1)], 1)))
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = self.q3_object(pred2_object)
        # pred2_object = torch.sigmoid(pred2_object)

        pred2_state = self.activation(self.q1_state(pred2_input))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = self.q3_state(pred2_state)
        # pred2_state = torch.sigmoid(pred2_state)
        pred2_output = torch.cat([pred2_object.view(1, -1), pred2_state], 1)

        predicted_actions = torch.cat((action, pred1_output.view(1, -1), pred2_output.view(1, -1)),
                                      1).flatten()

        return predicted_actions
