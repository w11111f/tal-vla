import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from src.tal.layers import GatedHeteroRGCNLayer


class AFE(nn.Module):
    """
    Model: Action Feature Extractor, AFE
    Input: State, Goal(State)
    Output: Action, Delta_Feature
    """

    def __init__(self, config, in_feats, n_objects, n_hidden, n_states, n_layers, etypes,
                 activation, dropout):
        super().__init__()
        self.name = 'AFE'
        cprint('Mode: {}'.format(self.name), 'green')
        self.config = config
        self.n_hidden = n_hidden  # * 128
        self.n_objects = n_objects
        self.n_states = n_states

        self.relu = nn.ReLU()
        self.activation = nn.PReLU()
        self.layers = nn.ModuleList()
        self.layers.append(
            GatedHeteroRGCNLayer(config, in_feats, n_hidden, etypes, activation=activation)
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GatedHeteroRGCNLayer(config, n_hidden, n_hidden, etypes, activation=activation)
            )
        # * Init parameters.
        for layer in self.layers:
            layer.reset_parameters()

        self.embed = nn.Sequential(
            nn.Linear(config.PRETRAINED_VECTOR_SIZE, n_hidden),
            self.activation,
            nn.Linear(n_hidden, n_hidden)
        )
        self.metric_1 = nn.Linear(in_feats, n_hidden)
        self.metric_2 = nn.Linear(n_hidden, n_hidden)

        self.scene_fc_1 = nn.Linear(n_hidden * 2, n_hidden * 4)
        self.scene_fc_2 = nn.Linear(self.n_objects, 1)

        self.fc_final_to_decode_extension = nn.Linear(n_hidden * 4, n_hidden * 32)
        self.fc_final_to_decode_reduction = nn.Linear(n_hidden * 32, n_hidden * 4)

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

        l = []
        for obj in config.all_objects:
            l.append(config.object2vec[obj])
        if self.config.device is not None:
            self.object_vec = torch.tensor(l, dtype=torch.float32, device=self.config.device)
        else:
            self.object_vec = torch.tensor(l, dtype=torch.float32)

    def _encode_graph(self, graph):
        hidden = graph.ndata['feat']
        for layer in self.layers:
            hidden = layer(graph, hidden)
        metric_part = graph.ndata['feat']
        metric_part = self.activation(self.metric_1(metric_part))
        metric_part = self.activation(self.metric_2(metric_part))
        return torch.cat([hidden, metric_part], dim=1)

    def _reshape_objects(self, tensor):
        total_nodes = tensor.shape[0]
        if total_nodes % self.n_objects != 0:
            raise ValueError(
                f"Expected node count to be divisible by n_objects={self.n_objects}, got {total_nodes}."
            )
        batch_size = total_nodes // self.n_objects
        return tensor.view(batch_size, self.n_objects, -1), batch_size

    def _decode_outputs(self, final_to_decode):
        batch_size = final_to_decode.shape[0]
        action_logits = self.fc3(self.activation(self.fc2(self.activation(self.fc1(final_to_decode)))))
        action = F.softmax(action_logits, dim=1)
        max_action_idx = torch.argmax(action, dim=1, keepdim=True)
        one_hot_action = torch.zeros_like(action)
        one_hot_action.scatter_(1, max_action_idx, 1.0)

        object_embed = self.activation(self.embed(self.object_vec)).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        pred1_input = torch.cat([final_to_decode, one_hot_action], dim=1)
        pred1_expand = pred1_input.unsqueeze(1).expand(-1, self.n_objects, -1)
        pred1_object = self.activation(self.p1_object(torch.cat([pred1_expand, object_embed], dim=2)))
        pred1_object = self.activation(self.p2_object(pred1_object))
        pred1_logits = self.p3_object(pred1_object).squeeze(-1)
        pred1_output = torch.sigmoid(pred1_logits)

        pred2_input = torch.cat([final_to_decode, one_hot_action], dim=1)
        pred2_expand = pred2_input.unsqueeze(1).expand(-1, self.n_objects, -1)
        pred2_object = self.activation(
            self.q1_object(
                torch.cat([pred2_expand, object_embed, pred1_logits.unsqueeze(-1)], dim=2)
            )
        )
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = torch.sigmoid(self.q3_object(pred2_object).squeeze(-1))

        pred2_state = self.activation(self.q1_state(pred2_input))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = torch.sigmoid(self.q3_state(pred2_state))
        pred2_output = torch.cat([pred2_object, pred2_state], dim=1)
        predicted_actions = torch.cat((action, pred1_output, pred2_output), dim=1)
        return predicted_actions

    def forward(self, g, goalVec):
        # * g: Start state.
        # * goalVec: Final state.
        h_start = self._encode_graph(g)
        h_final = self._encode_graph(goalVec)

        scene_embedding = torch.abs(h_final - h_start)
        scene_embedding = self.activation(self.scene_fc_1(scene_embedding))
        scene_embedding, batch_size = self._reshape_objects(scene_embedding)
        scene_embedding = self.activation(self.scene_fc_2(scene_embedding.transpose(1, 2))).squeeze(-1)

        delta_feature = self.relu(self.fc_final_to_decode_extension(scene_embedding))
        final_to_decode = self.fc_final_to_decode_reduction(delta_feature)
        predicted_actions = self._decode_outputs(final_to_decode)

        if batch_size == 1:
            return predicted_actions.squeeze(0), delta_feature.squeeze(0)
        return predicted_actions, delta_feature


class AFE_MLP(nn.Module):
    """
    Replace GNN with MLP.
    Input: State, Goal(State)
    Output: Action, Delta_Feature
    """

    def __init__(self, config, in_feats, n_objects, n_hidden, n_states, n_layers, etypes,
                 activation, dropout):
        super().__init__()
        self.name = 'AFE_MLP'
        cprint('Mode: {}'.format(self.name), 'green')
        self.config = config
        self.n_hidden = n_hidden  # * 128
        self.n_objects = n_objects
        self.n_states = n_states

        self.relu = nn.ReLU()
        self.activation = nn.PReLU()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, n_hidden))
        self.layers.append(self.activation)
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(self.activation)
        # * Init parameters.
        for layer in self.layers:
            layer.reset_parameters()

        self.embed = nn.Sequential(
            nn.Linear(config.PRETRAINED_VECTOR_SIZE, n_hidden),
            self.activation,
            nn.Linear(n_hidden, n_hidden)
        )
        self.metric_1 = nn.Linear(in_feats, n_hidden)
        self.metric_2 = nn.Linear(n_hidden, n_hidden)

        self.scene_fc_1 = nn.Linear(n_hidden * 2, n_hidden * 4)
        self.scene_fc_2 = nn.Linear(self.n_objects, 1)

        self.fc_final_to_decode_extension = nn.Linear(n_hidden * 4, n_hidden * 32)
        self.fc_final_to_decode_reduction = nn.Linear(n_hidden * 32, n_hidden * 4)

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

        l = []
        for obj in config.all_objects:
            l.append(config.object2vec[obj])
        if self.config.device is not None:
            self.object_vec = torch.tensor(l, dtype=torch.float32, device=self.config.device)
        else:
            self.object_vec = torch.tensor(l, dtype=torch.float32)

    def _encode_graph(self, graph):
        hidden = graph.ndata['feat']
        for layer in self.layers:
            hidden = layer(hidden)
        metric_part = graph.ndata['feat']
        metric_part = self.activation(self.metric_1(metric_part))
        metric_part = self.activation(self.metric_2(metric_part))
        return torch.cat([hidden, metric_part], dim=1)

    def _reshape_objects(self, tensor):
        total_nodes = tensor.shape[0]
        if total_nodes % self.n_objects != 0:
            raise ValueError(
                f"Expected node count to be divisible by n_objects={self.n_objects}, got {total_nodes}."
            )
        batch_size = total_nodes // self.n_objects
        return tensor.view(batch_size, self.n_objects, -1), batch_size

    def _decode_outputs(self, final_to_decode):
        batch_size = final_to_decode.shape[0]
        action_logits = self.fc3(self.activation(self.fc2(self.activation(self.fc1(final_to_decode)))))
        action = F.softmax(action_logits, dim=1)
        max_action_idx = torch.argmax(action, dim=1, keepdim=True)
        one_hot_action = torch.zeros_like(action)
        one_hot_action.scatter_(1, max_action_idx, 1.0)

        object_embed = self.activation(self.embed(self.object_vec)).unsqueeze(0).expand(
            batch_size, -1, -1
        )

        pred1_input = torch.cat([final_to_decode, one_hot_action], dim=1)
        pred1_expand = pred1_input.unsqueeze(1).expand(-1, self.n_objects, -1)
        pred1_object = self.activation(self.p1_object(torch.cat([pred1_expand, object_embed], dim=2)))
        pred1_object = self.activation(self.p2_object(pred1_object))
        pred1_logits = self.p3_object(pred1_object).squeeze(-1)
        pred1_output = torch.sigmoid(pred1_logits)

        pred2_input = torch.cat([final_to_decode, one_hot_action], dim=1)
        pred2_expand = pred2_input.unsqueeze(1).expand(-1, self.n_objects, -1)
        pred2_object = self.activation(
            self.q1_object(
                torch.cat([pred2_expand, object_embed, pred1_logits.unsqueeze(-1)], dim=2)
            )
        )
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = torch.sigmoid(self.q3_object(pred2_object).squeeze(-1))

        pred2_state = self.activation(self.q1_state(pred2_input))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = torch.sigmoid(self.q3_state(pred2_state))
        pred2_output = torch.cat([pred2_object, pred2_state], dim=1)
        predicted_actions = torch.cat((action, pred1_output, pred2_output), dim=1)
        return predicted_actions

    def forward(self, g, goalVec):
        # * g: Start state.
        # * goalVec: Final state.
        h_start = self._encode_graph(g)
        h_final = self._encode_graph(goalVec)

        scene_embedding = torch.abs(h_final - h_start)
        scene_embedding = self.activation(self.scene_fc_1(scene_embedding))
        scene_embedding, batch_size = self._reshape_objects(scene_embedding)
        scene_embedding = self.activation(self.scene_fc_2(scene_embedding.transpose(1, 2))).squeeze(-1)

        delta_feature = self.relu(self.fc_final_to_decode_extension(scene_embedding))
        final_to_decode = self.fc_final_to_decode_reduction(delta_feature)
        predicted_actions = self._decode_outputs(final_to_decode)

        if batch_size == 1:
            return predicted_actions.squeeze(0), delta_feature.squeeze(0)
        return predicted_actions, delta_feature
