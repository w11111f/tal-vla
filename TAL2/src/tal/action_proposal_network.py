import torch
import torch.nn as nn
import torch.nn.functional as F
from src.tal.layers import GatedHeteroRGCNLayer


def action2vec(config, action, num_objects, num_states):
    actionArray = torch.zeros(len(config.possibleActions))
    actionArray[config.possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects + 1)
    # * Predicate 2 and 3 will be predicted together.
    predicate2 = torch.zeros(num_objects + 1)
    predicate3 = torch.zeros(num_states)
    if len(action['args']) == 0:
        predicate1[-1] = 1
        predicate2[-1] = 1
    elif len(action['args']) == 1:
        predicate1[config.object2idx[action['args'][0]]] = 1
        predicate2[-1] = 1
    else:
        predicate1[config.object2idx[action['args'][0]]] = 1
        predicate2[config.object2idx[action['args'][1]]] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)


def action2vec_cons(config, action, num_objects, num_states):
    actionArray = torch.zeros(len(config.possibleActions))
    actionArray[config.possibleActions.index(action['name'])] = 1
    predicate1 = torch.zeros(num_objects)
    predicate2 = torch.zeros(num_objects)
    predicate3 = torch.zeros(num_states)
    if len(action['args']) == 1:
        predicate1[config.object2idx[action['args'][0]]] = 1
    elif len(action['args']) == 2:
        predicate1[config.object2idx[action['args'][0]]] = 1
        predicate2[config.object2idx[action['args'][1]]] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)


def action2vec_lstm(config, action, num_objects, num_states, num_hidden, embedder):
    if config.device is not None:
        actionArray = torch.zeros(len(config.possibleActions), device=config.device)
        actionArray[config.possibleActions.index(action['name'])] = 1
        predicate3 = torch.zeros(num_states, device=config.device)
        predicate2 = torch.zeros(num_hidden, device=config.device)
        if len(action['args']) == 0:
            predicate1 = torch.zeros(num_hidden)
        elif len(action['args']) == 1:
            predicate1 = embedder(
                torch.tensor(config.object2vec[action['args'][0]],
                             dtype=torch.float32,
                             device=config.device)
            )
        else:
            predicate1 = embedder(
                torch.tensor(config.object2vec[action['args'][0]],
                             dtype=torch.float32,
                             device=config.device)
            )
            predicate2 = embedder(
                torch.tensor(config.object2vec[action['args'][1]],
                             dtype=torch.float32,
                             device=config.device)
            )
    else:
        actionArray = torch.zeros(len(config.possibleActions))
        actionArray[config.possibleActions.index(action['name'])] = 1
        predicate3 = torch.zeros(num_states)
        predicate2 = torch.zeros(num_hidden)
        if len(action['args']) == 0:
            predicate1 = torch.zeros(num_hidden)
        elif len(action['args']) == 1:
            predicate1 = embedder(
                torch.tensor(config.object2vec[action['args'][0]], dtype=torch.float32)
            )
        else:
            predicate1 = embedder(
                torch.tensor(config.object2vec[action['args'][0]], dtype=torch.float32)
            )
            predicate2 = embedder(
                torch.tensor(config.object2vec[action['args'][1]], dtype=torch.float32)
            )

    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)


def action2ids(config, action, num_objects, num_states):
    actionID = config.possibleActions.index(action['name'])
    predicate1, predicate2 = 0, 0
    if len(action['args']) == 0:
        predicate1 = num_objects + 1
        predicate2 = num_objects + 1
    elif len(action['args']) == 1:
        predicate1 = config.object2idx[action['args'][0]]
        predicate2 = num_objects + 1
    else:
        predicate1 = config.object2idx[action['args'][0]]
        predicate2 = config.object2idx[action['args'][1]]
    return actionID, predicate1, predicate2


def vec2action(config, vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(config.possibleActions)])
    ret_action['name'] = config.possibleActions[action_array.index(max(action_array))]
    ret_action['args'] = []
    if ret_action['name'] in ['moveUp', 'moveDown', 'placeRamp']:
        return ret_action
    object1_array = list(
        vec[len(config.possibleActions):len(config.possibleActions) + num_objects + 1])
    object1_ind = object1_array.index(max(object1_array))
    if object1_ind == len(object1_array) - 1:
        return ret_action
    else:
        ret_action['args'].append(idx2object[object1_ind])
    object2_or_state_array = list(vec[len(config.possibleActions) + num_objects + 1:])
    object2_or_state_ind = object2_or_state_array.index(max(object2_or_state_array))
    if (object2_or_state_ind < num_objects):
        ret_action['args'].append(idx2object[object2_or_state_ind])
    elif (object2_or_state_ind == num_objects):
        pass
    else:
        ret_action['args'].append(config.possibleStates[object2_or_state_ind - num_objects - 1])
    return ret_action


def vec2action_grammatical(config, vec, num_objects, num_states, idx2object):
    ret_action = {}
    action_array = list(vec[:len(config.possibleActions)])
    ret_action['name'] = config.possibleActions[action_array.index(max(action_array))]
    ret_action['args'] = []
    if ret_action['name'] in config.noArgumentActions:
        return ret_action
    object1_array = list(
        vec[len(config.possibleActions):len(config.possibleActions) + num_objects])
    object1_ind = object1_array.index(max(object1_array))
    ret_action['args'].append(idx2object[object1_ind])
    if ret_action['name'] in config.singleArgumentActions:
        return ret_action
    object2_array = list(
        vec[len(config.possibleActions) + num_objects:len(
            config.possibleActions) + num_objects + num_objects])
    state_array = list(vec[len(config.possibleActions) + num_objects + num_objects:])
    assert len(state_array) == len(config.possibleStates)
    # * 2 objects
    object2_ind = object2_array.index(max(object2_array))
    # * Due to the robot can not place object on itself, if object1 and object2 are the same object,
    # * we choose the second maximum value.
    if object1_ind == object2_ind:
        object2_array[object2_ind] = 0
        object2_ind = object2_array.index(max(object2_array))
    ret_action['args'].append(idx2object[object2_ind])
    return ret_action


def tool2object_likelihoods(config, num_objects, tool_likelihoods):
    object_likelihoods = torch.zeros(num_objects)
    for i, tool in enumerate(config.TOOLS2):
        object_likelihoods[config.object2idx[tool]] = tool_likelihoods[i]
    return object_likelihoods


class APN(nn.Module):
    """
    APN: Action Proposal Network
    Model:
        Input: State, Goal(State)
        Output: Action
    """

    def __init__(self, config, in_feats, n_objects, n_hidden, n_states, n_layers, etypes,
                 activation, dropout):
        super().__init__()
        self.name = 'APN'
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
        # * -----------------------------------------------------------
        if self.config.goal_representation_type == 2:
            self.pre_embed = nn.Sequential(
                nn.Linear(config.PRETRAINED_VECTOR_SIZE * 3 + config.N_STATES,
                          config.PRETRAINED_VECTOR_SIZE),
                self.activation,
                nn.Linear(config.PRETRAINED_VECTOR_SIZE, config.PRETRAINED_VECTOR_SIZE)
            )
        # * -----------------------------------------------------------
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
        self.metric_3 = nn.Linear(in_feats, n_hidden)
        self.metric_4 = nn.Linear(n_hidden, n_hidden)

        l = []
        for obj in config.all_objects:
            l.append(config.object2vec[obj])
        object_vec = torch.tensor(l, dtype=torch.float32)
        if self.config.device is not None:
            object_vec = object_vec.to(self.config.device)
        self.register_buffer('object_vec', object_vec)

    def _model_device(self):
        return next(self.parameters()).device

    def _ensure_graph_device(self, graph, device):
        if hasattr(graph, 'to'):
            graph = graph.to(device)
        if hasattr(graph, 'ndata'):
            for key, value in list(graph.ndata.items()):
                if torch.is_tensor(value) and value.device != device:
                    graph.ndata[key] = value.to(device)
        return graph

    def _encode_graph(self, graph):
        hidden = graph.ndata['feat']
        for layer in self.layers_1:
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
        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.fc3(action)
        action = F.softmax(action, dim=1)
        ind_max_action = torch.argmax(action, dim=1, keepdim=True)
        one_hot_action = torch.zeros_like(action)
        one_hot_action.scatter_(1, ind_max_action, 1.0)

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
                torch.cat([pred2_expand, object_embed, pred1_output.unsqueeze(-1)], dim=2)
            )
        )
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = torch.sigmoid(self.q3_object(pred2_object).squeeze(-1))

        pred2_state = self.activation(self.q1_state(pred2_input))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = torch.sigmoid(self.q3_state(pred2_state))
        pred2_output = torch.cat([pred2_object, pred2_state], dim=1)
        return torch.cat((action, pred1_output, pred2_output), dim=1)

    def forward(self, g, goalVec):
        # * g: Start state.
        # * goalVec: Final state.
        device = self._model_device()
        g = self._ensure_graph_device(g, device)
        goalVec = self._ensure_graph_device(goalVec, device)

        h_start = self._encode_graph(g)
        h_final = self._encode_graph(goalVec)
        scene_embedding = torch.abs(h_final - h_start)
        scene_embedding = self.scene_fc_1(scene_embedding)
        scene_embedding, batch_size = self._reshape_objects(scene_embedding)
        final_to_decode = self.scene_fc_2(scene_embedding.transpose(1, 2)).squeeze(-1)

        predicted_actions = self._decode_outputs(final_to_decode)
        if batch_size == 1:
            return predicted_actions.squeeze(0)
        return predicted_actions
