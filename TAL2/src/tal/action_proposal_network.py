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

        # h_final = goalVec.ndata['feat']
        # for i, layer in enumerate(self.layers_2):
        #     h_final = layer(goalVec, h_final)
        # metric_part_final = goalVec.ndata['feat']
        # metric_part_final = self.activation(self.metric_3(metric_part_final))
        # metric_part_final = self.activation(self.metric_4(metric_part_final))
        # h_final = torch.cat([h_final, metric_part_final], dim=1)

        # * Use same layer as h_start.
        h_final = goalVec.ndata['feat']
        for i, layer in enumerate(self.layers_1):
            h_final = layer(goalVec, h_final)
        metric_part_final = goalVec.ndata['feat']
        metric_part_final = self.activation(self.metric_1(metric_part_final))
        metric_part_final = self.activation(self.metric_2(metric_part_final))
        h_final = torch.cat([h_final, metric_part_final], dim=1)

        # # * Cat
        # scene_embedding = torch.cat([h_start, h_final], 1)  # * [39, 512]

        # # * Delta state
        # scene_embedding = h_final - h_start  # * Delta feature.
        # scene_embedding = self.scene_fc_1(scene_embedding)  # * [39, 512]

        # * -----------------------------------------------------------
        # # * Cat
        # scene_embedding = torch.cat([h_start, h_final], 1)  # * [39, 512]

        # * Delta feature
        # scene_embedding = h_final - h_start
        scene_embedding = torch.abs(h_final - h_start)
        scene_embedding = self.scene_fc_1(scene_embedding)  # * [39, 512]
        # * -----------------------------------------------------------

        final_to_decode = self.scene_fc_2(scene_embedding.t())  # * [512, 1]
        final_to_decode = final_to_decode.t()

        action = self.activation(self.fc1(final_to_decode))
        action = self.activation(self.fc2(action))
        action = self.fc3(action)
        action = F.softmax(action, dim=1)
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
        pred1_output = pred1_object
        pred1_output = torch.sigmoid(pred1_object)

        # * Predicting the second argument of the action
        pred2_input = torch.cat([final_to_decode, one_hot_action], 1)
        pred2_object = self.activation(self.q1_object(
            torch.cat([pred2_input.view(-1).repeat(self.n_objects).view(self.n_objects, -1),
                       self.activation(self.embed(self.object_vec)),
                       pred1_output.view(self.n_objects, 1)], 1)))
        pred2_object = self.activation(self.q2_object(pred2_object))
        pred2_object = self.q3_object(pred2_object)
        pred2_object = torch.sigmoid(pred2_object)

        pred2_state = self.activation(self.q1_state(pred2_input))
        pred2_state = self.activation(self.q2_state(pred2_state))
        pred2_state = self.q3_state(pred2_state)
        pred2_state = torch.sigmoid(pred2_state)
        pred2_output = torch.cat([pred2_object.view(1, -1), pred2_state], 1)

        predicted_actions = torch.cat((action, pred1_output.view(1, -1), pred2_output.view(1, -1)),
                                      1).flatten()

        return predicted_actions
