# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File             : graph_dataset.py
@Project          : tango_i
@Time             : 2021/11/22 9:00
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/04/30
@Desciption       : None   
"""
import os
import pickle
import random
import torch
from tqdm import tqdm

from src.utils.misc import convertToDGLGraph
from src.tal.action_proposal_network import action2vec_cons
from src.utils.graph import action2vec_cons_wide, get_data_files, merge_datapoint, \
    convert_goal_json_to_vec, getDGLSequence, convert_symbolicActions_to_goal_json


class GraphDataset():

    def __init__(
            self,
            config,
            graphs_dir,
            node_sequences_path,
            DATA_NUM=None,
            GOAL_OBJ_VEC=False,
            DATA_ARGUMENT=False
    ):
        self.config = config
        self.graphs_dir = graphs_dir
        self.node_sequences_path = node_sequences_path
        self.DATA_NUM = DATA_NUM
        self.GOAL_OBJ_VEC = GOAL_OBJ_VEC
        self.DATA_ARGUMENT = DATA_ARGUMENT

        self.embedding = config.embeddings
        self.features_dim = self.config.features_dim

        # * Data containers.
        self.total_dps = []
        self.hl_actions = []
        self.action2vecs = []
        self.goal_jsons = []
        self.goal2vec = []
        self.goalObjects2vecs = []

        # * Load graph data.
        self.graphs_by_path = {}
        self.graphs_by_key = {}
        self.graphs_by_world = {}
        graph_paths = get_data_files(self.graphs_dir, data_format='.graph')
        for g_path in graph_paths:
            world_name = os.path.split(os.path.split(g_path)[0])[1]
            abs_path = os.path.abspath(g_path)
            graph_file = os.path.basename(g_path)
            with open(g_path, 'rb') as f:
                DG = pickle.load(f)
                self.graphs_by_path[abs_path] = DG
                self.graphs_by_key[f"{world_name}/{graph_file}"] = DG
                if world_name not in self.graphs_by_world:
                    self.graphs_by_world[world_name] = DG

        # * Load node sequences.
        with open(self.node_sequences_path, 'rb') as f:
            self.node_sequences = pickle.load(f)
        if self.DATA_NUM is not None:
            # random.shuffle(self.node_sequences)
            self.node_sequences = self.node_sequences[:self.DATA_NUM]

        self._shuffle()  # * Only once.

        # * -----------------------------------------------------------
        # * Pre-processing datapoint.
        # * Change 'End' state to dgl graph and save to the directed graph.
        if not DATA_ARGUMENT:
            print('Pre-processing datapoint...')
            for tmp_DG in self.graphs_by_path.values():
                nodes_list = list(tmp_DG.nodes)
                for node_id in tqdm(nodes_list):
                    # for node_id in nodes_list:
                    # * Data format: DG.nodes[node_id]
                    # * Define: collect_data.py
                    # * {
                    # *     'state': <src.datapoint.Datapoint object at 0x0000025D4711F2C8>,
                    # *     'id': 0,
                    # *     'world_state': 1,
                    # *     'pre_constraints': {},
                    # *     'child_actions': [{'name': 'moveTo', 'args': ['bottle_gray']}]
                    # * }
                    tmp_dp = tmp_DG.nodes[node_id]['state']
                    for i in range(len(tmp_dp.metrics)):
                        if tmp_dp.actions[i] == 'End':
                            dgl_graph = convertToDGLGraph(
                                config,
                                tmp_dp.getGraph(i, embeddings=self.embedding)['graph_' + str(i)],
                                False,
                                -1
                            )
                            tmp_DG.nodes[node_id]['dgl_graph'] = dgl_graph

        print('Pre-processing goal data...')
        self.preprocess_data()

    def __len__(self):
        if self.DATA_NUM is not None:
            return self.DATA_NUM
        else:
            return len(self.node_sequences)

    def _resolve_graph(self, node_sequence):
        graph_path = node_sequence.get('graph_path')
        if graph_path is not None:
            abs_graph_path = os.path.abspath(graph_path)
            if abs_graph_path in self.graphs_by_path:
                return self.graphs_by_path[abs_graph_path]

        world_name = node_sequence['world_name']
        graph_file = node_sequence.get('graph_file')
        if graph_file is not None:
            graph_key = f"{world_name}/{graph_file}"
            if graph_key in self.graphs_by_key:
                return self.graphs_by_key[graph_key]

        return self.graphs_by_world[world_name]

    def __getitem__(self, idx):
        # * Get node sequence.
        node_sequence = self.node_sequences[idx]
        world_name = node_sequence['world_name']
        node_seq = node_sequence['nodes']
        DG = self._resolve_graph(node_sequence)
        # * Get data.
        total_dp = self.total_dps[idx]
        hl_action = self.hl_actions[idx]
        action2vec = self.action2vecs[idx]
        goal_json = self.goal_jsons[idx]
        goal2vec = self.goal2vec[idx]
        goalObjects2vec = self.goalObjects2vecs[idx]

        # * -----------------------------------------------------------
        # * Get node's states(datapoint).
        # # * Each node use 'End' datapoint as state.
        start_node = DG.nodes[node_seq[0]]['state']
        dgl_graphs = []
        for node_id in node_seq[:-1]:  # * Without last state.
            try:
                dgl_graph = DG.nodes[node_id]['dgl_graph']
            except:
                dgl_graphs = []
                break
            dgl_graphs.append(dgl_graph)

        # * Get edge's actions.
        # hl_action = []
        # for i in range(len(node_seq) - 1):
        #     action = DG[node_seq[i]][node_seq[i + 1]]['action']
        #     hl_action.append(action['actions'])

        # * Convert datapoint to graphSeq and actionSeq.
        if len(dgl_graphs) != 0:
            graph = getDGLSequence(self.config, total_dp, self.embedding, hl_action,
                                   goal_json['goal-objects'],
                                   DGL_GRAPHS=dgl_graphs, DATA_ARGUMENT=self.DATA_ARGUMENT)
        else:
            graph = getDGLSequence(self.config, total_dp, self.embedding, hl_action,
                                   goal_json['goal-objects'],
                                   DATA_ARGUMENT=self.DATA_ARGUMENT)
        goal_num, world_num, toolSeq, (actionSeq, graphSeq), time = graph

        # * Trans data to device.
        if (len(dgl_graphs) == 0) and (self.config.device is not None):
            if len(dgl_graphs) == 0:
                graphSeq = [graph.to(self.config.device) for graph in graphSeq]

        try:
            assert len(graphSeq) == len(actionSeq), 'len(graphSeq) != len(actionSeq)'
        except:
            print('node sequence')
            print(node_seq)
            print('len(graphSeq): {}'.format(len(graphSeq)))
            print('len(actionSeq): {}'.format(len(actionSeq)))
            print(actionSeq)

        return graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node

    def _shuffle(self):
        random.shuffle(self.node_sequences)

    def preprocess_data(self):
        for idx, node_sequence in enumerate(tqdm(self.node_sequences)):
            # * Get node sequence.
            world_name = node_sequence['world_name']
            node_seq = node_sequence['nodes']
            # * Get graph.
            DG = self._resolve_graph(node_sequence)

            # * -------------------------------------------------------
            # * Get node's states(datapoint).
            # * Each node use 'End' datapoint as state.
            state_list = []
            dgl_graphs = []
            for node_id in node_seq[:-1]:
                state_list.append(DG.nodes[node_id]['state'])
                try:
                    dgl_graph = DG.nodes[node_id]['dgl_graph']
                    dgl_graphs.append(dgl_graph)
                except:
                    pass
            total_dp = merge_datapoint(state_list)  # * Merge node state.
            self.total_dps.append(total_dp)

            # * -------------------------------------------------------
            # * Get edge's actions.
            # * SymbolicActions stored in datapoint is the action from
            # * a parent node of this node.
            # * For the node which has only one parent node,
            # * hl_actions == total_dp.symbolicActions, but when we
            # * add graph's edges, the node may has many parent,
            # * so we must use actions stored in the edge.
            hl_action = []
            action_vecs = []
            for i in range(len(node_seq) - 1):
                action = DG[node_seq[i]][node_seq[i + 1]]['action']
                hl_action.append(action['actions'])
                action_vec = action2vec_cons(self.config, action['actions'][0],
                                             self.config.num_objects,
                                             len(self.config.possibleStates))
                if self.config.device is not None:
                    action_vec = action_vec.to(self.config.device)
                action_vecs.append(action_vec)
            self.hl_actions.append(hl_action)
            self.action2vecs.append(action_vecs)

            # * Generate goal json and convert to goal vector.
            goal_json = convert_symbolicActions_to_goal_json(self.config, hl_action)
            goal2vec, goalObjects2vec = convert_goal_json_to_vec(self.config, goal_json,
                                                                 self.GOAL_OBJ_VEC)
            # * Trans data to device.
            if self.config.device is not None:
                goal2vec = goal2vec.to(self.config.device)
                if self.GOAL_OBJ_VEC:
                    goalObjects2vec = goalObjects2vec.to(self.config.device)
            self.goal_jsons.append(goal_json)
            self.goal2vec.append(goal2vec)
            self.goalObjects2vecs.append(goalObjects2vec)


class ActionDataset():

    def __init__(self, config, graphs_dir, device=None, width=1):
        super(ActionDataset, self).__init__()
        self.config = config
        self.device = device
        self.graphs_dir = graphs_dir
        self.graph_paths = get_data_files(self.graphs_dir, data_format='.graph')
        self.width = width
        # self.features = self.data[0][0].ndata['feat'].shape[1]
        self.embedding = config.embeddings
        # * size:3, pos:3, orn:4
        self.features_dim = self.config.PRETRAINED_VECTOR_SIZE + self.config.N_STATES + 10
        # * Load graph data.
        self.graphs = {}
        graph_paths = get_data_files(self.graphs_dir, data_format='.graph')
        for g_path in graph_paths:
            world_name = os.path.split(os.path.split(g_path)[0])[1]
            with open(g_path, 'rb') as f:
                DG = pickle.load(f)
                self.graphs[world_name] = DG
        self.node_sequences = self.generate_node_list()
        self.dataset_len = len(self.node_sequences)

    def __len__(self):
        return len(self.node_sequences)

    def shuffle(self):
        random.shuffle(self.node_sequences)

    def generate_node_list(self):
        node_sequences = []
        for world_name, DG in self.graphs.items():
            node_list = list(DG.nodes)
            for id in node_list[1:]:
                node_sequences.append({'world_name': world_name, 'nodes': id})
        return node_sequences

    def __getitem__(self, idx):
        world_name = self.node_sequences[idx]['world_name']
        node_id = self.node_sequences[idx]['nodes']
        DG = self.graphs[world_name]  # * Get graph.
        data = DG.nodes[node_id]['state']
        start_index = data.actions.index('Start')
        start_dgl = convertToDGLGraph(
            self.config, data.getGraph(start_index, embeddings=self.config.embeddings)[
                'graph_' + str(start_index)], False, -1)
        end_index = data.actions.index('End')
        end_dgl = convertToDGLGraph(
            self.config,
            data.getGraph(end_index, embeddings=self.config.embeddings)['graph_' + str(end_index)],
            False,
            -1
        )
        action = data.symbolicActions[0][0]
        action_embed = action2vec_cons_wide(self.config, action, self.config.num_objects,
                                            len(self.config.possibleStates), width=self.width)

        if self.device is not None:
            return start_dgl.to(self.device), end_dgl.to(self.device), action, action_embed.to(
                self.device)
        else:
            return start_dgl, end_dgl, action, action_embed


class GraphDataset_State():
    """Goal vector representation: state"""

    def __init__(
            self,
            config,
            graphs_dir,
            node_sequences_path,
            DATA_NUM=None,
            GOAL_OBJ_VEC=False,
            DATA_ARGUMENT=False
    ):
        self.config = config
        self.graphs_dir = graphs_dir
        self.node_sequences_path = node_sequences_path
        self.DATA_NUM = DATA_NUM
        self.embedding = config.embeddings
        self.features_dim = self.config.features_dim
        self.GOAL_OBJ_VEC = GOAL_OBJ_VEC
        self.DATA_ARGUMENT = DATA_ARGUMENT

        # * Data containers.
        self.total_dps = []
        self.hl_actions = []
        self.action2vecs = []
        self.goal_jsons = []
        self.goal2vec = []
        self.goalObjects2vecs = []

        # * Load graph data.
        self.graphs_by_path = {}
        self.graphs_by_key = {}
        self.graphs_by_world = {}
        graph_paths = get_data_files(self.graphs_dir, data_format='.graph')
        for g_path in graph_paths:
            world_name = os.path.split(os.path.split(g_path)[0])[1]
            abs_path = os.path.abspath(g_path)
            graph_file = os.path.basename(g_path)
            with open(g_path, 'rb') as f:
                DG = pickle.load(f)
                self.graphs_by_path[abs_path] = DG
                self.graphs_by_key[f"{world_name}/{graph_file}"] = DG
                if world_name not in self.graphs_by_world:
                    self.graphs_by_world[world_name] = DG

        # * Load node sequences.
        with open(self.node_sequences_path, 'rb') as f:
            self.node_sequences = pickle.load(f)

        self._shuffle()  # * Only once.

        if DATA_NUM is not None:
            self.node_sequences = self.node_sequences[:DATA_NUM]

        # * -----------------------------------------------------------
        # * Pre-processing datapoint.
        # * Change 'End' state to dgl graph and save to the directed graph.
        if not DATA_ARGUMENT:
            print('Pre-processing datapoint...')
            for tmp_DG in self.graphs_by_path.values():
                nodes_list = list(tmp_DG.nodes)
                for node_id in tqdm(nodes_list):
                    # for node_id in nodes_list:
                    # * Data format: DG.nodes[node_id]
                    # * Define: collect_data.py
                    # * {
                    # *     'state': <src.datapoint.Datapoint object at 0x0000025D4711F2C8>,
                    # *     'id': 0,
                    # *     'world_state': 1,
                    # *     'pre_constraints': {},
                    # *     'child_actions': [{'name': 'moveTo', 'args': ['bottle_gray']}]
                    # * }
                    tmp_dp = tmp_DG.nodes[node_id]['state']
                    for i in range(len(tmp_dp.metrics)):
                        if tmp_dp.actions[i] == 'End':
                            dgl_graph = convertToDGLGraph(
                                config,
                                tmp_dp.getGraph(i, embeddings=self.embedding)['graph_' + str(i)],
                                False,
                                -1
                            )
                            tmp_DG.nodes[node_id]['dgl_graph'] = dgl_graph

        print('Pre-processing goal data...')
        self.preprocess_data()

    def __len__(self):
        if self.DATA_NUM is not None:
            return self.DATA_NUM
        else:
            return len(self.node_sequences)

    def _resolve_graph(self, node_sequence):
        graph_path = node_sequence.get('graph_path')
        if graph_path is not None:
            abs_graph_path = os.path.abspath(graph_path)
            if abs_graph_path in self.graphs_by_path:
                return self.graphs_by_path[abs_graph_path]

        world_name = node_sequence['world_name']
        graph_file = node_sequence.get('graph_file')
        if graph_file is not None:
            graph_key = f"{world_name}/{graph_file}"
            if graph_key in self.graphs_by_key:
                return self.graphs_by_key[graph_key]

        return self.graphs_by_world[world_name]

    def __getitem__(self, idx):
        # * Get node sequence.
        node_sequence = self.node_sequences[idx]
        world_name = node_sequence['world_name']
        node_seq = node_sequence['nodes']
        DG = self._resolve_graph(node_sequence)
        # * Get data.
        total_dp = self.total_dps[idx]
        hl_action = self.hl_actions[idx]
        action2vec = self.action2vecs[idx]
        goal_json = self.goal_jsons[idx]
        goal2vec = self.goal2vec[idx]
        goalObjects2vec = self.goalObjects2vecs[idx]

        # * -----------------------------------------------------------
        # * Get node's states(datapoint).
        # * Each node use 'End' datapoint as state.
        start_node = DG.nodes[node_seq[0]]['state']
        dgl_graphs = []

        for node_id in node_seq:
            try:
                dgl_graph = DG.nodes[node_id]['dgl_graph']
            except:
                dgl_graphs = []
                break
            dgl_graphs.append(dgl_graph)

        # * -----------------------------------------------------------
        # * Get edge's actions.
        # * SymbolicActions stored in datapoint is the action from a
        # * parent node of this node.
        # * For the node which has only one parent node,
        # * hl_actions == total_dp.symbolicActions, but when we
        # * add graph's edges, the node may has many parent,
        # * so we must use actions stored in the edge.

        # hl_actions = []
        # for i in range(len(node_seq) - 1):
        #     action = DG[node_seq[i]][node_seq[i + 1]]['action']
        #     hl_actions.append(action['actions'])

        # * Convert datapoint to graphSeq and actionSeq.
        if (len(dgl_graphs) != 0) and (not self.DATA_ARGUMENT):
            graph = getDGLSequence(self.config, total_dp, self.embedding, hl_action,
                                   goal_json['goal-objects'],
                                   DGL_GRAPHS=dgl_graphs, DATA_ARGUMENT=self.DATA_ARGUMENT)
        else:
            graph = getDGLSequence(self.config, total_dp, self.embedding, hl_action,
                                   goal_json['goal-objects'],
                                   DATA_ARGUMENT=self.DATA_ARGUMENT)
        goal_num, world_num, toolSeq, (actionSeq, graphSeq), time = graph

        goal2vec = graphSeq[-1]
        graphSeq = graphSeq[:-1]
        try:
            assert len(graphSeq) == len(actionSeq), 'len(graphSeq) != len(actionSeq)'
        except:
            print('node sequence')
            print(node_seq)
            print('len(graphSeq): {}'.format(len(graphSeq)))
            print('len(actionSeq): {}'.format(len(actionSeq)))
            print(actionSeq)
        return graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node

    def _shuffle(self):
        random.shuffle(self.node_sequences)

    def preprocess_data(self):
        for node_sequence in tqdm(self.node_sequences):
            # * Get node sequence.
            world_name = node_sequence['world_name']
            node_seq = node_sequence['nodes']
            # * Get graph.
            DG = self._resolve_graph(node_sequence)

            # * -------------------------------------------------------
            # * Get node's states(datapoint).
            # * Each node use 'End' datapoint as state.
            state_list = []
            dgl_graphs = []
            # for node_id in node_seq[:-1]:
            for node_id in node_seq:
                state_list.append(DG.nodes[node_id]['state'])
                try:
                    dgl_graph = DG.nodes[node_id]['dgl_graph']
                    dgl_graphs.append(dgl_graph)
                except:
                    pass
            total_dp = merge_datapoint(state_list)  # * Merge node state.
            self.total_dps.append(total_dp)

            # * -------------------------------------------------------
            # * Get edge's actions.
            # * SymbolicActions stored in datapoint is the action from
            # * a parent node of this node.
            # * For the node which has only one parent node,
            # * hl_actions == total_dp.symbolicActions, but when we
            # * add graph's edges, the node may has many parent,
            # * so we must use actions stored in the edge.
            hl_action = []
            action_vecs = []
            for i in range(len(node_seq) - 1):
                action = DG[node_seq[i]][node_seq[i + 1]]['action']
                hl_action.append(action['actions'])
                action_vec = action2vec_cons(self.config, action['actions'][0],
                                             self.config.num_objects,
                                             len(self.config.possibleStates))
                # print(action_vec.shape)
                action_vecs.append(action_vec)
            self.hl_actions.append(hl_action)
            self.action2vecs.append(action_vecs)

            # * Generate goal json and convert to goal vector.
            goal_json = convert_symbolicActions_to_goal_json(self.config, hl_action)
            goal2vec, goalObjects2vec = convert_goal_json_to_vec(self.config, goal_json,
                                                                 self.GOAL_OBJ_VEC)

            self.goal_jsons.append(goal_json)
            self.goal2vec.append(goal2vec)
            self.goalObjects2vecs.append(goalObjects2vec)


class AFEPairDataset:
    """Expanded AFE pair samples: (state_A, state_B, action_AB)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AFETripletDataset:
    """Expanded AFE triplet samples: (state_A, state_B, state_C, action_AB, action_BC)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class AFEExpandedGraphDataset:
    """
    Expand GraphDataset_State sequences into pair/triplet AFE training samples once on CPU.
    """

    def __init__(self, sequence_dataset):
        self.sequence_dataset = sequence_dataset
        self.config = sequence_dataset.config
        self.pair_samples = []
        self.triplet_samples = []
        self._build_samples()

    def _clone_graph(self, graph):
        cloned = graph.to(graph.device)
        cloned.ndata = {key: value.clone() for key, value in graph.ndata.items()}
        cloned.nodes = type(graph.nodes)(cloned)
        return cloned

    def _clone_tensor(self, tensor):
        return tensor.clone() if torch.is_tensor(tensor) else tensor

    def _build_samples(self):
        for idx in tqdm(range(len(self.sequence_dataset)), desc='Expanding AFE samples', ncols=80):
            graph_seq, goal_graph, _, action_seq, action2vec_seq, _, _ = self.sequence_dataset[idx]
            states = list(graph_seq) + [goal_graph]
            if len(states) < 2 or len(action2vec_seq) == 0:
                continue

            if len(states) == 2:
                pair_sample = {
                    'state_a': self._clone_graph(states[0]),
                    'state_b': self._clone_graph(states[1]),
                    'action_ab': self._clone_tensor(action2vec_seq[0]).float(),
                    'action_ab_text': action_seq[0],
                    'sequence_index': idx,
                    'step_index': 0,
                }
                self.pair_samples.append(pair_sample)

            for step in range(len(states) - 2):
                triplet_sample = {
                    'state_a': self._clone_graph(states[step]),
                    'state_b': self._clone_graph(states[step + 1]),
                    'state_c': self._clone_graph(states[step + 2]),
                    'action_ab': self._clone_tensor(action2vec_seq[step]).float(),
                    'action_bc': self._clone_tensor(action2vec_seq[step + 1]).float(),
                    'action_ab_text': action_seq[step],
                    'action_bc_text': action_seq[step + 1],
                    'sequence_index': idx,
                    'step_index': step,
                }
                self.triplet_samples.append(triplet_sample)

    def get_pair_dataset(self):
        return AFEPairDataset(self.pair_samples)

    def get_triplet_dataset(self):
        return AFETripletDataset(self.triplet_samples)


class APNPairDataset:
    """Expanded APN pair samples: (state_a, goal_state, action_ab)."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class APNExpandedGraphDataset:
    """
    Expand GraphDataset_State sequences into independent APN training samples once on CPU.
    Each transition becomes one (current_state, goal_state, action) sample.
    """

    def __init__(self, sequence_dataset):
        self.sequence_dataset = sequence_dataset
        self.config = sequence_dataset.config
        self.samples = []
        self._build_samples()

    def _clone_graph(self, graph):
        cloned = graph.to(graph.device)
        cloned.ndata = {key: value.clone() for key, value in graph.ndata.items()}
        cloned.nodes = type(graph.nodes)(cloned)
        return cloned

    def _clone_tensor(self, tensor):
        return tensor.clone() if torch.is_tensor(tensor) else tensor

    def _build_samples(self):
        for idx in tqdm(range(len(self.sequence_dataset)), desc='Expanding APN samples', ncols=80):
            graph_seq, goal_graph, _, action_seq, action2vec_seq, _, _ = self.sequence_dataset[idx]
            if len(graph_seq) == 0 or len(action2vec_seq) == 0:
                continue

            for step in range(len(graph_seq)):
                self.samples.append({
                    'state_a': self._clone_graph(graph_seq[step]),
                    'goal_state': self._clone_graph(goal_graph),
                    'action_ab': self._clone_tensor(action2vec_seq[step]).float(),
                    'action_ab_text': action_seq[step],
                    'sequence_index': idx,
                    'step_index': step,
                })

    def get_pair_dataset(self):
        return APNPairDataset(self.samples)
