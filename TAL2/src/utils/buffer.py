"""
@File             : buffer.py
@Project          : TAL_2024
@Time             : 2024/11/08
@Author           : Xianqi ZHANG
@Last Modify Time : 2024/11/08
@Version          : 1.0  
@Desciption       : None   
"""
import os
import pickle
import random
import torch
from tqdm import tqdm
from dgl import DGLGraph
from src.utils.misc import convertToDGLGraph
from src.tal.action_proposal_network import action2vec_cons
from src.utils.graph import get_data_files, merge_datapoint, convert_goal_json_to_vec, \
    getDGLSequence, convert_symbolicActions_to_goal_json


class ReplayBuffer():
    """Goal vector representation: state"""

    def __init__(
            self,
            config,
            graphs_dir,
            node_sequences_path,
            gamma=0.1,
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
        self.gamma = gamma
        self.idx = 0

        # * Data containers.
        self.total_dps = []
        self.hl_actions = []
        self.action2vecs = []
        self.goal_jsons = []
        self.goal2vec = []
        self.goalObjects2vecs = []

        # * Load graph data.
        self.graphs = {}
        graph_paths = get_data_files(self.graphs_dir, data_format='.graph')
        for g_path in graph_paths:
            world_name = os.path.split(os.path.split(g_path)[0])[1]
            with open(g_path, 'rb') as f:
                DG = pickle.load(f)
                self.graphs[world_name] = DG

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
            for tmp_DG in self.graphs.values():
                nodes_list = list(tmp_DG.nodes)
                for node_id in tqdm(nodes_list):
                    tmp_dp = tmp_DG.nodes[node_id]['state']
                    for i in range(len(tmp_dp.metrics)):
                        if tmp_dp.actions[i] == 'End':
                            dgl_graph = convertToDGLGraph(
                                config,
                                tmp_dp.getGraph(i, embeddings=self.embedding)['graph_' + str(i)],
                                False,
                                -1
                            )
                            if self.config.device is not None:
                                dgl_graph = dgl_graph.to(self.config.device)
                            tmp_DG.nodes[node_id]['dgl_graph'] = dgl_graph

        print('Pre-processing goal data...')
        self.preprocess_data()

    def __len__(self):
        if self.DATA_NUM is not None:
            return self.DATA_NUM
        else:
            return len(self.node_sequences)

    def __getitem__(self, idx):
        # * Get node sequence.
        world_name = self.node_sequences[idx]['world_name']
        node_seq = self.node_sequences[idx]['nodes']
        DG = self.graphs[world_name]
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
        # * parent node of this node. For the node which has only one
        # * parent node, hl_actions == total_dp.symbolicActions, but
        # * when we add graph's edges, the node may has many parent, so
        # * we must use actions stored in the edge.
        # * hl_actions = []
        # for i in range(len(node_seq) - 1):
        #     action = DG[node_seq[i]][node_seq[i + 1]]['action']
        #     hl_actions.append(action['actions'])

        # * Convert datapoint to graphSeq and actionSeq.
        if (len(dgl_graphs) != 0) and (not self.DATA_ARGUMENT):
            graph = getDGLSequence(
                self.config, total_dp, self.embedding, hl_action, goal_json['goal-objects'],
                DGL_GRAPHS=dgl_graphs, DATA_ARGUMENT=self.DATA_ARGUMENT
            )
        else:
            graph = getDGLSequence(
                self.config, total_dp, self.embedding, hl_action, goal_json['goal-objects'],
                DATA_ARGUMENT=self.DATA_ARGUMENT
            )
        goal_num, world_num, toolSeq, (actionSeq, graphSeq), time = graph

        # * Trans data to device.
        if (len(dgl_graphs) == 0) and (self.config.device is not None):
            if len(dgl_graphs) == 0:
                graphSeq = [graph.to(self.config.device) for graph in graphSeq]

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

        # * -----------------------------------------------------------
        # * 1.
        # * Reward: reach goal -> reward = 1, else -> reward = 0
        # rewards = [0 for _ in range((len(graphSeq) - 1))] + [1]
        # rewards = torch.tensor(rewards).to(self.config.device)

        # * 2.
        distance = []
        s_g = goal2vec.ndata['feat']
        for i in range(len(graphSeq)):
            s_i = graphSeq[i].ndata['feat']
            dis = torch.sum(torch.abs(s_g - s_i))  # * [0, 100]
            # dis = self.gamma * dis
            distance.append(dis)

        min_distance = distance[0]
        rewards = []
        for i in range(1, len(distance)):
            r = 0.0
            if distance[i] < min_distance:
                # r = min(2, min_distance - distance[i])
                # r = max(min(2.0, min_distance - distance[i]), 1.0)  # * [1, 2]
                r = 1.0
                min_distance = distance[i]
            # else:
            #     r = 0.5
            # print('Reward: {}'.format(r))

            rewards.append(r)
        rewards.append(100)  # * Add reward for last action.
        assert len(rewards) == len(graphSeq)

        rewards = torch.tensor(rewards, device=self.config.device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # * dones.
        dones = [0 for _ in range((len(graphSeq) - 1))] + [100]
        assert len(dones) == len(graphSeq)
        dones = torch.tensor(dones).to(self.config.device)

        return graphSeq, goal2vec, goal_json, actionSeq, action2vec, \
            world_name, start_node, rewards, dones

    def _shuffle(self):
        random.shuffle(self.node_sequences)

    def preprocess_data(self):
        for node_sequence in tqdm(self.node_sequences):
            # * Get node sequence.
            world_name = node_sequence['world_name']
            node_seq = node_sequence['nodes']
            # * Get graph.
            DG = self.graphs[world_name]

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
            # * a parent node of this node. For the node which has only
            # * one parent node, hl_actions == total_dp.symbolicActions,
            # * but when we add graph's edges, the node may has many
            # * parent, so we must use actions stored in the edge.
            hl_action = []
            action_vecs = []
            for i in range(len(node_seq) - 1):
                action = DG[node_seq[i]][node_seq[i + 1]]['action']
                hl_action.append(action['actions'])
                action_vec = action2vec_cons(self.config,
                                             action['actions'][0],
                                             self.config.num_objects,
                                             len(self.config.possibleStates))
                if self.config.device is not None:
                    action_vec = action_vec.to(self.config.device)
                action_vecs.append(action_vec)
            self.hl_actions.append(hl_action)
            self.action2vecs.append(action_vecs)

            # # * Generate goal json and convert to goal vector.
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

    def sample(
            self,
            idx: int = None
    ) -> [list[DGLGraph], list[torch.Tensor], torch.Tensor, DGLGraph]:
        """
        Return:
            obs (graphSeq), action (action2vec), reward, task (goal2vec)
        Single action dim=111
            - act_name, 0:11
            - obj1, 11:47
            - obj2, 47:83
            - state, 83:111
        """
        # idx = random.randint(0, self.__len__() - 1) if idx is None else idx
        if idx is None:
            idx = self.idx
            self.idx += 1
            if self.idx >= len(self):
                self.idx = 0
        graphSeq, goal2vec, _, _, action2vec, _, _, rewards, _ = self.__getitem__(idx)
        obs = graphSeq
        action = action2vec
        task = goal2vec
        reward = rewards
        # assert len(rewards) == len(action)
        # reward = []
        # for i in range(len(action)):
        #     act = action[i]
        #     r = torch.zeros_like(act, device=self.config.device)
        #     r[act==1] = rewards[i]
        #     reward.append(r)
        return obs, action, reward, task
