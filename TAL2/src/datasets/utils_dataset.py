import os
import json
import pickle
import dgl
import torch
from tqdm import tqdm
from sys import maxsize
import math


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def euler_to_quaternion(rpy):
    """Convert [roll, pitch, yaw] to quaternion [x, y, z, w]."""
    if rpy is None or len(rpy) != 3:
        raise ValueError("Euler angles must be a 3-element iterable [roll, pitch, yaw].")
    roll, pitch, yaw = [float(v) for v in rpy]
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def orientation_to_quaternion(orientation):
    if orientation is None:
        return [0.0, 0.0, 0.0, 1.0]
    if len(orientation) == 4:
        return [float(v) for v in orientation]
    if len(orientation) == 3:
        return euler_to_quaternion(orientation)
    return [0.0, 0.0, 0.0, 1.0]


############################ * DGL ############################

def getToolSequence(config, actionSeq):
    """
    Returns the sequence of tools that were used in the plan.
    """
    toolSeq = ['no-tool'] * len(actionSeq)
    currentTool = 'no-tool'
    for i in range(len(toolSeq) - 1, -1, -1):
        for obj in actionSeq[i]['args']:
            if obj in config.TOOLS2:
                currentTool = obj
                break
        toolSeq[i] = currentTool
    return toolSeq


def getGlobalID(config, dp):
    maxID = 0
    for i in dp.metrics[0].keys():
        maxID = max(maxID, config.object2idx[i])
    return maxID + 1


def convertToDGLGraph(config, graph_data, globalNode, goal_num, globalID, ignore: list = None):
    """
    Converts the graph from the datapoint into a DGL form of graph.
    """
    if ignore is None:
        ignore = []
    # * Make edge sets.
    close, inside, on, stuck = [], [], [], []
    closeToAgent = []
    for edge in graph_data['edges']:
        if edge['relation'] == 'Close':
            close.append((edge['from'], edge['to']))
            if edge['from'] == config.all_objects.index('husky'):
                closeToAgent.append(edge['to'])
        elif edge['relation'] == 'Inside':
            inside.append((edge['from'], edge['to']))
        elif edge['relation'] == 'On':
            on.append((edge['from'], edge['to']))
        elif edge['relation'] == 'Stuck':
            stuck.append((edge['from'], edge['to']))
    edgeDict = {
        ('object', 'Close', 'object'): close + [(i, i) for i in ignore],
        ('object', 'Inside', 'object'): inside,
        ('object', 'On', 'object'): on,
        ('object', 'Stuck', 'object'): stuck
    }
    if globalNode:
        globalList = []
        for i in range(globalID): globalList.append((i, globalID))
        edgeDict[('object', 'Global', 'object')] = globalList
    g = dgl.heterograph(edgeDict)
    # * Add node features
    n_nodes = len(config.all_objects)
    node_states = torch.zeros([n_nodes, config.N_STATES], dtype=torch.float)  # * State vector.
    node_vectors = torch.zeros([n_nodes, config.PRETRAINED_VECTOR_SIZE],
                               dtype=torch.float)  # * Fasttext embedding.
    node_size_and_pos = torch.zeros([n_nodes, 10], dtype=torch.float)  # * Size and position.
    node_in_goal = torch.zeros([n_nodes, 1], dtype=torch.float)  # * Object in goal.
    node_close_agent = torch.zeros([n_nodes, 1], dtype=torch.float)  # * Close to husky.
    for i, node in enumerate(graph_data['nodes']):
        states = node['states']
        node_id = node['id']

        if abs(node['position'][0][2]) >= 2.0:
            node['position'] = list(node['position'])
            node['position'][0] = list(node['position'][0])
            node['position'][0][2] = 0

        if node_id in ignore: continue
        for state in states:
            idx = config.state2indx[state]
            node_states[node_id, idx] = 1
        node_vectors[node_id] = torch.tensor(node['vector'], dtype=torch.float32)
        tmp_pos = orientation_to_quaternion(node['position'][1])
        node_size_and_pos[node_id] = torch.tensor(
            list(node['size']) + list(node['position'][0]) + tmp_pos, dtype=torch.float32
        )
        node_in_goal[node_id] = 0
        try:
            if (goal_num is not None) and (node['name'] in config.goalObjects[goal_num]):
                node_in_goal[node_id] = 1
        except:
            pass

    for node in closeToAgent: node_close_agent[node] = 1
    g.ndata['close'] = node_close_agent
    g.ndata['feat'] = torch.cat((node_vectors, node_states, node_size_and_pos), 1)
    return g


def getDGLGraph(
        config,
        pathToDatapoint,
        globalNode,
        ignoreNoTool,
        e,
        INT_TYPE_GOAL=False,
        DATAPOINT_DIRECT=False
):
    """
    Returns the intital state DGL graph from the path to the
    given datapoint.
    """
    if DATAPOINT_DIRECT:
        datapoint = pathToDatapoint
    else:
        datapoint = pickle.load(open(pathToDatapoint, 'rb'))
    datapoint.config = config  # * Add new attributes.

    time = datapoint.totalTime()
    tools = datapoint.getTools(not ignoreNoTool)
    if ignoreNoTool and len(tools) == 0: return None
    if INT_TYPE_GOAL:
        goal_num = datapoint.goal
    else:
        goal_num = _safe_numeric_suffix(datapoint.goal, default=0)
    world_num = _safe_numeric_suffix(datapoint.world, default=0)
    # * Initial Graph
    graph_data = datapoint.getGraph(embeddings=e)['graph_0']
    g = convertToDGLGraph(config, graph_data, globalNode, goal_num,
                          getGlobalID(config, datapoint) if globalNode else -1)

    return (goal_num, world_num, tools, g, time)


def getDGLSequence(
        config,
        pathToDatapoint,
        globalNode,
        ignoreNoTool,
        e,
        INT_TYPE_GOAL=False,
        DATAPOINT_DIRECT=False
):
    """
    Returns the entire sequence of graphs and actions from the plan in
    the provided datapoint.
    """
    if DATAPOINT_DIRECT:
        datapoint = pathToDatapoint
    else:
        datapoint = pickle.load(open(pathToDatapoint, 'rb'))
    datapoint.config = config  # * Add new attributes

    time = datapoint.totalTime()
    tools = datapoint.getTools(not ignoreNoTool)
    if ignoreNoTool and len(tools) == 0: return None
    if INT_TYPE_GOAL:
        goal_num = datapoint.goal
    else:
        goal_num = _safe_numeric_suffix(datapoint.goal, default=0)
    world_num = _safe_numeric_suffix(datapoint.world, default=0)
    actionSeq = []
    graphSeq = []
    for action in datapoint.symbolicActions:
        if not (str(action[0]) == 'E' or str(action[0]) == 'U'): actionSeq.append(action[0])
    for i in range(len(datapoint.metrics)):
        if datapoint.actions[i] == 'Start' and config.AUGMENTATION == 1:
            graphSeq.append(
                convertToDGLGraph(
                    config,
                    datapoint.getGraph(i, embeddings=e)['graph_' + str(i)],
                    globalNode,
                    goal_num,
                    getGlobalID(config, datapoint) if globalNode else -1)
            )
        elif datapoint.actions[i] == 'Start' and config.AUGMENTATION > 1:
            ignoreList, graph = datapoint.getAugmentedGraph(i, embeddings=e)
            graphSeq.append(
                convertToDGLGraph(
                    config,
                    graph['graph_' + str(i)],
                    globalNode,
                    goal_num,
                    getGlobalID(config, datapoint) if globalNode else -1,
                    ignore=ignoreList
                )
            )

    assert len(actionSeq) == len(graphSeq)
    toolSeq = getToolSequence(config, actionSeq)
    return (goal_num, world_num, toolSeq, (actionSeq, graphSeq), time)


class DGLDataset():
    """
    Class which contains the entire data.
    For any i,
    self.graphs[i][-2] -> Every element of this object is a datapoint.
        If sequence is true this is a graph sequence, otherwise it
        contains the initial state of the datapoint.

    self.graphs[i][-3] -> The tools used in the datapoint.
        If sequence is true contains the sequence of next most recent tool.
        Otherwise, if sequence is false, contains a list of tools used in the plan.
    """

    def __init__(
            self,
            config,
            program_dir,
            augmentation=50,
            globalNode=False,
            ignoreNoTool=False,
            sequence=False,
            embedding='conceptnet',
            INT_TYPE_GOAL=False
    ):
        graphs = []
        # with open('src/envs/jsons/embeddings/' + embedding + '.vectors') as handle:
        #     e = json.load(handle)
        e = config.embeddings

        self.goal_scene_to_tools = {}
        self.min_time = {}
        all_files = list(os.walk(program_dir))
        for path, dirs, files in tqdm(all_files):
            if (len(files) > 0):
                for file in files:
                    file_path = path + '/' + file
                    for i in range(augmentation):
                        if not sequence:
                            graph = getDGLGraph(config, file_path, globalNode, ignoreNoTool, e,
                                                INT_TYPE_GOAL=INT_TYPE_GOAL)
                        else:
                            graph = getDGLSequence(config, file_path, globalNode, ignoreNoTool, e,
                                                   INT_TYPE_GOAL=INT_TYPE_GOAL)
                        if graph:
                            graphs.append(graph)
                            tools = graphs[-1][2]
                            goal_num = graphs[-1][0]
                            world_num = graphs[-1][1]
                            if (goal_num, world_num) not in self.goal_scene_to_tools:
                                self.goal_scene_to_tools[(goal_num, world_num)] = []
                                self.min_time[(goal_num, world_num)] = maxsize
                            for tool in tools:
                                if tool not in self.goal_scene_to_tools[(goal_num, world_num)]:
                                    self.goal_scene_to_tools[(goal_num, world_num)].append(tool)
                            self.min_time[(goal_num, world_num)] = min(
                                self.min_time[(goal_num, world_num)],
                                graphs[-1][4])
        self.graphs = graphs
        self.features = self.graphs[0][3].ndata['feat'].shape[1] if not sequence else \
            self.graphs[0][3][1][0].ndata['feat'].shape[1]
        self.num_objects = len(config.all_objects)
        if globalNode: self.num_objects -= 1


class TestDataset():
    def __init__(self, config, program_dir, augmentation=1):
        graphs = []
        all_files = os.walk(program_dir)
        for path, dirs, files in tqdm(all_files):
            if (len(files) > 0):
                for file in files:
                    file_path = path + '/' + file
                    for i in range(augmentation):
                        with open(file_path, 'r') as handle:
                            graph = json.load(handle)
                        g = convertToDGLGraph(
                            config, graph['graph_0'], False, graph['goal_num'], -1
                        )
                        graphs.append(
                            (
                                graph['goal_num'],
                                graph['world_num'],
                                int(path[-1]),
                                graph['tools'],
                                convertToDGLGraph(config, graph['graph_0'], False,
                                                  graph['goal_num'], -1),
                                graph['tool_embeddings'], graph['object_embeddings']
                            )
                        )
        self.graphs = graphs
