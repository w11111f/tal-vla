import torch
import random
import numpy as np
from src.envs.utils_env import convertToDGLGraph as cvtGraph


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def convertToDGLGraph(config, graph_data, globalNode, globalID, ignore: list = None):
    """ Converts the graph from the datapoint into a DGL form of graph."""
    return cvtGraph(config, graph_data, globalNode, globalID, ignore)

def generate_action_gt_tensor(config, action_seq: list):
    y_true = torch.zeros((1, config.action_num), device=config.device)
    for action in action_seq:
        idx = config.dataset_action_list.index(action)
        y_true[0][idx] = 1
    return y_true
