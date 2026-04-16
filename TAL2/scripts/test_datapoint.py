"""
@Project     ：TAL_2024
@File        ：test_datapoint.py
@Author      ：Xianqi-Zhang
@Date        ：2025/2/24
@Last        : 2025/2/24
@Description : 
"""
import os
import torch
import pickle
import colorama
import warnings
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.utils.misc import convertToDGLGraph
from src.utils.graph import get_data_files

colorama.init()
warnings.filterwarnings('ignore')


def main():
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)

    dataset_n = 2
    graphs_dir = './data/{}/home/'.format(dataset_n)
    train_data_path = './data/{}/train_dataset.pkl'.format(dataset_n)
    # train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    # print('Train data num: {}'.format(len(train_dataset)))

    graphs = {}  # * len(graphs): 1
    graph_paths = get_data_files(graphs_dir, data_format='.graph')
    for g_path in graph_paths:
        world_name = os.path.split(os.path.split(g_path)[0])[1]
        with open(g_path, 'rb') as f:
            DG = pickle.load(f)
            graphs[world_name] = DG

    for world_name, DG in graphs.items():
        # nodes_list = list(DG.nodes)
        nodes_list = [0]  # * Select node 0.
        for node_id in nodes_list:
            tmp_dp = DG.nodes[node_id]['state']
            for i in range(len(tmp_dp.metrics)):
                if tmp_dp.actions[i] == 'End':
                    dgl_graph = convertToDGLGraph(
                        config,
                        tmp_dp.getGraph(i, embeddings=config.embeddings)['graph_' + str(i)],
                        False,
                        -1
                    )
                    DG.nodes[node_id]['dgl_graph'] = dgl_graph

            print('==' * 20)
            # print(world_name)  # * world_home0
            print(len(DG.nodes))  # * 2995
            # print(DG.nodes[0]['dgl_graph'].ndata['feat'].shape)  # * (36, 338)
            # print(DG.nodes[0]['dgl_graph'].nodes['object'].data['close'].shape)  # * (36, 1)
            # print(DG.nodes[0]['dgl_graph'])
            # print(DG.nodes[0]['dgl_graph'].ntypes)  # * ['object']
            # print(DG.nodes[0]['dgl_graph'].etypes)  # * ['Close', 'Inside', 'On', 'Stuck']
            print('==' * 20)


if __name__ == '__main__':
    main()
