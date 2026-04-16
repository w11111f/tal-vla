import pickle
import torch
import colorama
from src.utils.misc import setup_seed
from src.baselines.cql_dqn.utils import test_policy_graph_dataset_cql
from src.baselines.cql_dqn.agent import CQLAgent
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State

colorama.init()

if __name__ == '__main__':
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'DDQN'
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    with open('checkpoints/home/action_list_train_dataset.pkl', 'rb') as f:
        action_set = pickle.load(f)

    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    # val_data_path = './data/val_dataset.pkl'
    # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
    test_data_path = './data/test_dataset.pkl'
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

    agent = CQLAgent(config, action_set)

    print('\n\n//////////////////////////////////////////////////////')
    print('CQL')
    print('//////////////////////////////////////////////////////////')
    seeds = [0]
    for seed in seeds:
        PCA_FLAG = True
        setup_seed(seed=seed)

        print('\n\n-----------------------------------------')
        print('Random seed: {}'.format(seed))
        print('-----------------------------------------')

        print('\n\nTraining set.')
        test_policy_graph_dataset_cql(config, train_dataset, agent, TQDM=False)

        # print('\n\nVal set.')
        # test_policy_graph_dataset_cql(config, val_dataset, agent, TQDM=False)

        print('\n\nTest set.')
        test_policy_graph_dataset_cql(config, test_dataset, agent, TQDM=False)
