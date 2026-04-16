import torch
import warnings
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.baselines.plan_transformer.utils import test_policy_pt

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'PlanTransformer'
    config = EnvironmentConfig(args)
    config.context_len = 3

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/2/home/'
    train_data_path = './data/2/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    print('Train data num: {}'.format(len(train_dataset)))
    # val_data_path = './data/val_dataset.pkl'
    # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
    # print('Val data num: {}'.format(len(val_dataset)))
    test_data_path = './data/2/test_dataset.pkl'
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)
    print('Test data num: {}'.format(len(test_dataset)))

    # * ----------------------------------------------------------------
    model_action_effect = None
    action_effect_features = None

    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model, optimizer, epoch, accuracy_list = load_model(config,
                                                        seqTool + model.name + '_Trained',
                                                        model)
    model = model.to(config.device)
    print('Model: {} | epoch: {}'.format(args.model_name, epoch))

    # * ----------------------------------------------------------------
    # * Policy test.
    # seeds = [0, 1, 42]
    seeds = [0]
    for seed in seeds:
        PCA_FLAG = True
        setup_seed(seed=seed)
        print('-----------------------------------------')
        print('Random seed: {}'.format(seed))
        print('-----------------------------------------')

        print('\n\nTraining set...')
        test_policy_pt(config, train_dataset, model, config.num_objects, TQDM=False)

        # print('\n\nVal set...')
        # test_policy_pt(config, val_dataset, model, config.num_objects, TQDM=False)

        print('\n\nTest set...')
        test_policy_pt(config, test_dataset, model, config.num_objects, TQDM=False)
