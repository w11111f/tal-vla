import pickle
import torch
import warnings
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_planning import test_policy_with_action_effect_features

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/home/'
    # train_data_path = './data/train_dataset.pkl'
    # train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    # train_data_num = len(train_dataset)
    # print('Train data num: {}'.format(train_data_num))
    # val_data_path = './data/val_dataset.pkl'
    # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
    test_data_path = './data/test_dataset.pkl'
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

    # * ----------------------------------------------------------------
    # * Create model and load parameters.
    # * 01.Action Feature Extractor: AFE
    model_action_effect = get_model(config, config.model_name, config.features_dim,
                                    config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action_effect, optimizer, epoch, accuracy_list = load_model(config,
                                                                      seqTool + model_action_effect.name + '_Trained',
                                                                      model_action_effect)
    print('Model: {} | epoch: {}'.format(model_action_effect.name, epoch))

    model_action_effect = model_action_effect.to(config.device)

    # * 02.Action Proposal Network: APN
    model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action, optimizer, epoch, accuracy_list = load_model(config,
                                                               seqTool + model_action.name + '_Trained',
                                                               model_action)
    print('Model: APN | epoch: {}'.format(epoch))

    model_action = model_action.to(config.device)

    # * ----------------------------------------------------------------
    # * Load features.
    # features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features.pkl'
    features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
    with open(features_save_path, 'rb') as f:
        action_effect_features = pickle.load(f)
    print('Action effect features num: {}'.format(len(action_effect_features['names'])))

    # * ----------------------------------------------------------------
    # * Policy test.

    # seeds = [0, 1, 42]
    seeds = [0]
    all_selected_pool = [(2, 2), (5, 5), (10, 10), (15, 15), (20, 20), (30, 30)]
    partially_selected_pool = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]

    # # * 01. w/ Early Stop, w/ PCA  (Same as 02. Without MCAS in ablation_study.py)
    # print('\n\n')
    # print('//' * 20)
    # print('01. w/ Early Stop, w/ PCA')
    # print('pool_sets: [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]')
    # print('//' * 20)
    #
    # for seed in seeds:
    #     PCA_FLAG = True
    #     setup_seed(seed=seed)
    #     for pool_set in partially_selected_pool:
    #         pool_set = [pool_set]
    #         print('\n\n-----------------------------------------')
    #         print('Random seed: {}'.format(seed))
    #         print('Pool: {}'.format(pool_set))
    #         print('PCA: {}'.format(PCA_FLAG))
    #         print('-----------------------------------------')
    #         test_policy_with_action_effect_features(config,
    #                                                 test_dataset,
    #                                                 model_action,
    #                                                 model_action_effect,
    #                                                 action_effect_features,
    #                                                 multiscale_pool=pool_set,
    #                                                 TQDM=False,
    #                                                 ONLY_ACTION_MODEL=False,
    #                                                 ONLY_FEATURE_MODEL=False,
    #                                                 WITH_PCA=PCA_FLAG,
    #                                                 STATE_FORMAT_GOAL=True,
    #                                                 INIT_DATAPOINT=True,
    #                                                 IGNORE_ACTION_MODEL=True)

    # * 02. w/ Early Stop, w/o PCA
    print('\n\n')
    print('//' * 20)
    print('02. w/ Early Stop, w/o PCA')
    print('pool_sets: [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]')
    print('//' * 20)

    for seed in seeds:
        PCA_FLAG = False  # * !!!
        setup_seed(seed=seed)
        for pool_set in partially_selected_pool:
            pool_set = [pool_set]
            print('\n\n-----------------------------------------')
            print('Random seed: {}'.format(seed))
            print('Pool: {}'.format(pool_set))
            print('PCA: {}'.format(PCA_FLAG))
            print('-----------------------------------------')
            test_policy_with_action_effect_features(config,
                                                    test_dataset,
                                                    model_action,
                                                    model_action_effect,
                                                    action_effect_features,
                                                    multiscale_pool=pool_set,
                                                    TQDM=False,
                                                    ONLY_ACTION_MODEL=False,
                                                    ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG,
                                                    STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    IGNORE_ACTION_MODEL=True)

    # * 03. w/o Early Stop, w/ PCA
    print('\n\n')
    print('//' * 20)
    print('03. w/o Early Stop, w/ PCA')
    print('pool_sets: [(2, 2), (5, 5), (10, 10), (15, 15), (20, 20), (30, 30)]')
    print('//' * 20)

    for seed in seeds:
        PCA_FLAG = True
        setup_seed(seed=seed)
        for pool_set in all_selected_pool:
            pool_set = [pool_set]
            print('\n\n-----------------------------------------')
            print('Random seed: {}'.format(seed))
            print('Pool: {}'.format(pool_set))
            print('PCA: {}'.format(PCA_FLAG))
            print('-----------------------------------------')
            test_policy_with_action_effect_features(config,
                                                    test_dataset,
                                                    model_action,
                                                    model_action_effect,
                                                    action_effect_features,
                                                    multiscale_pool=pool_set,
                                                    TQDM=False,
                                                    ONLY_ACTION_MODEL=False,
                                                    ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG,
                                                    STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    IGNORE_ACTION_MODEL=True)

    # * 04. w/o Early Stop, w/o PCA
    print('\n\n')
    print('//' * 20)
    print('04. w/o Early Stop, w/o PCA')
    print('pool_sets: [(2, 2), (5, 5), (10, 10), (15, 15), (20, 20), (30, 30)]')
    print('//' * 20)

    for seed in seeds:
        PCA_FLAG = False  # * !!!
        setup_seed(seed=seed)
        for pool_set in all_selected_pool:
            pool_set = [pool_set]
            print('\n\n-----------------------------------------')
            print('Random seed: {}'.format(seed))
            print('Pool: {}'.format(pool_set))
            print('PCA: {}'.format(PCA_FLAG))
            print('-----------------------------------------')
            test_policy_with_action_effect_features(config,
                                                    test_dataset,
                                                    model_action,
                                                    model_action_effect,
                                                    action_effect_features,
                                                    multiscale_pool=pool_set,
                                                    TQDM=False,
                                                    ONLY_ACTION_MODEL=False,
                                                    ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG,
                                                    STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    IGNORE_ACTION_MODEL=True)
