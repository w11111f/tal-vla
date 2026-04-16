import os
import pickle
import torch
import colorama
import warnings
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_planning import test_policy_with_action_effect_features

colorama.init()
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    dataset_n = 2
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'  # * Same structure as AFE, but trained with only action cls loss.
    config = EnvironmentConfig(args)
    config.MODEL_SAVE_PATH = os.path.join(config.MODEL_SAVE_PATH, 'dataset_{}'.format(dataset_n))
    print('==' * 10)
    print('Dataset-{}'.format(dataset_n))
    print('==' * 10)

    # * ---------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/{}/home/'.format(dataset_n)
    train_data_path = './data/{}/train_dataset.pkl'.format(dataset_n)
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    train_data_num = len(train_dataset)
    # val_data_path = './data/{}/val_dataset.pkl'.format(dataset_n)
    # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
    test_data_path = './data/{}/test_dataset.pkl'.format(dataset_n)
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

    # * ---------------------------------------------------------------
    # * Create model and load parameters.
    # * 01.Action Feature Extractor: AFE
    model_action_effect = get_model(config, config.model_name, config.features_dim,
                                    config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action_effect, optimizer, epoch, accuracy_list = load_model(
        config, seqTool + model_action_effect.name + '_OA_Trained', model_action_effect
    )  # * !!!
    print('Model: {} | epoch: {}'.format(model_action_effect.name, epoch))
    model_action_effect = model_action_effect.to(config.device)

    # * 02.Action Proposal Network: APN
    model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action, optimizer, epoch, accuracy_list = load_model(
        config, seqTool + model_action.name + '_Trained', model_action
    )
    print('Model: APN | epoch: {}'.format(epoch))
    model_action = model_action.to(config.device)

    # * ---------------------------------------------------------------
    # * Load action features. (Learned Fragmented Knowledge.)
    # features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
    features_save_path = './' + config.MODEL_SAVE_PATH + '/AFE_OA_action_effect_features_avg.pkl'
    with open(features_save_path, 'rb') as f:
        action_effect_features = pickle.load(f)
    print('Action effect features num: {}'.format(len(action_effect_features['names'])))

    # * ---------------------------------------------------------------
    # * Policy test.
    seeds = [0, 1, 42]
    MCAS = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
    PCA_FLAG = True

    print('\n\n')
    print('//' * 20)
    print('Candidate Action Pool (w/ PCA, Early Stopping)')
    print('pool_sets: {}'.format(MCAS))
    print('PCA_FLAG: {}'.format(PCA_FLAG))
    print('//' * 20)

    # print('\n\n-----------------------------------------')
    # print('Test on training set.')
    # print('-----------------------------------------')
    # for seed in seeds:
    #     setup_seed(seed=seed)
    #     print('Random seed: {}'.format(seed))
    #     test_policy_with_action_effect_features(
    #         config,
    #         train_dataset,
    #         model_action,
    #         model_action_effect,
    #         action_effect_features,
    #         multiscale_pool=MCAS,
    #         TQDM=False,
    #         ONLY_ACTION_MODEL=False,
    #         ONLY_FEATURE_MODEL=False,
    #         WITH_PCA=PCA_FLAG,
    #         STATE_FORMAT_GOAL=True,
    #         INIT_DATAPOINT=True,
    #         IGNORE_ACTION_MODEL=True
    #     )

    # print('\n\n-----------------------------------------')
    # print('Test on val set.')
    # print('-----------------------------------------')
    # for seed in seeds:
    #     setup_seed(seed=seed)
    #     print('Random seed: {}'.format(seed))
    #     test_policy_with_action_effect_features(
    #         config,
    #         val_dataset,
    #         model_action,
    #         model_action_effect,
    #         action_effect_features,
    #         multiscale_pool=MCAS,
    #         TQDM=False,
    #         ONLY_ACTION_MODEL=False,
    #         ONLY_FEATURE_MODEL=False,
    #         WITH_PCA=PCA_FLAG,
    #         STATE_FORMAT_GOAL=True,
    #         INIT_DATAPOINT=True,
    #         IGNORE_ACTION_MODEL=True
    #     )

    print('\n\n-----------------------------------------')
    print('Test on test set.')
    print('-----------------------------------------')
    for seed in seeds:
        setup_seed(seed=seed)
        print('Random seed: {}'.format(seed))
        test_policy_with_action_effect_features(
            config,
            test_dataset,
            model_action,
            model_action_effect,
            action_effect_features,
            multiscale_pool=MCAS,
            TQDM=False,
            ONLY_ACTION_MODEL=False,
            ONLY_FEATURE_MODEL=False,
            WITH_PCA=PCA_FLAG,
            STATE_FORMAT_GOAL=True,
            INIT_DATAPOINT=True,
            IGNORE_ACTION_MODEL=True
        )
