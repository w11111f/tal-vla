import pickle
import torch
import colorama
import warnings
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.envs import approx
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_planning import test_policy_with_action_effect_features

colorama.init()
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    config = EnvironmentConfig(args)
    try:
        # * ------------------------------------------------------------------------------------------
        # * Load data.
        graphs_dir = './data/home/'
        train_data_path = './data/train_dataset.pkl'
        train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
        print('Train data num: {}'.format(len(train_dataset)))

        # val_data_path = './data/val_dataset.pkl'
        # val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

        test_data_path = './data/test_dataset.pkl'
        test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)
        print('Test data num: {}'.format(len(test_dataset)))
        if args.max_samples >= 0:
            print('Debug max_samples: {}'.format(args.max_samples))
        print('Debug print_sample_info: {}'.format(args.print_sample_info))

        # * ------------------------------------------------------------------------------------------
        # * Create model and load parameters.
        # * 01.Action Feature Extractor: AFE
        model_action_effect = get_model(config, config.model_name, config.features_dim,
                                        config.num_objects)
        seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
        model_action_effect, optimizer, epoch, accuracy_list = load_model(
            config,
            seqTool + model_action_effect.name + '_Trained',
            model_action_effect
        )
        print('Model: {} | epoch: {}'.format(model_action_effect.name, epoch))
        model_action_effect = model_action_effect.to(config.device)

        # * 02.Action Proposal Network: APN
        model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
        seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
        model_action, optimizer, epoch, accuracy_list = load_model(
            config,
            seqTool + model_action.name + '_Trained',
            model_action
        )
        print('Model: APN | epoch: {}'.format(epoch))
        model_action = model_action.to(config.device)
        print('Policy backend: {}'.format(config.policy_backend))

        # * ------------------------------------------------------------------------------------------
        # * Load features.
        features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
        with open(features_save_path, 'rb') as f:
            action_effect_features = pickle.load(f)
        print('Action effect features num: {}'.format(len(action_effect_features['names'])))

        # * ------------------------------------------------------------------------------------------
        # * Policy test.
        seeds = [42]
        MCAS = [(5, 2), (10, 5), (15, 5), (20, 5), (20, 10), (30, 5), (30, 10)]
        PCA_FLAG = True

        print('\n\n')
        print('//' * 20)
        print('Candidate Action Pool (w/ PCA, Early Stopping)')
        print('pool_sets: {}'.format(MCAS))
        print('PCA_FLAG: {}'.format(PCA_FLAG))
        print('//' * 20)

        print('\n\n-----------------------------------------')
        print('Test on training set.')
        print('-----------------------------------------')
        for seed in seeds:
            setup_seed(seed=seed)
            print('Random seed: {}'.format(seed))
            print('Evaluating {} samples from train_dataset...'.format(len(train_dataset)))
            test_policy_with_action_effect_features(config,
                                                    train_dataset,
                                                    model_action,
                                                    model_action_effect,
                                                    action_effect_features,
                                                    multiscale_pool=MCAS,
                                                    TQDM=True,
                                                    ONLY_ACTION_MODEL=False,
                                                    ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG,
                                                    STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    IGNORE_ACTION_MODEL=True,
                                                    MAX_SAMPLES=args.max_samples,
                                                    PRINT_SAMPLE_INFO=args.print_sample_info,
                                                    PROGRESS_DESC='TAL Train')

        # print('\n\n-----------------------------------------')
        # print('Test on val set.')
        # print('-----------------------------------------')
        # for seed in seeds:
        #     setup_seed(seed=seed)
        #     print('Random seed: {}'.format(seed))
        #     test_policy_with_action_effect_features(config,
        #                                             val_dataset,
        #                                             model_action,
        #                                             model_action_effect,
        #                                             action_effect_features,
        #                                             multiscale_pool=MCAS,
        #                                             TQDM=True,
        #                                             ONLY_ACTION_MODEL=False,
        #                                             ONLY_FEATURE_MODEL=False,
        #                                             WITH_PCA=PCA_FLAG,
        #                                             STATE_FORMAT_GOAL=True,
        #                                             INIT_DATAPOINT=True,
        #                                             IGNORE_ACTION_MODEL=True)

        print('\n\n-----------------------------------------')
        print('Test on test set.')
        print('-----------------------------------------')
        for seed in seeds:
            setup_seed(seed=seed)
            print('Random seed: {}'.format(seed))
            print('Evaluating {} samples from test_dataset...'.format(len(test_dataset)))
            test_policy_with_action_effect_features(config,
                                                    test_dataset,
                                                    model_action,
                                                    model_action_effect,
                                                    action_effect_features,
                                                    multiscale_pool=MCAS,
                                                    TQDM=True,
                                                    ONLY_ACTION_MODEL=False,
                                                    ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG,
                                                    STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    IGNORE_ACTION_MODEL=True,
                                                    MAX_SAMPLES=args.max_samples,
                                                    PRINT_SAMPLE_INFO=args.print_sample_info,
                                                    PROGRESS_DESC='TAL Test')
    finally:
        approx.close_backend()
