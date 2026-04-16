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
    config = EnvironmentConfig(args)
    try:
        # * ----------------------------------------------------------------
        # * Load data.
        graphs_dir = './data/home/'
        train_data_path = './data/train_dataset.pkl'
        train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
        print('Train data num: {}'.format(len(train_dataset)))

        val_data_path = './data/val_dataset.pkl'
        val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
        print('Val data num: {}'.format(len(val_dataset)))

        test_data_path = './data/test_dataset.pkl'
        test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)
        print('Test data num: {}'.format(len(test_dataset)))
        if args.max_samples >= 0:
            print('Debug max_samples: {}'.format(args.max_samples))
        print('Debug print_sample_info: {}'.format(args.print_sample_info))

        # * ----------------------------------------------------------------
        model_action_effect = None
        action_effect_features = None

        # * Action Proposal Network: APN
        model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
        seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
        model_action, optimizer, epoch, accuracy_list = load_model(
            config,
            seqTool + model_action.name + '_Trained',
            model_action
        )
        model_action = model_action.to(config.device)
        print('Model: APN | epoch: {}'.format(epoch))
        print('Policy backend: {}'.format(config.policy_backend))

        # * ----------------------------------------------------------------
        seeds = [42]
        print('\n\n///////////////////////////////////////////////////////')
        print('Test BC.')
        print('Random seeds: {}'.format(str(seeds)))
        print('///////////////////////////////////////////////////////////')

        for seed in seeds:
            PCA_FLAG = True
            setup_seed(seed=seed)

            print('\n\n-------------------------------------')
            print('Random seed: {}'.format(seed))
            print('-----------------------------------------')

            print('\n\nTraining set.')
            print('Evaluating {} samples from train_dataset...'.format(len(train_dataset)))
            test_policy_with_action_effect_features(config, train_dataset, model_action,
                                                    model_action_effect,
                                                    action_effect_features, multiscale_pool=None,
                                                    TQDM=True,
                                                    ONLY_ACTION_MODEL=True, ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG, STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    MAX_SAMPLES=args.max_samples,
                                                    PRINT_SAMPLE_INFO=args.print_sample_info,
                                                    PROGRESS_DESC='BC Train')

            print('\n\nVal set.')
            print('Evaluating {} samples from val_dataset...'.format(len(val_dataset)))
            test_policy_with_action_effect_features(config, val_dataset, model_action,
                                                    model_action_effect,
                                                    action_effect_features, multiscale_pool=None,
                                                    TQDM=True,
                                                    ONLY_ACTION_MODEL=True, ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG, STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    MAX_SAMPLES=args.max_samples,
                                                    PRINT_SAMPLE_INFO=args.print_sample_info,
                                                    PROGRESS_DESC='BC Val')

            print('\n\nTest set.')
            print('Evaluating {} samples from test_dataset...'.format(len(test_dataset)))
            test_policy_with_action_effect_features(config, test_dataset, model_action,
                                                    model_action_effect,
                                                    action_effect_features, multiscale_pool=None,
                                                    TQDM=True,
                                                    ONLY_ACTION_MODEL=True, ONLY_FEATURE_MODEL=False,
                                                    WITH_PCA=PCA_FLAG, STATE_FORMAT_GOAL=True,
                                                    INIT_DATAPOINT=True,
                                                    MAX_SAMPLES=args.max_samples,
                                                    PRINT_SAMPLE_INFO=args.print_sample_info,
                                                    PROGRESS_DESC='BC Test')
    finally:
        approx.close_backend()
