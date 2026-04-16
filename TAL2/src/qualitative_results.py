import pickle
import torch
import warnings
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import get_model, load_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_planning import generate_qualitative_result

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    rnd_seed = 1
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/home/'
    test_data_path = './data/test_dataset.pkl'
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

    # * ----------------------------------------------------------------
    # * Create model and load parameters.
    # * 01.Action Feature Extractor: AFE
    model_action_effect = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action_effect, optimizer, epoch, accuracy_list = load_model(config,
                                                                      seqTool + model_action_effect.name + '_Trained',
                                                                      model_action_effect)
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
    qualitative_data = generate_qualitative_result(config, test_dataset, model_action, model_action_effect,
                                                   action_effect_features, TQDM=False, WITH_PCA=True,
                                                   STATE_FORMAT_GOAL=True, INIT_DATAPOINT=True, target_length=None)

    print('--' * 20)
    print('Data num: {}'.format(len(qualitative_data)))
    print('--' * 20)
    results_save_path = './results/qualitative_results.pkl'
    with open(results_save_path, 'wb') as f:
        pickle.dump(qualitative_data, f)


