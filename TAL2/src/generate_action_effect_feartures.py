import pickle
import torch
import warnings
from tqdm import tqdm
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model
from src.datasets.graph_dataset import GraphDataset_State

warnings.filterwarnings('ignore')


def generate_action_features(config, dataset, model, action_names, action_features):
    model.eval()

    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) in tqdm(dataset, ncols=80):
        graphSeq.append(goal2vec)
        with torch.no_grad():
            for i in range(len(graphSeq) - 1):
                tmp_idx = None
                if actionSeq[i] in action_names:
                    tmp_idx = action_names.index(actionSeq[i])
                else:
                    output, output_features = model(graphSeq[i], graphSeq[i + 1])
                    if tmp_idx is not None:
                        tmp_idx += 1
                        action_names.insert(tmp_idx, actionSeq[i])
                        action_features.insert(tmp_idx, output_features)
                    else:
                        action_names.append(actionSeq[i])
                        action_features.append(output_features)

    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))
    # return {'names': tuple(action_names), 'features': tuple(action_features)}
    return action_names, action_features


def generate_action_features_average(config, dataset, model, action_names, action_features):
    '''
    Averaging same actions' feature.
    action_names: ['Action_1', 'Action_2', ...]
    action_features: [Action_1_feature_list, Action_2_feature_list, ...]
    '''
    model.eval()

    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) in tqdm(dataset, ncols=80):
        graphSeq.append(goal2vec)

        for i in range(len(graphSeq) - 1):
            with torch.no_grad():
                output, output_features = model(graphSeq[i], graphSeq[i + 1])
            tmp_idx = None
            if actionSeq[i] in action_names:  # * Aciton already exists.
                tmp_idx = action_names.index(actionSeq[i])
                action_features[tmp_idx].append(output_features)
            else:
                if tmp_idx is not None:
                    tmp_idx += 1
                    action_names.insert(tmp_idx, actionSeq[i])
                    action_features.insert(tmp_idx, [output_features])  # * Feature list!!!
                else:
                    action_names.append(actionSeq[i])
                    action_features.append([output_features])

    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))
    # return {'names': tuple(action_names), 'features': tuple(action_features)}
    return action_names, action_features


if __name__ == '__main__':
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'AFE'
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    train_data_num = len(train_dataset)
    print('Train data num: {}'.format(train_data_num))

    # * ----------------------------------------------------------------
    # * Create model and load parameters.
    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model, optimizer, epoch, accuracy_list = load_model(config, seqTool + model.name + '_Trained', model)
    model = model.to(config.device)

    # * ----------------------------------------------------------------
    # * Generate action effect features.
    action_names = []
    action_features = []
    action_names, action_features = generate_action_features_average(config, train_dataset, model,
                                                                     action_names, action_features)
    # * Average action_features.
    avg_action_features = [sum(x) / len(x) for x in action_features]
    assert len(action_names) == len(action_features)
    print('Action num: {}'.format(len(action_names)))

    # * -----------------------------------------------------------------
    action_feature_dict = {'names': tuple(action_names), 'features': tuple(avg_action_features)}
    # * Save features.
    features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
    with open(features_save_path, 'wb') as f:
        pickle.dump(action_feature_dict, f)
