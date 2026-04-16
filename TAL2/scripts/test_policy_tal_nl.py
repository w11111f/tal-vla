import json
import pickle
import re
import warnings

import colorama
import torch

from src.config.config import init_args
from src.datasets.graph_dataset import GraphDataset_State
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_planning import plan_with_natural_language_instruction
from src.tal.utils_training import get_model, load_model

colorama.init()
warnings.filterwarnings('ignore')


def infer_world_num(world_path):
    match = re.search(r'(\d+)\.json$', world_path)
    if match is None:
        raise ValueError('Could not infer world index from path: {}'.format(world_path))
    return int(match.group(1))


def safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


if __name__ == '__main__':
    args = init_args()
    if args.instruction.strip() == '':
        raise ValueError('Please provide a natural language instruction via --instruction.')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)

    model_action_effect = get_model(config, 'AFE', config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    model_action_effect, optimizer, epoch, accuracy_list = load_model(
        config,
        seqTool + model_action_effect.name + '_Trained',
        model_action_effect
    )
    print('Model: {} | epoch: {}'.format(model_action_effect.name, epoch))
    model_action_effect = model_action_effect.to(config.device)

    model_action = get_model(config, 'APN', config.features_dim, config.num_objects)
    model_action, optimizer, epoch, accuracy_list = load_model(
        config,
        seqTool + model_action.name + '_Trained',
        model_action
    )
    print('Model: APN | epoch: {}'.format(epoch))
    model_action = model_action.to(config.device)

    features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
    with open(features_save_path, 'rb') as f:
        action_effect_features = pickle.load(f)
    print('Action effect features num: {}'.format(len(action_effect_features['names'])))

    start_node = None
    current_state_graph = None
    world_num = infer_world_num(config.world)

    if args.sample_index >= 0:
        graphs_dir = './data/{}/'.format(config.domain)
        dataset_path = args.dataset_path if args.dataset_path != '' else './data/test_dataset.pkl'
        dataset = GraphDataset_State(config, graphs_dir, dataset_path)
        data_item = dataset[args.sample_index]
        graph_seq, _, goal_json, action_seq, action2vec, world_name, start_node = data_item
        current_state_graph = graph_seq[0]
        world_num = safe_numeric_suffix(world_name, default=world_num)
        print('Initialized from dataset sample {} in {}.'.format(args.sample_index, world_name))

    result = plan_with_natural_language_instruction(
        config,
        model_action=model_action,
        model_extract_feature=model_action_effect,
        action_effect_features=action_effect_features,
        instruction=args.instruction,
        world_num=world_num,
        start_node=start_node,
        current_state_graph=current_state_graph,
        qwen_model_name=args.qwen_model,
        qwen_api_key=args.qwen_api_key,
        candidate_action_num=args.candidate_action_num,
        select_from_candidate=args.select_from_candidate,
        trajectory_length=args.max_planning_steps,
        with_pca=True
    )

    print('--' * 20)
    print('Status: {}'.format(result['status']))
    if 'error' in result:
        print('Error: {}'.format(result['error']))
    print('Predicted actions:')
    for action in result['predicted_actions']:
        print(action)
    print('--' * 20)
    print('Current scene graph JSON:')
    print(json.dumps(result['current_scene_graph_json'], ensure_ascii=False, indent=2))
    print('--' * 20)
    print('Goal scene graph JSON:')
    print(json.dumps(result['goal_scene_graph_json'], ensure_ascii=False, indent=2))
