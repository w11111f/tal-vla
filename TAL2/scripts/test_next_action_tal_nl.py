import os
import pickle
import re
import traceback
import warnings

import colorama
import torch

from src.config.config import init_args
from src.datasets.graph_dataset import GraphDataset_State
from src.envs import approx
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.scene_graph_translator import get_current_scene_graph_json, \
    translate_instruction_to_goal_state_graph
from src.tal.utils_planning import get_action_pred_with_model_actions, process_feature_with_pca, \
    scene_graph_goal_reached
from src.tal.utils_training import get_model, load_model

colorama.init()
warnings.filterwarnings('ignore')


def safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def resolve_model_checkpoint(config, model_name, seq_prefix=''):
    stable_ckpt = os.path.join(config.MODEL_SAVE_PATH, f'{seq_prefix}{model_name}_Trained.ckpt')
    if os.path.exists(stable_ckpt):
        return stable_ckpt

    pattern = re.compile(rf'^{re.escape(seq_prefix + model_name)}_(\d+)\.ckpt$')
    latest_epoch = -1
    latest_ckpt = None
    if not os.path.isdir(config.MODEL_SAVE_PATH):
        return None

    for filename in os.listdir(config.MODEL_SAVE_PATH):
        match = pattern.match(filename)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_ckpt = os.path.join(config.MODEL_SAVE_PATH, filename)
    return latest_ckpt


def load_required_model(config, model_name):
    model = get_model(config, model_name, config.features_dim, config.num_objects)
    seq_prefix = 'Seq_' if config.training == 'gcn_seq' else ''
    ckpt_path = resolve_model_checkpoint(config, model.name, seq_prefix=seq_prefix)
    if ckpt_path is None:
        raise FileNotFoundError(
            'Could not find checkpoint for model {} under {}.'.format(
                model.name,
                config.MODEL_SAVE_PATH
            )
        )
    model, optimizer, epoch, accuracy_list = load_model(
        config,
        seq_prefix + model.name + '_Trained',
        model,
        file_path=ckpt_path
    )
    model = model.to(config.device)
    return model, epoch, ckpt_path


def select_next_action_with_tal(
        config,
        model_action,
        model_extract_feature,
        action_effect_features,
        current_state_graph,
        goal_state_graph,
        candidate_action_num=20,
        select_from_candidate=10,
        with_pca=True,
        pca_q=500
):
    model_action.eval()
    model_extract_feature.eval()

    action_names = action_effect_features['names']
    action_features = action_effect_features['features']
    action_features_tensor = torch.stack(action_features).squeeze(1)
    principal_directions = None
    if with_pca:
        action_features_tensor, principal_directions = process_feature_with_pca(
            action_features_tensor,
            q_value=pca_q
        )

    with torch.no_grad():
        _, current_task_feature = model_extract_feature(current_state_graph, goal_state_graph)
        if with_pca:
            current_task_feature, _ = process_feature_with_pca(
                current_task_feature,
                principal_directions=principal_directions
            )

    act_generalized_inverse_mat = torch.linalg.pinv(action_features_tensor)
    candidate_scores = torch.matmul(current_task_feature, act_generalized_inverse_mat).squeeze(0)
    topk_count = min(candidate_action_num, candidate_scores.shape[0])
    topk_actions = torch.topk(candidate_scores, topk_count)
    candidate_indices = torch.sort(topk_actions.indices).values
    candidate_actions = [action_names[idx] for idx in candidate_indices]

    actions_prob = get_action_pred_with_model_actions(
        config,
        model_action,
        action_effect_features,
        current_state_graph,
        goal_state_graph
    )

    candidate_probs = []
    for act in candidate_actions:
        idx = action_names.index(act)
        candidate_probs.append(actions_prob[0][idx])

    max_prob = max(candidate_probs)
    next_action = candidate_actions[candidate_probs.index(max_prob)]
    return next_action


if __name__ == '__main__':
    args = init_args()
    if args.instruction.strip() == '':
        raise ValueError('Please provide a natural language instruction via --instruction.')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.exec_type = 'policy'
    config = EnvironmentConfig(args)

    try:
        print('[Stage 1/5] Loading AFE model...')
        model_action_effect, afe_epoch, afe_ckpt = load_required_model(config, 'AFE')
        print('Model: AFE | epoch: {} | ckpt: {}'.format(afe_epoch, afe_ckpt))

        print('[Stage 2/5] Loading APN model...')
        model_action, apn_epoch, apn_ckpt = load_required_model(config, 'APN')
        print('Model: APN | epoch: {} | ckpt: {}'.format(apn_epoch, apn_ckpt))

        print('[Stage 3/5] Loading action effect features...')
        features_save_path = './' + config.MODEL_SAVE_PATH + 'action_effect_features_avg.pkl'
        if not os.path.exists(features_save_path):
            raise FileNotFoundError(
                'Action effect feature file not found: {}'.format(features_save_path)
            )
        with open(features_save_path, 'rb') as f:
            action_effect_features = pickle.load(f)
        print('Action effect features num: {}'.format(len(action_effect_features['names'])))

        start_node = None
        current_state_graph = None
        current_scene_graph_json = None
        world_num = safe_numeric_suffix(config.graph_world_name, default=0)

        if args.sample_index >= 0:
            graphs_dir = './data/{}/'.format(config.domain)
            dataset_path = args.dataset_path if args.dataset_path != '' else './data/test_dataset.pkl'
            print('[Stage 4/5] Loading dataset sample {} from {}...'.format(
                args.sample_index,
                dataset_path
            ))
            dataset = GraphDataset_State(config, graphs_dir, dataset_path)
            graph_seq, _, goal_json, action_seq, action2vec, world_name, start_node = dataset[
                args.sample_index
            ]
            current_state_graph = graph_seq[0]
            world_num = safe_numeric_suffix(world_name, default=world_num)
            print('Initialized from dataset sample {} in {}.'.format(args.sample_index, world_name))

        print('[Stage 4/5] Initializing current scene state...')
        approx.initPolicy(
            config,
            config.domain,
            goal_json=None,
            world_num=world_num,
            SET_GAOL_JSON=False,
            INPUT_DATAPOINT=start_node
        )

        if current_state_graph is None:
            current_state_graph = approx.getInitializeDGLGraph(config)
        if config.device is not None:
            current_state_graph = current_state_graph.to(config.device)

        print('[Stage 5/5] Translating instruction to goal scene graph...')
        current_scene_graph_json = get_current_scene_graph_json(config, state_name='Initialize')
        _, goal_scene_graph_json, goal_state_graph = translate_instruction_to_goal_state_graph(
            config,
            instruction=args.instruction,
            current_scene_graph_json=current_scene_graph_json,
            model_name=args.qwen_model,
            api_key=args.qwen_api_key
        )
        if config.device is not None:
            goal_state_graph = goal_state_graph.to(config.device)

        print('--' * 20)
        print('Instruction: {}'.format(args.instruction))

        if scene_graph_goal_reached(config, current_state_graph, goal_state_graph):
            print('--' * 20)
            print('The current scene already satisfies the goal scene graph.')
            print('Next action: None')
        else:
            print('[Final] Running TAL next-action selection...')
            next_action = select_next_action_with_tal(
                config,
                model_action,
                model_action_effect,
                action_effect_features,
                current_state_graph,
                goal_state_graph,
                candidate_action_num=args.candidate_action_num,
                select_from_candidate=args.select_from_candidate,
                with_pca=True
            )
            print('--' * 20)
            print('Next action:')
            print(next_action)
    except Exception as exc:
        print('[Error] test_next_action_tal_nl.py failed: {}'.format(exc))
        traceback.print_exc()
    finally:
        approx.close_backend()
