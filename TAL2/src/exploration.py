import os
import time
import random
import pickle
import torch
import colorama
import numpy as np
import networkx as nx
from copy import deepcopy
from termcolor import cprint
from collections import deque
from torch.utils.data import WeightedRandomSampler

from src.config.config import init_args
from src.envs import isaac_env as env
# from src.envs import husky_ur5 as env
from src.envs.CONSTANTS import EnvironmentConfig

colorama.init()


def load_Aall_and_split(config, upsample=True):
    with open(config.Aall_path, 'rb') as f:
        Aall = pickle.load(f)
    possible_action_set = []  # * without moveTo
    moveTo_set = []  # * moveTo
    for action in Aall:
        if action['actions'][0]['name'] == 'moveTo':
            moveTo_set.append(action)
        else:
            possible_action_set.append(action)

    print('--' * 20)
    cprint('Aall actions num: {}'.format(len(Aall)), 'green')  # * 1364
    cprint('possible_action_set num: {}'.format(len(possible_action_set)), 'green')  # * 1332
    cprint('moveTo_set num: {}'.format(len(moveTo_set)), 'green')  # * 32
    print('--' * 20)

    # * ---------------------------------------------------------------
    # * UpSampling.
    if upsample:
        hl_actions_upsample = []
        hl_actions_dict = {}
        for action in possible_action_set:
            action_name = action['actions'][0]['name']
            if action_name not in hl_actions_dict:
                hl_actions_dict[action_name] = [action]
            else:
                hl_actions_dict[action_name].append(action)

        max_num = -1
        for value in hl_actions_dict.values():
            if len(value) > max_num:
                max_num = len(value)
        for key, value in hl_actions_dict.items():
            delta_num = max_num - len(value)
            if key == 'pick':
                delta_num = delta_num / 3
            # * UpSampling
            if (delta_num > 0):
                value += random.choices(value, k=int(delta_num))
            hl_actions_upsample += value

        possible_action_set = deque(hl_actions_upsample)
        data_num = len(possible_action_set)

        # * -----------------------------------------------------------
        # * Shuffle
        for _ in range(10):
            random.shuffle(possible_action_set)
            possible_action_set.rotate(random.randint(int(data_num / 5), data_num))

        print('--' * 20)
        cprint('Up-sample', 'green')
        cprint('possible_action_set num: {}'.format(len(possible_action_set)), 'green')  # * 4239
        cprint('moveTo_set num: {}'.format(len(moveTo_set)), 'green')  # * 32
        print('--' * 20)

    return possible_action_set, moveTo_set


def select_actions_from_set(action_set, actions_selected_num, child_actions):
    """Random select high level actions."""
    # * Convert actions_selected_num to sample_weights.
    # * actions_selected_num: record how many times each action was selected.
    # * sample_weights: the more times selected in the history, the lower the probability.
    sample_weights = 1 / actions_selected_num  # 1 / 0 -> inf.
    sample_weights[torch.isinf(sample_weights)] = 0  # * inf -> 0.
    while True:
        # data = random.sample(possible_hl_actions, 1)
        # * data: list with num_samples elems.
        data = list(WeightedRandomSampler(weights=sample_weights, num_samples=1))
        data_idx = data[0]
        if action_set[data_idx] in child_actions:
            continue
        else:
            hl_item = action_set[data_idx]['actions'][0]
            hl_actions = action_set[data_idx]
            # * Add selected num. If this action raise error, it will be set to 0 later.
            actions_selected_num[data_idx] += 1
            return hl_item, hl_actions, data_idx, actions_selected_num


def explore_from_node_i(config, graph, start_node_id, graph_last_node_id, explore_step,
                        possible_hl_actions,
                        actions_selected_num):
    (possible_action_set, moveTo_set) = possible_hl_actions
    # * Reload world state of start_node_id.
    child_actions = graph.nodes[start_node_id]['child_actions']
    state_id = graph.nodes[start_node_id]['world_state']
    previous_constraints = deepcopy(graph.nodes[start_node_id]['pre_constraints'])
    previous_state_datapoint = deepcopy(graph.nodes[start_node_id]['state'])
    env.restoreState(state_id, previous_constraints, previous_state_datapoint)
    env.resetDatapoint(config, previous_state_datapoint)

    cursor_start = start_node_id
    cursor_end = graph_last_node_id + 1

    # * ---------------------------------------------------------------
    # * Process action 'pick'.
    PRE_PICK = None
    PRE_PICK_AND_MOVE = False
    held_obj = env.get_held_object(previous_constraints)
    if held_obj is not None:
        PRE_PICK = {'name': 'pick', 'args': [held_obj]}
    # * ---------------------------------------------------------------
    step = 0
    data_idx = None
    PRE_UP = None
    error_act_num = 0
    while (step < explore_step):
        if error_act_num > 30:
            break
        # time.sleep(1)

        # * -----------------------------------------------------------
        # * Select actions.
        # * Last exploration step.
        if (step + 1) == explore_step:
            # * Process actions: climbUp/climbDown and moveUp/moveDown.
            if (PRE_UP is not None):
                if (PRE_UP['name'] == 'climbUp'):
                    PRE_UP['name'] = 'climbDown'
                elif (PRE_UP['name'] == 'moveUp'):
                    PRE_UP['name'] = 'moveDown'
                hl_item = deepcopy(PRE_UP)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_UP = None
            # * Process actions: pick and drop.
            elif (PRE_PICK is not None):
                PRE_PICK['name'] = 'drop'
                hl_item = deepcopy(PRE_PICK)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_PICK = None
            # * Process other actions, random select.
            else:
                hl_item, hl_actions, data_idx, actions_selected_num = select_actions_from_set(
                    possible_action_set, actions_selected_num, child_actions
                )
        else:  # * Random select high level actions.
            rnd_val = np.random.rand()
            # * Process actions: climbUp/climbDown and moveUp/moveDown.
            if (rnd_val > 0.7) and (PRE_UP is not None):
                if (PRE_UP['name'] == 'climbUp'):
                    PRE_UP['name'] = 'climbDown'
                elif (PRE_UP['name'] == 'moveUp'):
                    PRE_UP['name'] = 'moveDown'
                hl_item = deepcopy(PRE_UP)
                hl_actions = {'actions': [hl_item]}
                data_idx = None
                PRE_UP = None
            # * Process actions: pick and drop.
            elif PRE_PICK is not None:
                # * pick --> moveTo --> drop
                if PRE_PICK_AND_MOVE is False:
                    hl_actions = random.choice(moveTo_set)
                    hl_item = hl_actions['actions'][0]
                    data_idx = None
                    PRE_PICK_AND_MOVE = True
                else:
                    PRE_PICK['name'] = 'drop'
                    hl_item = deepcopy(PRE_PICK)
                    hl_actions = {'actions': [hl_item]}
                    data_idx = None
                    PRE_PICK = None
                    PRE_PICK_AND_MOVE = False
            # * Process other actions, random select.
            else:
                hl_item, hl_actions, data_idx, actions_selected_num = select_actions_from_set(
                    possible_action_set, actions_selected_num, child_actions
                )

        # * -----------------------------------------------------------
        # * Execute action.
        cprint('Action num: {} '.format(cursor_end), color='green', end='')
        cprint(hl_actions, 'green')
        try:
            done = env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)
            error_act_num = 0
            if hl_item['name'] in ['climbDown', 'moveDown']:
                PRE_UP = None
            elif hl_item['name'] == 'drop':
                PRE_PICK = None
        except Exception as e:
            error_act_num += 1
            cprint(str(e), 'red')
            if 'Gripper is free' in str(e):
                PRE_PICK = None

            # cprint(str(e.__traceback__.tb_frame.f_globals['__file__']), 'red')  # File
            # cprint(str(e.__traceback__.tb_lineno), 'red')   # Line
            # if str(e).startswith('Error'):
            if str(e).startswith('Error') or ('Can not complete' in str(e)) or (
                    'list indices' in str(e)):
                error_act_num = 0
                # * Due to the actions_selected_num will be converted to the sample_weights,
                # * 0 means the sampling probability is 0.
                if data_idx is not None:
                    actions_selected_num[data_idx] = 0
            cprint('Reload previous state.'.format(e), 'green')
            # * Reload world state.
            env.restoreState(state_id, previous_constraints, previous_state_datapoint)
            env.resetDatapoint(config, previous_state_datapoint)
            continue

        # * -----------------------------------------------------------
        # * Process 'climbUp', 'moveUp', 'pick'
        if (PRE_UP is None) and (hl_item['name'] in ['climbUp', 'moveUp']):
            PRE_UP = deepcopy(hl_item)
        elif (PRE_PICK is None) and (hl_item['name'] == 'pick'):
            PRE_PICK = deepcopy(hl_item)

        # * -----------------------------------------------------------
        # * Store data to graph.
        env_state_datapoint = env.getDatapoint(config, RESET_DATAPOINT=True)
        previous_state_datapoint = deepcopy(env_state_datapoint)
        assert len(env_state_datapoint.metrics) == len(env_state_datapoint.actions), \
            '[Error]: len(datapoint.metrics) != len(datapoint.actions)'
        state_id, previous_constraints = env.saveState()  # * Save world state.
        previous_constraints = deepcopy(previous_constraints)
        graph.add_node(
            cursor_end,
            state=env_state_datapoint,
            id=cursor_end,
            parent_id=cursor_start,
            world_state=state_id,
            pre_constraints=previous_constraints,
            child_actions=[],
        )  # * Add node.
        graph.add_weighted_edges_from([(cursor_start, cursor_end, 1.0), ])  # * Add edge.
        graph[cursor_start][cursor_end]['action'] = hl_actions  # * Set edge attribute.
        graph.nodes[cursor_start]['child_actions'].append(hl_item)

        cursor_start = cursor_end
        cursor_end += 1
        step += 1
        # * -----------------------------------------------------------

    return graph, possible_hl_actions, actions_selected_num


def collect_data(config, first_explore_steps=10, random_select_node=99, explore_steps_per_node=10,
                 world_num=1, start_world_id=0):
    # * Load actions.
    possible_action_set, moveTo_set = load_Aall_and_split(config)
    actions_selected_num = torch.ones(len(possible_action_set))

    for i in range(start_world_id, world_num):
        # * -----------------------------------------------------------
        # * Create config.
        config.world = config.graph_world_name
        print('...' * 30)
        print('Creating new world in Isaac Lab (expff.usd) ...')
        print('...' * 30)
        time.sleep(3)
        env.start(config)

        # * -----------------------------------------------------------
        # * Create directed graph.
        DG = nx.DiGraph()
        node_id = 0  # * State (datapoint) in node 0 is empty.
        state_id, previous_constraints = env.saveState()
        env.initRootNode()  # * Add 'End' to the datapoint, use 'End' as node final state.
        env_state_datapoint = env.getDatapoint(config, RESET_DATAPOINT=True)
        DG.add_node(
            node_id,
            state=env_state_datapoint,
            id=node_id,
            parent_id=None,
            world_state=state_id,
            # * Pybullet state id. State id is used to restore the environment.
            pre_constraints=previous_constraints,
            child_actions=[]
        )  # * Add node.

        # * -----------------------------------------------------------
        # * Collect data.
        # * 1. Explore from root node.
        graph, possible_hl_actions, actions_selected_num = explore_from_node_i(
            config, graph=DG, start_node_id=0, graph_last_node_id=0,
            explore_step=first_explore_steps,
            possible_hl_actions=(possible_action_set, moveTo_set),
            actions_selected_num=actions_selected_num
        )

        # * 2. Explore from random selected node.
        for _ in range(random_select_node):
            # * Must be here due to the node num is changed after exploration.
            nodes_list = list(DG.nodes)

            # * Random choice node and reload world state.
            UP_FLAG = False
            selected_node_id = random.choice(nodes_list)
            while True:
                parent_node_id = graph.nodes[selected_node_id]['parent_id']
                if parent_node_id is not None:  # * parent_node_id is None in root node.
                    origin_hl_action = graph[parent_node_id][selected_node_id]['action']
                else:
                    selected_node_id = (selected_node_id + 1) % len(nodes_list)
                    continue
                origin_hl_item = origin_hl_action['actions'][0]
                if origin_hl_item['name'] in ['climbUp', 'moveUp']:
                    UP_FLAG = True
                    selected_node_id = (selected_node_id + 1) % len(nodes_list)
                elif UP_FLAG and (origin_hl_item['name'] in ['climbDown', 'moveDown']):
                    break
                else:
                    break

            cprint('---' * 20, 'yellow')
            cprint('Explore from node: {}'.format(selected_node_id), 'yellow')
            graph, possible_hl_actions, actions_selected_num = explore_from_node_i(
                config,
                graph=DG,
                start_node_id=selected_node_id,
                graph_last_node_id=nodes_list[-1],
                explore_step=explore_steps_per_node,
                possible_hl_actions=possible_hl_actions,
                actions_selected_num=actions_selected_num
            )

        # * -----------------------------------------------------------
        # * Save graph.
        folder_name = os.path.join('data', config.domain, config.graph_world_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        data_num = len(os.listdir(folder_name))
        graph_save_path = folder_name + '/' + str(data_num) + '.graph'
        with open(graph_save_path, 'wb') as f:
            pickle.dump(DG, f)

        # * Destroy world.
        env.destroy()


if __name__ == '__main__':
    args = init_args()
    config = EnvironmentConfig(args)
    config.display = True

    collect_data(config)
    # load_Aall_and_split(config)
