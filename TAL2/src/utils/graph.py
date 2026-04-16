"""
@File             : graph.py
@Time             : 2021/11/2
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/05/31
@Desciption       : None
"""
import os
import json
import pickle
import torch
import warnings
import colorama
import itertools
import numpy as np
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from termcolor import cprint

from src.envs import approx
from src.utils.misc import convertToDGLGraph
from src.datasets.utils_dataset import DGLDataset

warnings.filterwarnings('ignore')
colorama.init()


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


def merge_datapoint(datapoints):
    tmp_dp = deepcopy(datapoints[0])  # * Necessary!!! Can't remove deepcopy here.
    for dp in datapoints[1:]:
        tmp_dp.symbolicActions.extend(dp.symbolicActions)  # * Symbolic actions
        tmp_dp.position.extend(dp.position)  # * Robot position list
        tmp_dp.metrics.extend(dp.metrics)  # * Metrics of all objects
        # metrics = {key: [list(a), list(b)] for key, (a, b) in dp.metrics[0].items()}
        # tmp_dp.metrics.extend([metrics])  # * Metrics of all objects
        tmp_dp.sticky.extend(dp.sticky)  # * Sticky objects
        tmp_dp.fixed.extend(dp.fixed)  # * Fixed objects
        tmp_dp.cleaner.extend(dp.cleaner)  # * Has cleaner
        tmp_dp.actions.extend(dp.actions)  # * Action
        tmp_dp.constraints.extend(dp.constraints)  # * Constraints
        tmp_dp.on.extend(dp.on)  # * Objects on
        tmp_dp.clean.extend(dp.clean)  # * Objects Cleaned
        tmp_dp.stick.extend(dp.stick)  # * Stick with object
        tmp_dp.welded.extend(dp.welded)  # * Objects welded
        tmp_dp.drilled.extend(dp.drilled)  # * Objects drilled
        tmp_dp.painted.extend(dp.painted)  # * Objects painted
        tmp_dp.fueled.extend(dp.fueled)  # * Objects fueled
        tmp_dp.cut.extend(dp.cut)  # * Objects cut

    return tmp_dp


def load_dataset(config, root_dir='data/', INT_TYPE_GOAL=False):
    if not os.path.exists(config.MODEL_SAVE_PATH): os.makedirs(config.MODEL_SAVE_PATH)
    filename = (root_dir + '/../' + config.domain + '_' +
                ('global_' if config.globalnode else '') +
                ('NoTool_' if not config.ignoreNoTool else '') +
                ('seq_' if config.sequence else '') +
                (config.embedding) +
                str(config.AUGMENTATION) + '.pkl')
    print(filename)
    if config.globalnode: config.etypes.append('Global')
    if os.path.exists(filename):
        return pickle.load(open(filename, 'rb'))
    data = DGLDataset(config,
                      root_dir,
                      augmentation=config.AUGMENTATION,
                      globalNode=config.globalnode,
                      ignoreNoTool=config.ignoreNoTool,
                      sequence=config.sequence,
                      embedding=config.embedding,
                      INT_TYPE_GOAL=INT_TYPE_GOAL)
    pickle.dump(data, open(filename, 'wb'))
    return data


def convert_symbolicActions_to_goal_json(config, symbolicActions):
    """
    Convert symbolicActions to goal json.
    symbolicActions data format:
        [
            [{'name': 'pushTo', 'args': ['bottle_gray', 'table2']}],
            [{'name': 'pick', 'args': ['orange']}],
            [{'name': 'dropTo', 'args': ['stool', 'tray2']}],
            [{'name': 'moveTo', 'args': ['light']}]
        ]
    """
    goal_json = {'goals': [], 'goal-objects': []}
    objects_processed = {}
    for act_idx, action_item in enumerate(symbolicActions):
        action = action_item[0]
        # * Remove actions which use robot itself as object.
        if config.ACTION_ARGS_NUM[action['name']] == 1:
            if action['args'][0] == 'husky':
                continue
        elif config.ACTION_ARGS_NUM[action['name']] == 2:
            if action['args'][0] == 'husky':
                continue
            elif action['args'][1] == 'husky':
                obj = action['args'][0]
                if obj in objects_processed:
                    del objects_processed[obj]
                continue

        # * Process symbolic actions.
        if action['name'] in ['pushTo', 'dropTo', 'apply']:  # * 2 args
            # * Push obj_A to obj_B.
            # * Obj_A -> object
            # * Obj_B -> position
            # * Action sequence contain these actions may fail due to
            # * the collisions, especially 'dropTo'.
            object = action['args'][0]
            target = ''
            state = []
            position = action['args'][1]

            # * Because the result of these actions are too random
            # * to test, we set a large tolerance.
            # tolerance = 5.0
            tolerance = 2.5 if position == 'light' else 2.0
            if (object in objects_processed) and objects_processed[object]['position'] != position:
                objects_processed[object]['position'] = position
                objects_processed[object]['target'] = target
                objects_processed[object]['tolerance'] = tolerance
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': state, 'position': position,
                    'tolerance': tolerance
                }
            # * Same code as in 'pickNplaceAonB'.
            for obj in list(objects_processed):
                containers = ['box', 'dumpster', 'tray', 'tray2', 'big-tray']
                if obj != object:
                    tmp_position = objects_processed[obj]['position']
                    tmp_target = objects_processed[obj]['target']
                    if tmp_position == object:
                        objects_processed[obj]['position'] = ''
                    if tmp_target == object:
                        if tmp_target in containers:
                            objects_processed[obj]['position'] = target
                        else:
                            objects_processed[obj]['target'] = ''
                    if objects_processed[obj]['target'] == '' and objects_processed[obj][
                        'position'] == '' and len(
                        objects_processed[obj]['state']) == 0:
                        del objects_processed[obj]

        elif action['name'] == 'climbUp':  # * 1 args
            continue

        elif action['name'] == 'pick':  # * 1 args
            continue

        elif action['name'] == 'drop':  # * 1 args
            tmp_pick = None
            tmp_pick_idx = None
            tmp_moveTo = None
            for tmp_idx, tmp_action_item in enumerate(symbolicActions):
                if (tmp_idx < act_idx):
                    # * Find the most resent 'pick' before 'drop'.
                    if (tmp_action_item[0]['name'] == 'pick'):
                        tmp_pick = tmp_action_item[0]
                        tmp_pick_idx = tmp_idx
                    # * Store the robot move (most resent to the 'drop).
                    if (tmp_pick_idx is not None) and (tmp_idx > tmp_pick_idx) and (
                            tmp_action_item[0]['name'] == 'moveTo'):
                        tmp_moveTo = tmp_action_item[0]
            if (tmp_pick_idx is None) or ((tmp_pick_idx - act_idx) == 1):
                # * No 'pick' (only drop) or 'drop' next to 'pick'.
                continue
            elif tmp_moveTo is not None:
                # * 'pick' --> 'moveTo' --> 'drop': same as 'push/pushTo'
                object = tmp_pick['args'][0]
                target = ''
                state = []
                position = tmp_moveTo['args'][0]
                tolerance = 1.5

                if (object in objects_processed) and objects_processed[object][
                    'position'] != position:
                    objects_processed[object]['position'] = position
                    objects_processed[object]['target'] = target
                    objects_processed[object]['tolerance'] = tolerance
                else:
                    objects_processed[object] = {
                        'object': object, 'target': target, 'state': state, 'position': position,
                        'tolerance': tolerance
                    }
                # * Same code as in 'pickNplaceAonB'.
                for obj in list(objects_processed):
                    containers = ['box', 'dumpster', 'tray', 'tray2', 'big-tray']
                    if obj != object:
                        tmp_position = objects_processed[obj]['position']
                        tmp_target = objects_processed[obj]['target']
                        if tmp_position == object:
                            objects_processed[obj]['position'] = ''
                        if tmp_target == object:
                            if tmp_target in containers:
                                objects_processed[obj]['position'] = target
                            else:
                                objects_processed[obj]['target'] = ''
                        if (
                                objects_processed[obj]['target'] == ''
                                and objects_processed[obj]['position'] == ''
                                and len(objects_processed[obj]['state']) == 0
                        ):
                            del objects_processed[obj]
            else:
                # * Execute other actions, like 'clean'
                continue

        elif action['name'] == 'climbDown':  # * 1 args
            continue

        elif action['name'] == 'pickNplaceAonB':  # * 2 args
            # * Pick obj_A and place it on obj_B.
            # * Obj_A -> object
            # * Obj_B -> target
            object = action['args'][0]
            target = action['args'][1]
            state = []
            position = ''
            tolerance = 1.0
            if (object in objects_processed) and objects_processed[object]['target'] != target:
                objects_processed[object]['target'] = target
                objects_processed[object]['position'] = position
                objects_processed[object]['tolerance'] = tolerance
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': state, 'position': position,
                    'tolerance': tolerance
                }

            # * If other object's position object is move away, then remove this position object.
            # * If other object's target object is move away, then remove this target object.
            # * If object's target object is a container, the object will be move away at the same time.
            for obj in list(objects_processed):
                containers = ['box', 'dumpster', 'tray', 'tray2', 'big-tray']
                if obj != object:
                    tmp_position = objects_processed[obj]['position']
                    tmp_target = objects_processed[obj]['target']
                    # * tmp_object near to the object. Now the object is picked and place on another object.
                    if tmp_position == object:
                        objects_processed[obj]['position'] = ''
                    # * tmp_target is a surface.
                    # * If this object is picked, objects on tmp_object will be move away at the same time.
                    if tmp_target == object:
                        if tmp_target in containers:
                            # * Only set position (means near), not target, due to they don't have constraints.
                            objects_processed[obj]['position'] = target
                        else:  # * (tmp_target not in containers):
                            objects_processed[obj]['target'] = ''
                    # * If all others attribute is null, then remove this object.
                    if objects_processed[obj]['target'] == '' and objects_processed[obj][
                        'position'] == '' and len(
                        objects_processed[obj]['state']) == 0:
                        del objects_processed[obj]

        elif action['name'] == 'moveTo':  # * 1 args
            continue
            # object = 'husky'
            # target = ''
            # state = []
            # position = action['args'][0]
            # tolerance = 1.5
            # if (object in objects_processed) and objects_processed[object]['position'] != position:
            #     objects_processed[object]['position'] = position
            # else:
            #     objects_processed[object] = {
            #         'object': object, 'target': target, 'state': state, 'position': position, 'tolerance': tolerance
            #     }

        elif action['name'] == 'clean':  # * 1 args
            object = action['args'][0]
            # if object not in ['dirt', 'water', 'oil']:
            #     continue

            target = ''
            state = 'clean'
            position = ''
            tolerance = 0
            if (object in objects_processed) and objects_processed[object]['state'] != state:
                objects_processed[object]['state'] = [state]
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'stick':  # * 2 args
            object = action['args'][0]
            target = ''
            state = 'stuck'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        # * Factory only.
        elif action['name'] == 'placeRamp':  # * 0 args
            continue

        elif action['name'] == 'moveUp':  # * 0 args
            continue

        elif action['name'] == 'moveDown':  # * 0 args
            continue

        elif action['name'] == 'drill':  # * 1 args
            object = action['args'][0]
            target = ''
            state = 'fixed'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'cut':  # * 2 args
            object = action['args'][0]
            target = action['args'][1]
            state = 'cut'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'print':  # * 1 args
            continue

        elif action['name'] == 'drive':  # * 2 args
            object = action['args'][0]
            target = action['args'][1]
            state = 'driven'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'fuel':  # * 2 args
            object = action['args'][0]
            target = action['args'][1]
            state = 'fueled'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'weld':  # * 1 args
            object = action['args'][0]
            target = ''
            state = 'welded'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

        elif action['name'] == 'paint':  # * 1 args
            object = action['args'][0]
            target = ''
            state = 'painted'
            position = ''
            tolerance = 0
            if object in objects_processed:
                if config.INVERSE_STATE[state] in objects_processed[object]['state']:
                    objects_processed[object]['state'] = [
                        state if x == config.INVERSE_STATE[state] else x
                        for x in objects_processed[object]['state']
                    ]
                else:
                    objects_processed[object]['state'].append(state)
            else:
                objects_processed[object] = {
                    'object': object, 'target': target, 'state': [state], 'position': position,
                    'tolerance': tolerance
                }

    # * Add data to goal_json.
    for key, value in objects_processed.items():
        goal_json['goals'].append(value)

    for data in goal_json['goals']:
        if data['object'] not in goal_json['goal-objects']:
            goal_json['goal-objects'].append(data['object'])
        if data['target'] != '' and data['target'] not in goal_json['goal-objects']:
            goal_json['goal-objects'].append(data['target'])
        if data['position'] != '' and data['position'] not in goal_json['goal-objects']:
            goal_json['goal-objects'].append(data['position'])

    return goal_json


def check_goal(config, actionSeq, goal_json, world_num, start_node, set_goal_json, action_set,
               only_res):
    """Check goal. (Gaol vector format: json)"""
    e = config.embeddings
    approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=set_goal_json,
                      INPUT_DATAPOINT=start_node)
    res, g, err = False, None, None
    if action_set:
        res, g, err = approx.execAction(config, actionSeq, e, ACTION_SET=action_set,
                                        ONLY_RES=only_res)
    else:
        for action_gt in actionSeq:
            res, g, err = approx.execAction(config, action_gt, e, ONLY_RES=only_res)
            if err != '':
                return False, g, err
    return res, g, err


def convert_graph_to_datapoint(
        config,
        graph_data_path,
        dataset_save_dir=None,
        max_sequence_length=None,
        max_data_num=None,
        start_index=0,
        target_length=None,
        check_json=True
):
    with open(graph_data_path, 'rb') as f:
        DG = pickle.load(f)
    world_name = os.path.split(os.path.split(graph_data_path)[0])[1]
    # * Datapoint save dir.
    if config.domain == 'home':
        data_save_dir = './new_dataset/home_datapoint/' + world_name + '/'
    else:
        data_save_dir = './new_dataset/factory_datapoint/' + world_name + '/'
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    # * Goal json finle save dir.
    goal_save_dir = data_save_dir.replace('_datapoint', '_json')
    if not os.path.exists(goal_save_dir):
        os.makedirs(goal_save_dir)

    nodes_list = list(DG.nodes)
    # * Get all path.
    data_index = 0
    for i in range(start_index, len(nodes_list) - 1):
        # Save multi-steps symbolicActions datapoint.
        for j in range(i + 1, len(nodes_list)):
            cprint('--' * 20)
            cprint('Start node {} --  End node {}'.format(i, j), 'green')
            start_node_id = nodes_list[i]
            end_node_id = nodes_list[j]
            # * Get path from source to target.
            for path in nx.all_simple_paths(DG, source=start_node_id, target=end_node_id):
                if (target_length is not None) and (len(path) != target_length):
                    continue
                elif (max_sequence_length is not None) and (len(path) > max_sequence_length):
                    continue

                # * ---------------------------------------------------
                # * Get node state (datapoint).
                state_list = []
                for p in path[:-1]:
                    state_list.append(DG.nodes[p]['state'])
                total_dp = merge_datapoint(state_list)  # * Merge node state.
                total_dp.goal = data_index  # * Change datapoint goal.

                # * ---------------------------------------------------
                # * Get edge actions.
                # * SymbolicActions stored in datapoint is the action
                # * from a parent node of this node. For the node which
                # * has only one parent node,
                # * hl_actions == total_dp.symbolicActions, but when we
                # * add graph's edges, the node may has many parent,
                # * so we must use actions stored in the edge.
                hl_actions = []
                for idx in range(len(path) - 1):
                    action = DG[path[idx]][path[idx + 1]]['action']
                    hl_actions.append(action['actions'])
                goal_json = convert_symbolicActions_to_goal_json(config, hl_actions)
                # * Add new attributes.
                goal_json.update({'nodes': list(path), 'actions': hl_actions})
                if check_json:
                    if len(goal_json['goals']) != 0:
                        # * Save datapoint.
                        graph_datapoint_save_path = data_save_dir + str(data_index) + '.datapoint'
                        print(graph_datapoint_save_path)
                        with open(graph_datapoint_save_path, 'wb') as f:
                            pickle.dump(total_dp, f)
                        # * Save goal json file.
                        goal_save_path = graph_datapoint_save_path.replace(
                            '_datapoint', '_json'
                        ).replace('.datapoint', '.json')
                        with open(goal_save_path, 'w') as f:
                            json.dump(goal_json, f)
                        data_index += 1
                        if (max_data_num is not None) and (data_index >= max_data_num):
                            return
                else:
                    # * Save datapoint.
                    graph_datapoint_save_path = data_save_dir + str(data_index) + '.datapoint'
                    print(graph_datapoint_save_path)
                    with open(graph_datapoint_save_path, 'wb') as f:
                        pickle.dump(total_dp, f)
                    # * Save goal json file.
                    goal_save_path = graph_datapoint_save_path.replace(
                        '_datapoint', '_json'
                    ).replace('.datapoint', '.json')
                    with open(goal_save_path, 'w') as f:
                        json.dump(goal_json, f)
                    data_index += 1
                    if (max_data_num is not None) and (data_index >= max_data_num):
                        return

    # # * -------------------------------------------------------------
    # # * Generate data.
    # data = load_dataset(config, root_dir='./new_dataset/home_datapoint', INT_TYPE_GOAL=True)


def get_path_data(
        config,
        DG_graph,
        data_idx,
        world_name,
        world_num,
        target_length,
        max_sequence_length,
        min_sequence_length,
        check_json,
        STATE_FORMAT_GOAL
):
    start_node_id, end_node_id = data_idx
    # cprint('Start node {} --  End node {}'.format(start_node_id, end_node_id), 'green')

    rt_data = []
    # * Get path from source to target.
    for path in nx.all_simple_paths(DG_graph, source=start_node_id, target=end_node_id):
        if (target_length is not None) and (len(path) != target_length):
            continue
        if (max_sequence_length is not None) and (len(path) > max_sequence_length):
            continue
        if (min_sequence_length is not None) and (len(path) < min_sequence_length):
            continue

        # * Get edge actions.
        # * SymbolicActions stored in datapoint is the action from a
        # * parent node of this node. For the node which has only one
        # * parent node, hl_actions == total_dp.symbolicActions, but
        # * when we add graph's edges, we must use actions stored in
        # * the edge. The added edges are be predicted by the
        # * model(node_1_state['End'], node_2_state['end']).
        hl_actions = []
        for idx in range(len(path) - 1):
            action = DG_graph[path[idx]][path[idx + 1]]['action']
            hl_actions.append(action['actions'])
        actionSeq = []
        for action in hl_actions:
            if not (str(action[0]) == 'E' or str(action[0]) == 'U'):
                actionSeq.append(action[0])
        start_data = DG_graph.nodes[path[0]]['state']
        if not STATE_FORMAT_GOAL:
            goal_json = convert_symbolicActions_to_goal_json(config, hl_actions)
            if len(goal_json['goals']) == 0: continue
            res, _, _ = check_goal(config, actionSeq, goal_json, world_num, start_data,
                                   set_goal_json=True, action_set=False, only_res=True)
            if res:
                rt_data.append(
                    {
                        'world_name': world_name,
                        'graph_path': os.path.abspath(graph_data_path),
                        'graph_file': os.path.basename(graph_data_path),
                        'nodes': tuple(path),
                    }
                )
        else:
            rt_data.append(
                {
                    'world_name': world_name,
                    'graph_path': os.path.abspath(graph_data_path),
                    'graph_file': os.path.basename(graph_data_path),
                    'nodes': tuple(path),
                }
            )
    return rt_data


def _path_to_dataset_item(
        config,
        DG_graph,
        path,
        world_name,
        world_num,
        STATE_FORMAT_GOAL,
        graph_data_path=None
):
    hl_actions = []
    for idx in range(len(path) - 1):
        action = DG_graph[path[idx]][path[idx + 1]]['action']
        hl_actions.append(action['actions'])

    actionSeq = []
    for action in hl_actions:
        if not (str(action[0]) == 'E' or str(action[0]) == 'U'):
            actionSeq.append(action[0])

    start_data = DG_graph.nodes[path[0]]['state']
    if not STATE_FORMAT_GOAL:
        goal_json = convert_symbolicActions_to_goal_json(config, hl_actions)
        if len(goal_json['goals']) == 0:
            return None
        res, _, _ = check_goal(
            config,
            actionSeq,
            goal_json,
            world_num,
            start_data,
            set_goal_json=True,
            action_set=False,
            only_res=True,
        )
        if not res:
            return None

    item = {'world_name': world_name, 'nodes': tuple(path)}
    if graph_data_path is not None:
        item['graph_path'] = os.path.abspath(graph_data_path)
        item['graph_file'] = os.path.basename(graph_data_path)
    return item


def _collect_forward_paths(DG_graph, start_node_id, min_nodes, max_nodes):
    candidate_paths = []

    def dfs(curr_path):
        if len(curr_path) > max_nodes:
            return
        if len(curr_path) >= min_nodes:
            candidate_paths.append(tuple(curr_path))
        for next_node in DG_graph.successors(curr_path[-1]):
            dfs(curr_path + [next_node])

    dfs([start_node_id])
    return candidate_paths


def generate_node_list_from_graph(
        config,
        graph_data_path,
        file_save_path=None,
        min_sequence_length=None,
        max_sequence_length=None,
        target_length=None,
        start_index=0,
        check_json=True,
        embedding_file_path='../jsons/embeddings/conceptnet.vectors',
        STATE_FORMAT_GOAL=False
):
    """
    Generate node list from single graph (write to file if node_list_save_path is not None).
    Input: graph save path
    Output: node list (datapoint idx list).
    Return data format:
        [
            {'world_name': 'world_home0', 'nodes': (1, 2, 3)},
            {'world_name': 'world_home0', 'nodes': (2, 3, 4)},
            {'world_name': 'world_home0', 'nodes': (2, 3, 4)},
            ...
        ]
    Convert graph to datapoint directly will generate a huge data.
    A graph with M nodes will generate nearly M * N datapoints, which N = max_sequence_length.
    (1.7GB graph --> 120GB datapoints)
    """
    cprint('Processing data: {}'.format(graph_data_path), 'green')
    # with open(embedding_file_path) as handle:  # * Embedding file.
    #     e = json.load(handle)
    e = config.embeddings

    with open(graph_data_path, 'rb') as f:  # * Directed graph file.
        DG = pickle.load(f)
    # * File save dir.
    if file_save_path is not None:
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)

    world_name = os.path.split(os.path.split(graph_data_path)[0])[1]
    world_num = _safe_numeric_suffix(world_name, default=0)
    rt_data = []
    nodes_list = list(DG.nodes)

    # * ---------------------------------------------------------------
    # # * Change 'End' state to dgl graph and save to the directed graph.
    # # for node_id in tqdm(nodes_list):
    # for node_id in nodes_list:
    #     # * Data format: DG.nodes[node_id]
    #     # * Define: collect_data.py
    #     # * {
    #     # *     'state': <src.datapoint.Datapoint object at 0x0000025D4711F2C8>,
    #     # *     'id': 0,
    #     # *     'world_state': 1,
    #     # *     'pre_constraints': {},
    #     # *     'child_actions': [{'name': 'moveTo', 'args': ['bottle_gray']}]
    #     # * }
    #     tmp_dp = DG.nodes[node_id]['state']
    #     for i in range(len(tmp_dp.metrics)):
    #         if tmp_dp.actions[i] == 'End':
    #             dgl_graph = convertToDGLGraph(config, tmp_dp.getGraph(i, embeddings=e)['graph_' + str(i)], False, -1)
    #             DG.nodes[node_id]['dgl_graph'] = dgl_graph

    # * ---------------------------------------------------------------
    # * Collect forward paths directly from each start node.
    min_nodes = min_sequence_length if min_sequence_length is not None else 1
    max_nodes = max_sequence_length if max_sequence_length is not None else len(nodes_list)
    for start_node_id in tqdm(nodes_list[start_index:]):
        candidate_paths = _collect_forward_paths(DG, start_node_id, min_nodes, max_nodes)
        for path in candidate_paths:
            if (target_length is not None) and (len(path) != target_length):
                continue
            item = _path_to_dataset_item(
                config,
                DG,
                path,
                world_name,
                world_num,
                STATE_FORMAT_GOAL,
                graph_data_path=graph_data_path,
            )
            if item is not None:
                rt_data.append(item)

    # with tqdm(total=len(nodes_list)) as pbar:
    #     end_id = len(nodes_list) - 1
    #     while (end_id > 0):
    #         previous_max_length_data = None
    #         for start_id in range(start_index, end_id):  # * start_index = 0
    #             tmp_data = get_path_data(config, DG, [start_id, end_id], world_name, world_num, target_length,
    #                                      max_sequence_length, min_sequence_length, check_json, STATE_FORMAT_GOAL)
    #             if (len(tmp_data) > 0):
    #                 if previous_max_length_data is None:
    #                     rt_data += tmp_data
    #                     previous_max_length_data = tmp_data[0]['nodes']
    #                 else:
    #                     for td in tmp_data:
    #                         if set(previous_max_length_data) > set(td['nodes']):
    #                             pass
    #                         else:
    #                             rt_data += td
    #         end_id -= 1
    #         pbar.update(1)

    # j_nums = multiprocessing.cpu_count() * 3
    # res = Parallel(n_jobs=j_nums)(delayed(get_path_data)(config, DG, data_idx, world_name, world_num, target_length,
    #                                                      max_sequence_length, min_sequence_length, check_json,
    #                                                      STATE_FORMAT_GOAL) for data_idx in tqdm(start_end_index_list))
    # for tmp_data in res:
    #     rt_data += tmp_data

    if file_save_path is not None:
        with open(file_save_path, 'wb') as f:
            pickle.dump(rt_data, f)

    return rt_data


def generate_node_list_from_graphs(
        config,
        graphs_dir,
        file_save_path=None,
        max_sequence_length=None,
        target_length=None,
        start_index=0,
        check_json=True,
        STATE_FORMAT_GOAL=False
):
    """
    Generate node list from all graph in dir (write to file if node_list_save_path is not None).
    Input: graphs save dir
    Output: node list (datapoint idx list).
    Return data format:
        [
            {'world_name': 'world_home0', 'nodes': (1, 2, 3)},
            {'world_name': 'world_home1', 'nodes': (2, 3, 4)},
            {'world_name': 'world_home2', 'nodes': (2, 3, 4)},
            ...
        ]
    """
    graph_paths = get_data_files(graphs_dir, data_format='.graph')
    cprint('Graph num: {}.'.format(len(graph_paths)), 'green')
    rt_data = []
    for graph_path in graph_paths:
        tmp_data = generate_node_list_from_graph(
            config,
            graph_path,
            None,
            max_sequence_length=max_sequence_length,
            target_length=target_length,
            start_index=start_index,
            check_json=check_json,
            STATE_FORMAT_GOAL=STATE_FORMAT_GOAL
        )
        cprint('Data num: {}.'.format(len(tmp_data)), 'green')
        rt_data.extend(tmp_data)
    # * File save dir.
    if file_save_path is not None:
        if not os.path.exists(file_save_path):
            os.makedirs(file_save_path)
        with open(file_save_path, 'wb') as f:
            pickle.dump(rt_data, f)
    return rt_data


def action2vec_cons_wide(config, action, num_objects, num_states, width=1):
    actionArray = torch.zeros(len(config.possibleActions) * width)
    tmp_idx = config.possibleActions.index(action['name'])
    actionArray[tmp_idx * width: (tmp_idx + 1) * width] = 1

    predicate1 = torch.zeros(num_objects * width)
    predicate2 = torch.zeros(num_objects * width)
    predicate3 = torch.zeros(num_states * width)
    if len(action['args']) == 1:
        tmp_idx = config.object2idx[action['args'][0]]
        predicate1[tmp_idx * width: (tmp_idx + 1) * width] = 1
    elif len(action['args']) == 2:
        idx_1 = config.object2idx[action['args'][0]]
        idx_2 = config.object2idx[action['args'][1]]
        predicate1[idx_1 * width: (idx_1 + 1) * width] = 1
        predicate2[idx_2 * width: (idx_2 + 1) * width] = 1
    return torch.cat((actionArray, predicate1, predicate2, predicate3), 0)


def convert_symbolicActions_to_hlActions(symbolicActions):
    hlActions = []
    for i in range(len(symbolicActions)):
        action = symbolicActions[i]
        if str(action[0]) == 'E' or str(action[0]) == 'U':
            del hlActions[-1]
        else:
            hlActions.append(action[0])
    return hlActions


def allowed_file(filename, allow_formats='datapoint'):
    return '.' in filename and filename.rsplit('.')[-1] in allow_formats


def get_data_files(data_dir, data_format='.datapoint'):
    data_paths = []
    all_files = list(os.walk(data_dir))
    for path, dirs, files in all_files:
        if (len(files) > 0):
            for file in files:
                file_path = os.path.join(path, file)
                data_paths.append(file_path)
    new_data_paths = []
    for data_path in data_paths:
        if allowed_file(data_path, data_format):
            new_data_paths.append(data_path)
    return sorted(new_data_paths)


def check_goal_datapoint_symbolicActions(
        config,
        data_dir,
        embedding_file_path='jsons/embeddings/conceptnet.vectors',
        verbose=False
):
    data_paths = get_data_files(data_dir)
    data_num = len(data_paths)
    print('Datapoint num: {}'.format(data_num))
    if data_num == 0:
        raise Exception('Data number is 0 in {}.'.format(data_dir))

    # with open(embedding_file_path) as handle:
    #     e = json.load(handle)
    e = config.embeddings

    correct = 0
    incorrect = 0
    error = 0
    den = 0
    for data_path in tqdm(data_paths):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        # * Get daapoint symbolicActions and convert to high level actions.
        # hlActions = convert_symbolicActions_to_hlActions(data.symbolicActions)
        # inp = {'actions': hlActions}

        world_name = os.path.split(os.path.split(data_path)[0])[1]
        world_num = _safe_numeric_suffix(world_name, default=0)
        # print('world_num = {}'.format(world_num))
        goal_json_path = data_path.replace('_datapoint', '_json').replace('.datapoint', '.json')
        with open(goal_json_path, 'rb') as f:
            goal_json = json.load(f)
        hlActions = convert_symbolicActions_to_hlActions(goal_json['actions'])
        inp = {'actions': hlActions}

        approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                          INPUT_DATAPOINT=data)
        res, g, err = approx.execAction(config, inp, e, ACTION_SET=True, ONLY_RES=True)

        if res:
            correct += 1
        elif err == '':
            incorrect += 1
            print('Incorrect: {}'.format(data_path))
        elif err != '':
            error += 1
            print('Error: {} | {}'.format(data_path, err))
        den = correct + incorrect + error
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den),
          (error * 100 / den))


def convert_goal_json_to_vec(config, goal_json, GOAL_OBJ_VEC=False):
    # * Goal objects and vectors.
    goals_define = goal_json['goals']

    # * goal2vec
    goal2vec = torch.zeros(config.PRETRAINED_VECTOR_SIZE * 3 + config.N_STATES)
    if len(goals_define) == 0:
        goalObjects2vec = None
        if GOAL_OBJ_VEC:
            goalObjects2vec = torch.zeros(config.PRETRAINED_VECTOR_SIZE, dtype=torch.float32)
        return goal2vec, goalObjects2vec

    for goal in goals_define:
        object = goal['object']
        target = goal['target']
        states = goal['state']
        position = goal['position']
        object_vec = torch.tensor(config.object2vec[object], dtype=torch.float32) \
            if object != '' else torch.zeros(config.PRETRAINED_VECTOR_SIZE)
        target_vec = torch.tensor(config.object2vec[target], dtype=torch.float32) \
            if target != '' else torch.zeros(config.PRETRAINED_VECTOR_SIZE)
        position_vec = torch.tensor(config.object2vec[position], dtype=torch.float32) \
            if position != '' else torch.zeros(config.PRETRAINED_VECTOR_SIZE)
        state_vec = torch.zeros(config.N_STATES, dtype=torch.float)
        for state in states:
            if state == 'stuck':
                state = 'sticky'
            canonical_state = config.canonical_state_name(state)
            if canonical_state is None:
                continue
            idx = config.state2indx[canonical_state]
            state_vec[idx] = 1
        sub_goal = torch.cat((object_vec, target_vec, position_vec, state_vec))
        goal2vec += sub_goal
    goal2vec /= len(goals_define)
    goalObjects2vec = None
    if GOAL_OBJ_VEC:
        # * goalObjects2vec: Vector_Sum(embeddings['goal-objects in goal json']) / object_num
        goal_object_vec = np.zeros(300)
        for j in goal_json['goal-objects']:
            goal_object_vec += config.object2vec[j]
        if len(goal_json['goal-objects']) == 0:
            goalObjects2vec = torch.zeros(config.PRETRAINED_VECTOR_SIZE, dtype=torch.float32)
            return goal2vec, goalObjects2vec
        goal_object_vec /= len(goal_json['goal-objects'])
        goalObjects2vec = torch.tensor(goal_object_vec, dtype=torch.float32)

    return goal2vec, goalObjects2vec


# * Borrows from utils_dataset.py and modified.
def getToolSequence(config, actionSeq):
    """
    Returns the sequence of tools that were used in the plan.
    """
    toolSeq = ['no-tool'] * len(actionSeq)
    currentTool = 'no-tool'
    for i in range(len(toolSeq) - 1, -1, -1):
        for obj in actionSeq[i]['args']:
            if obj in config.TOOLS2:
                currentTool = obj
                break
        toolSeq[i] = currentTool
    return toolSeq


def getGlobalID(config, dp):
    maxID = 0
    for i in dp.metrics[0].keys():
        maxID = max(maxID, config.object2idx[i])
    return maxID + 1


def getDGLSequence(
        config,
        datapoint,
        e,
        hl_actions,
        goal_objs=None,
        globalNode=False,
        ignoreNoTool=False,
        DGL_GRAPHS=None,
        STATE_FLAG='End',
        DATA_ARGUMENT=False
):
    """
    Returns the entire sequence of graphs and actions from the plan
    in the provided datapoint.
    """
    datapoint.config = config  # * Add new attributes
    time = datapoint.totalTime()
    tools = datapoint.getTools(not ignoreNoTool)
    if ignoreNoTool and len(tools) == 0: return None
    goal_num = datapoint.goal
    world_num = _safe_numeric_suffix(datapoint.world, default=0)
    actionSeq = []
    graphSeq = []
    for action in hl_actions:
        if not (str(action[0]) == 'E' or str(action[0]) == 'U'): actionSeq.append(action[0])

    if (DGL_GRAPHS is not None) and (not DATA_ARGUMENT):
        graphSeq = DGL_GRAPHS
    else:
        for i in range(len(datapoint.metrics)):
            if datapoint.actions[i] == STATE_FLAG:
                graphSeq.append(
                    convertToDGLGraph(
                        config,
                        datapoint.getGraph(i, goal_objs=goal_objs, embeddings=e,
                                           DATA_ARGUMENT=DATA_ARGUMENT)['graph_' + str(i)],
                        globalNode,
                        getGlobalID(config, datapoint) if globalNode else -1)
                )

    # assert len(actionSeq) == len(graphSeq), 'len(actionSeq) != len(graphSeq)'
    toolSeq = getToolSequence(config, actionSeq)
    return (goal_num, world_num, toolSeq, (actionSeq, graphSeq), time)
