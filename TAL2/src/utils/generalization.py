# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File             : generalization.py
@Project          : M3
@Time             : 2022/7/21 14:00
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/7/21 14:00     
@Version          : 1.0  
@Desciption       : None   
"""
import os
import time
import pickle
import colorama
import networkx as nx
from copy import deepcopy
from termcolor import cprint

from src.envs import husky_ur5 as env

colorama.init()


def check_action(config, action):
    # * {'name': 'climbDown', 'args': ['stool']}
    # * {'name': 'pickNplaceAonB', 'args': ['milk', 'fridge']}
    if (action['name'] not in config.possibleActions):
        return False
    for obj in action['args']:
        if obj not in config.all_objects:
            return False
    return True


def get_actions_from_datapoint(config, datapoint_save_path):
    with open(datapoint_save_path, 'rb') as f:
        dp = pickle.load(f)
    actionSeq = []
    for action in dp.symbolicActions:
        if not (str(action[0]) == 'E' or str(action[0]) == 'U'):
            # print(action[0])
            if check_action(config, action[0]):
                actionSeq.append(action[0])
            else:
                return []
    return actionSeq


def collect_data_with_actions(config, actionSeq, file_save_path, ignore_error_data=False):
    if os.path.exists(file_save_path):
        return

    # * Init environment.
    print('...' * 30)
    print('Creating new world: {}'.format(config.world))
    print('...' * 30)

    time.sleep(5)
    env.start(config)

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
        world_state=state_id,  # * Pybullet state id. State id is used to restore the environment.
        pre_constraints=previous_constraints,
        child_actions=[]
    )  # * Add node.

    # * ---------------------------------------------------------------
    # * Collect data.
    cursor_start = node_id
    cursor_end = node_id + 1
    previous_state_datapoint = deepcopy(env_state_datapoint)
    for hl_item in actionSeq:
        hl_actions = {'actions': [hl_item]}
        cprint(str(hl_actions), 'green')

        try:
            done = env.execute_collect_data(config, hl_actions, goal_file=None, saveImg=False)
        except Exception as e:
            cprint(str(e), 'red')
            if ignore_error_data:
                env.destroy()  # * Close environment.
                return
            cprint('Reload previous state.'.format(e), 'green')
            # * Reload world state.
            env.restoreState(state_id, previous_constraints, previous_state_datapoint)
            env.resetDatapoint(config, previous_state_datapoint)
            continue

        env_state_datapoint = env.getDatapoint(config, RESET_DATAPOINT=True)
        previous_state_datapoint = deepcopy(env_state_datapoint)
        assert len(env_state_datapoint.metrics) == len(env_state_datapoint.actions), \
            '[Error]: len(datapoint.metrics) != len(datapoint.actions)'
        state_id, previous_constraints = env.saveState()  # * Save world state.
        previous_constraints = deepcopy(previous_constraints)
        DG.add_node(
            cursor_end,
            state=env_state_datapoint,
            id=cursor_end,
            parent_id=cursor_start,
            world_state=state_id,
            pre_constraints=previous_constraints,
            child_actions=[],
        )  # * Add node.
        DG.add_weighted_edges_from([(cursor_start, cursor_end, 1.0), ])  # * Add edge.
        DG[cursor_start][cursor_end]['action'] = hl_actions  # * Set edge attribute.
        DG.nodes[cursor_start]['child_actions'].append(hl_item)

        cursor_start = cursor_end
        cursor_end += 1

    # * ---------------------------------------------------------------
    # * Save data.
    file_save_folder = file_save_path.rsplit('\\', 1)[0]
    if not os.path.exists(file_save_folder):
        os.makedirs(file_save_folder)

    with open(file_save_path, 'wb') as f:
        pickle.dump(DG, f)

    env.destroy()  # * Close environment.
