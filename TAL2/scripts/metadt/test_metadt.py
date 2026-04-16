"""
@Project     ：TAL_2024
@File        ：test_metadt.py
@Author      ：Xianqi-Zhang
@Date        ：2024/12/8
@Last        : 2024/12/8
@Description : 
"""

import numpy as np
import pickle
import torch
from tqdm import tqdm
from termcolor import cprint

from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs import approx
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.action_proposal_network import vec2action_grammatical
from src.datasets.graph_dataset import GraphDataset_State

from src.baselines.metadt.utils_model import load_context_checkpoint
from src.baselines.metadt.utils_new.metadt_model import MetaDecisionTransformer
from src.baselines.metadt.utils_new.utils import load_context_model, load_metadt_checkpoint


def embedding_rewards(reward, reward_codebook, device):
    idx = reward_codebook.index(reward)
    r_embed = torch.zeros((1, len(reward_codebook))).to(device)
    r_embed[0, idx] = 1
    return r_embed


def postprocess_metadt(action_name, object_1, object_2, state, mask):
    action_name_dim = 11
    object_dim = 36
    state_dim = 28
    action_name = action_name.reshape(-1, action_name_dim)[mask.reshape(-1) > 0]
    object_1 = object_1.reshape(-1, object_dim)[mask.reshape(-1) > 0]
    object_2 = object_2.reshape(-1, object_dim)[mask.reshape(-1) > 0]
    state = state.reshape(-1, state_dim)[mask.reshape(-1) > 0]
    action_preds = torch.cat([action_name, object_1, object_2, state], dim=-1)
    action_preds = action_preds[-1, :].detach()
    return action_preds


def test_policy_metadt(config, dataset, context_model, metadt, num_objects=0, TQDM=False,
                       STATE_FORMAT_GOAL=True, INIT_DATAPOINT=True):
    context_model.eval()
    metadt.eval()
    action_dim = 111
    with open('./checkpoints/metadt/reward_codebook.pkl', 'rb') as f:
        reward_codebook = pickle.load(f)
    e = config.embeddings
    context_len = config.context_len
    device = config.device
    correct, incorrect, error = 0, 0, 0
    lenHuman, lenModel = [], []
    data_num = {}
    data_correct_num = {}

    data_container = tqdm(dataset, desc='Policy Testing', ncols=80) if TQDM else dataset
    for data_item in data_container:
        if STATE_FORMAT_GOAL:  # * GraphDataset_State
            graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node = data_item
        else:  # * GraphDataset
            graphSeq, goal2vec, goalObjects2vec, actionSeq, action2vec, goal_json, world_name, start_node = data_item

        # * Store action length to dict.
        if str(len(actionSeq)) in data_num:
            data_num[str(len(actionSeq))] += 1
        else:
            data_num[str(len(actionSeq))] = 1
        if str(len(actionSeq)) not in data_correct_num:
            data_correct_num[str(len(actionSeq))] = 0

        world_num = int(world_name[-1])
        plan_len = len(actionSeq)
        lenHuman.append(plan_len)
        predActionSeq = []  # * Initialize
        graphSeq_t = []
        # * Initialize environment.
        if INIT_DATAPOINT:
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True,
                              INPUT_DATAPOINT=start_node)
            graphSeq_t.append(graphSeq[0])
        else:
            # * Use wold_home1 to test.
            # world_num = 1
            approx.initPolicy(config, config.domain, goal_json, world_num, SET_GAOL_JSON=True)
            init_g = approx.getInitializeDGLGraph(config)
            if device is not None:
                init_g = init_g.to(device)
            graphSeq_t.append(init_g)

        # * Test model.
        target_return = 100
        return_scale = 100  # * Same as MetaDataset setting.
        min_distance = np.inf
        action2vec_t = [torch.zeros(action_dim).to(device)]
        rewards_t = [torch.tensor([1, 0, 0]).unsqueeze(0).float().to(device)]
        rtg_t = [torch.tensor([target_return / return_scale]).to(device)]
        context_t = []
        while True:
            # * -----------------------------------------------------------
            # * Process data.
            task = goal2vec
            states = graphSeq_t[-context_len:]
            tlen = len(states)
            dtlen = context_len - tlen
            mask = torch.cat([torch.zeros(dtlen), torch.ones(tlen)], dim=0).unsqueeze(0).to(device)
            start_time_steps = max(0, len(graphSeq_t) - context_len)
            timesteps = torch.arange(
                start=start_time_steps, end=start_time_steps + tlen, step=1
            ).to(device)
            actions = torch.stack(action2vec_t[-context_len:])
            rewards = torch.cat(rewards_t[-context_len:])
            rtg = torch.stack(rtg_t[-context_len:])
            # if len(action2vec_t) > 0:
            #     actions = torch.stack(action2vec_t[-context_len:])
            #     rewards = rewards_t[-context_len:]
            #     rtg = torch.tensor(rtg_t[-context_len:]).to(device)
            # else:
            #     actions = torch.zeros(1, action_dim).to(device)
            #     rewards = [torch.tensor([1, 0, 0]).unsqueeze(0).float().to(device)]   # * 0
            #     rtg = torch.tensor([target_return / return_scale]).unsqueeze(0).to(device)

            with torch.no_grad():
                context_info = context_model(states, actions, rewards, task)
                context_t.append(context_info)
            contexts = torch.stack(context_t[-context_len:])

            # * -----------------------------------------------------------
            with torch.no_grad():
                action_name, object_1, object_2, state = metadt(
                    task, contexts, states, actions, rtg, timesteps, mask, None
                )
                action_preds = postprocess_metadt(action_name, object_1, object_2, state, mask)

            action2vec_t.append(action_preds)
            action_hl = vec2action_grammatical(
                config, action_preds, num_objects, len(config.possibleStates), config.idx2object
            )

            # * Execute actions!!!
            res, g, err = approx.execAction(config, action_hl, e)

            # * Check success.
            if res:
                data_correct_num[str(len(actionSeq))] += 1
                correct += 1
                lenModel.append(len(predActionSeq))
                break
            elif err == '' and len(predActionSeq) > 60:
                incorrect += 1
                break
            elif err != '':
                error += 1
                break

            # * Update variables.
            predActionSeq.append(action_hl)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)
            dis = torch.sum(torch.abs(task.ndata['feat'] - g.ndata['feat']))
            if dis < min_distance:
                min_distance = dis
                reward = 1
            else:
                reward = 0
            rewards_t.append(embedding_rewards(reward, reward_codebook, config.device))
            target_return -= reward
            rtg_t.append(torch.tensor([target_return / return_scale]).to(device))

    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())
    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print(
        'Correct, Incorrect, Error: ',
        (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den)
    )
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def main():
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    cprint('Set random seed = {}'.format(rnd_seed), 'green')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = EnvironmentConfig(args)
    config.context_len = 3
    reward_type = 'classification'
    dataset_num = 2  # * !!!

    # * ---------------------------------------------------------------
    # * Dataset.
    graphs_dir = './data/{}/home/'.format(dataset_num)
    train_data_path = './data/{}/train_dataset.pkl'.format(dataset_num)
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    print('Train data num: {}'.format(len(train_dataset)))
    val_data_path = './data/{}/val_dataset.pkl'.format(dataset_num)
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)
    print('Val data num: {}'.format(len(val_dataset)))
    test_data_path = './data/{}/test_dataset.pkl'.format(dataset_num)
    test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)
    print('Test data num: {}'.format(len(test_dataset)))

    # * ---------------------------------------------------------------
    # * Model.
    context_ckpt_path = './checkpoints/metadt/context/context_models_dataset_{}.pt'
    context_ckpt_path = context_ckpt_path.format(dataset_num)
    context_model, reward_model = load_context_model(config, device, reward_type, False)
    context_model, _ = load_context_checkpoint(context_model, reward_model, context_ckpt_path)

    metadt_ckpt_path = './checkpoints/metadt/metadt/metadt_models_dataset_{}.pt'
    metadt_ckpt_path = metadt_ckpt_path.format(dataset_num)
    metadt = MetaDecisionTransformer(config, device).to(device).train()
    metadt = load_metadt_checkpoint(metadt, metadt_ckpt_path)

    # * ---------------------------------------------------------------
    # * Policy test.
    # seeds = [0, 1, 42]
    seeds = [0]
    for seed in seeds:
        setup_seed(seed=seed)
        print('-----------------------------------------')
        print('Random seed: {}'.format(seed))
        print('-----------------------------------------')

        print('\n\nTraining set...')
        test_policy_metadt(config, train_dataset, context_model, metadt, config.num_objects)
        print('\n\nVal set...')
        test_policy_metadt(config, val_dataset, context_model, metadt, config.num_objects)
        print('\n\nTest set...')
        test_policy_metadt(config, test_dataset, context_model, metadt, config.num_objects)


if __name__ == '__main__':
    main()
