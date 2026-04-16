'''
Code:
https://github.com/BY571/CQL
'''

import torch
from src.envs import approx
from tqdm import tqdm

from src.tal.utils_training import get_model, load_model, save_model


def _safe_numeric_suffix(name, default=0):
    digits = ''.join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else default


# def save(args, save_name, model, wandb, ep=None):
#     import os
#     save_dir = './checkpoints/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     if not ep == None:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
#     else:
#         torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")
#         wandb.save(save_dir + args.run_name + save_name + ".pth")


def save(config, args, wandb, save_name, model, optimizer, epoch_num, accuracy_list, ep=None):
    import os
    save_dir = './checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if ep is not None:
        save_model(config, model, optimizer, epoch_num, accuracy_list)
        wandb.save(save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        save_model(config, model, optimizer, epoch_num, accuracy_list)
        wandb.save(save_dir + args.run_name + save_name + ".pth")


def collect_random(env, dataset, num_samples=200):
    state = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state = env.reset()


####################################################################################################################
def generate_model(config, lr=1e-4, target_net=False):
    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    # model, optimizer, epoch, accuracy_list = load_model(config, seqTool + model.name + '_Trained', model, lr=lr)
    if target_net:
        file_name = seqTool + model.name + '_target_Trained'
    else:
        file_name = seqTool + model.name + '_Trained'
    model, optimizer, epoch, accuracy_list = load_model(config, file_name, model, lr=lr, strict=False)
    model = model.to(config.device)
    return model, optimizer, epoch, accuracy_list


def test_policy_graph_dataset_cql(config, dataset, agent, TQDM=True, STATE_FORMAT_GOAL=True, INIT_DATAPOINT=True):
    e = config.embeddings

    correct, incorrect, error = 0, 0, 0
    buckets = []
    lenHuman, lenModel = [], []
    for i in range(30): buckets.append([0, 0, 0])
    for i in range(8):
        lenHuman.append([])
        lenModel.append([])

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

        world_num = _safe_numeric_suffix(world_name, default=0)
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
            if config.device is not None:
                init_g = init_g.to(config.device)
            graphSeq_t.append(init_g)

        # * ------------------------------------------------------------------------------------------
        # * Test model.
        y_pred_list = []
        while True:
            action_pred = agent.get_action(graphSeq_t[-1], goal2vec, epsilon=-1)

            # * !!!
            res, g, err = approx.execAction(config, action_pred, e)
            predActionSeq.append(action_pred)
            if g is not None and config.device is not None:
                g = g.to(config.device)
            graphSeq_t.append(g)

            # * If (res == False) and (err == '') and (len(actionSeq) < xxx): continue
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

    print('--' * 20)
    fmt = '{:^20}\t{:^15,.4f}\t{:^10}\t{:^10}'
    print('Action sequence length  |  Accuracy  |  Data num  |  Correct num')
    key_items = list(data_num.keys())

    # print('key_items:')
    # print(key_items)

    key_items.sort(key=lambda x: int(x))  # * Convert str to int.
    for key in key_items:
        value = data_num[key]
        data_accuracy = data_correct_num[key] / value * 100
        print(fmt.format(key, data_accuracy, value, data_correct_num[key]))
    print('--' * 20)

    den = correct + incorrect + error
    print('Correct num, incorrect num, error num: ', correct, incorrect, error)
    print('Correct, Incorrect, Error: ', (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den))
    return (correct * 100 / den), (incorrect * 100 / den), (error * 100 / den), lenHuman, lenModel


def accuracy_score_cql(config, dataset, agent, num_objects=0, TQDM=True):
    total_correct = 0
    denominator = 0

    data_container = tqdm(dataset, desc='Accuracy Score', ncols=80) if TQDM else dataset
    for (graphSeq, goal2vec, goal_json, actionSeq, action2vec, world_name, start_node) in data_container:
        with torch.no_grad():
            y_pred_list = []
            for i in range(len(graphSeq)):
                output = agent.get_action(graphSeq[i], goal2vec, epsilon=-1)
                y_pred_list.append(output)

        for i, y_pred in enumerate(y_pred_list):
            denominator += 1
            action_pred = y_pred
            if (action_pred == actionSeq[i]):
                total_correct += 1
            # if (len(action_pred['args']) == 2 and action_pred['args'][0] == action_pred['args'][1]):
            #     stuttering += 1

    return ((total_correct / denominator) * 100)
