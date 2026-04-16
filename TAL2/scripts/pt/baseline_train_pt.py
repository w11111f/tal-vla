# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File             : baseline_train_pt.py
@Project          : M3
@Time             : 2022/7/23 19:08
@Author           : Xianqi ZHANG
@Last Modify Time : 2022/8/20
@Version          : 1.0
@Desciption       : None
"""

import pickle
import warnings
import torch
import torch.nn as nn
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.tal.utils_training import get_model, load_model, save_model
from src.baselines.plan_transformer.utils import test_policy_pt

warnings.filterwarnings('ignore')


def backprop(config, optimizer, dataset, model, act_dim=111, accumulate_grad=True):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()
    for iter_num, (graphSeq, goal2vec, _, actionSeq, action2vec, _, _) in enumerate(dataset):
        if accumulate_grad:
            optimizer.zero_grad()
        graphSeq_t = []
        action2vec_t = []
        for i in range(len(graphSeq)):
            graphSeq_t.append(graphSeq[i])

            # * --------------------------------------------------------
            # * Preprocessing data.
            traj_len = len(graphSeq_t)
            if len(graphSeq_t) >= config.context_len:
                # sample random index to slice trajectory
                # # si = random.randint(0, traj_len - config.context_len)
                action_target = torch.stack(action2vec[i - config.context_len + 1:i + 1])
                states = graphSeq_t[-config.context_len:]
                actions = action2vec_t[-config.context_len:] if len(
                    action2vec_t) > config.context_len else action2vec_t
                start_t = max(0, traj_len - config.context_len)
                time_steps = torch.arange(start=start_t, end=start_t + config.context_len,
                                          step=1).to(config.device)
                # * All ones since no padding
                traj_mask = torch.ones(config.context_len, dtype=torch.long).to(config.device)

            else:  # * Less than context_len, will be padded with zeros in model.
                action_target = torch.stack(action2vec[:i + 1])
                padding_len = config.context_len - traj_len
                states = graphSeq_t
                actions = action2vec_t
                time_steps = torch.arange(start=0, end=config.context_len, step=1).to(
                    config.device)
                traj_mask = torch.cat(
                    [torch.ones(traj_len, dtype=torch.long),
                     torch.zeros(padding_len, dtype=torch.long)],
                    dim=0).to(config.device)

            # * --------------------------------------------------------
            # * Train model.
            action_preds = model(time_steps=time_steps, prompt_state=goal2vec, states=states,
                                 actions=actions)
            # action_preds = model(prompt_state=goal2vec, states=states, actions=actions)
            # * Only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]

            # * Only constrain the last output.
            action_loss = criterion(action_preds[-1], action2vec[i])
            # action_loss = criterion(action_preds, action_target)

            optimizer.zero_grad()
            action_loss.backward()
            optimizer.step()

            total_loss += action_loss.item()
            action2vec_t.append(action2vec[i])

    return total_loss


def backprop_batch(config, optimizer, dataset, model, act_dim=111, accumulate_grad=True,
                   batch_size=16):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()

    count = 0
    for iter_num, (graphSeq, goal2vec, _, actionSeq, action2vec, _, _) in enumerate(dataset):
        if accumulate_grad:
            optimizer.zero_grad()
        graphSeq_t = []
        action2vec_t = []
        for i in range(len(graphSeq)):
            graphSeq_t.append(graphSeq[i])

            # * --------------------------------------------------------
            # * Preprocessing data.
            traj_len = len(graphSeq_t)
            if len(graphSeq_t) >= config.context_len:
                # sample random index to slice trajectory
                # # si = random.randint(0, traj_len - config.context_len)
                action_target = torch.stack(action2vec[i - config.context_len + 1:i + 1])
                states = graphSeq_t[-config.context_len:]
                actions = action2vec_t[-config.context_len:] if len(
                    action2vec_t) > config.context_len else action2vec_t
                start_t = max(0, traj_len - config.context_len)
                time_steps = torch.arange(start=start_t, end=start_t + config.context_len,
                                          step=1).to(config.device)
                # * All ones since no padding
                traj_mask = torch.ones(config.context_len, dtype=torch.long).to(config.device)

            else:  # * Less than context_len, will be padded with zeros in model.
                action_target = torch.stack(action2vec[:i + 1])
                padding_len = config.context_len - traj_len
                states = graphSeq_t
                actions = action2vec_t
                time_steps = torch.arange(start=0, end=config.context_len, step=1).to(
                    config.device)
                traj_mask = torch.cat(
                    [torch.ones(traj_len, dtype=torch.long),
                     torch.zeros(padding_len, dtype=torch.long)],
                    dim=0).to(config.device)

            # * --------------------------------------------------------
            # * Train model.
            action_preds = model(time_steps=time_steps, prompt_state=goal2vec, states=states,
                                 actions=actions)
            # * Only consider non padded elements
            action_preds = action_preds.view(-1, act_dim)[traj_mask.view(-1, ) > 0]
            action_loss = criterion(action_preds, action_target)
            # * Only constrain the last output.
            # action_loss = criterion(action_preds[-1], action2vec[i])
            total_loss += action_loss.item()

            if batch_size is not None:
                action_loss = action_loss / batch_size
            action_loss.backward()
            count += 1

            if (count % batch_size == 0):
                optimizer.step()
                optimizer.zero_grad()
                count = 0

            action2vec_t.append(action2vec[i])

    return total_loss


if __name__ == '__main__':
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'PlanTransformer'
    args.num_epochs = 1000
    config = EnvironmentConfig(args)
    config.context_len = 3

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/2/home/'
    train_data_path = './data/2/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)
    val_data_path = './data/2/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

    # * ----------------------------------------------------------------
    model = get_model(config, config.model_name, config.features_dim, config.num_objects,
                      n_layers=3)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''

    lr = 1e-4
    model, optimizer, epoch, accuracy_list = load_model(config, seqTool + model.name + '_Trained',
                                                        model, lr=lr, strict=False)
    epoch = 0
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)
    model = model.to(config.device)

    policy_test_frequency = 10
    if config.exec_type == 'train':
        print('Training ' + model.name + ' with ' + config.embedding)
        for epoch_num in range(epoch + 1, config.NUM_EPOCHS):
            # scheduler.step(epoch=epoch_num)

            # * Update learning rate.
            if epoch_num in [500]:  # * 1e-4, 1e-5
                lr = lr / 10
                print('Change learning rate to {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            print('EPOCH {}, lr: {:.5f}'.format(epoch_num, optimizer.param_groups[0]['lr']))
            # loss = backprop(config, optimizer, train_dataset, model)
            loss = backprop_batch(config, optimizer, train_dataset, model, batch_size=4)
            print('Loss: {}'.format(loss))

            t2, t1 = 0, 0
            c, i, e = 0, 0, 0
            g_c, g_i, g_e = 0, 0, 0
            if (loss < 80) and ((epoch_num + 1) % policy_test_frequency == 0):
                # * Test policy accuracy.
                print('Val data policy test.')
                c, i, e, _, _ = test_policy_pt(config, val_dataset, model, config.num_objects,
                                               TQDM=False)
            accuracy_list.append((t2, t1, loss, c, i, e, g_c, g_i, g_e))

            # * Save model
            file_path = save_model(config, model, optimizer, epoch_num, accuracy_list)

        policy_acc = [i[3] for i in accuracy_list]
        print('The maximum policy on test set is ', str(max(policy_acc)), ' at epoch ',
              policy_acc.index(max(policy_acc)))

        results_save_path = './' + config.MODEL_SAVE_PATH + config.model_name + '_results.pkl'
        with open(results_save_path, 'wb') as f:
            pickle.dump(accuracy_list, f)
