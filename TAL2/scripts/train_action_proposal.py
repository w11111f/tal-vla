import time
import pickle
import colorama
import warnings
import torch
import torch.nn as nn
from torch import optim
from termcolor import cprint
from copy import deepcopy
from src.config.config import init_args
from src.utils.misc import setup_seed
from src.envs.CONSTANTS import EnvironmentConfig
from src.tal.utils_training import get_model, load_model, save_model
from src.tal.utils_training import test_policy_graph_dataset
from src.datasets.graph_dataset import GraphDataset_State

colorama.init()
warnings.filterwarnings('ignore')


def backprop(config, optimizer, dataset, model, argument=False):
    model.train()
    data_num = len(dataset)
    total_loss = 0.0
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    for iter_num, (graphSeq, goal2vec, _, actionSeq, action2vec, _, _) in enumerate(dataset):
        for i in range(len(graphSeq)):

            if argument:
                tmp_graph = deepcopy(graphSeq[i])
                tmp_goal = deepcopy(goal2vec)
                # noise = (torch.rand_like(tmp_graph.ndata['feat']) * 2 - 1) * 0.1  # * [-0.1, 0.1)
                # tmp_graph.ndata['feat'] += noise
                mask = torch.rand_like(tmp_graph.ndata['feat']) < 0.9
                tmp_graph.ndata['feat'] *= mask
                tmp_goal.ndata['feat'] *= mask
            else:
                tmp_graph = graphSeq[i]
                tmp_goal = goal2vec

            # y_pred = model(tmp_graph, goal2vec)
            y_pred = model(tmp_graph, tmp_goal)
            y_true = action2vec[i]

            loss = criterion(y_pred, y_true)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # return (total_loss / data_num)
    return total_loss


def backprop_batch(config, optimizer, dataset, model, batch_size=4, argument=False):
    model.train()
    total_loss = 0.0
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    train_data_num = 0
    need_backward = False
    for iter_num, (graphSeq, goal2vec, _, actionSeq, action2vec, _, _) in enumerate(dataset):
        # * Use graphSeq and goal2vec.
        for i in range(len(graphSeq)):
            train_data_num += 1

            # * Add mask.
            if argument:
                tmp_graph = deepcopy(graphSeq[i])
                tmp_goal = deepcopy(goal2vec)

                noise = (torch.rand_like(tmp_graph.ndata['feat']) * 2 - 1) * 0.2  # * [-0.2, 0.2)
                tmp_graph.ndata['feat'] += noise
                # mask = torch.rand_like(tmp_graph.ndata['feat']) < 0.9
                # tmp_graph.ndata['feat'] *= mask
                # tmp_goal.ndata['feat'] *= mask
            else:
                tmp_graph = graphSeq[i]
                tmp_goal = goal2vec

            # y_pred = model(tmp_graph, goal2vec)
            y_pred = model(tmp_graph, tmp_goal)
            y_true = action2vec[i]

            tmp_loss = criterion(y_pred, y_true)
            total_loss += tmp_loss.item()

            tmp_loss = tmp_loss / batch_size
            tmp_loss.backward()
            need_backward = True

            if (train_data_num % batch_size == 0):
                need_backward = False
                optimizer.step()
                optimizer.zero_grad()

    if need_backward:
        # * Final data.
        optimizer.step()
        optimizer.zero_grad()

    return total_loss


if __name__ == '__main__':
    rnd_seed = 1
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Keep APN training/eval lightweight by avoiding Isaac Lab resets during periodic policy checks.
    args.policy_backend = 'symbolic'
    args.model_name = 'APN'
    args.num_epochs = 800
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    train_dataset = GraphDataset_State(config, graphs_dir, train_data_path)

    val_data_path = './data/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

    # test_data_path = './new_dataset/test_dataset_new.pkl'
    # test_dataset = GraphDataset_State(config, graphs_dir, test_data_path)

    train_data_num = len(train_dataset)
    val_data_num = len(val_dataset)
    # test_data_num = len(test_dataset)

    cprint('Train data num: {}'.format(train_data_num), 'green')
    cprint('Val data num: {}'.format(val_data_num), 'green')
    # cprint('Test data num: {}'.format(test_data_num), 'green')

    # * ----------------------------------------------------------------
    model = get_model(config, config.model_name, config.features_dim, config.num_objects)
    seqTool = 'Seq_' if config.training == 'gcn_seq' else ''
    # lr = 0.0005
    # lr = 1e-4
    lr = 5e-4
    model, optimizer, epoch, accuracy_list = load_model(config, seqTool + model.name + '_Trained',
                                                        model, lr=lr)
    model = model.to(config.device)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.9)

    test_frequency = 20
    if config.exec_type == 'train':
        print('Training ' + model.name + ' with ' + config.embedding)
        for epoch_num in range(epoch + 1, config.NUM_EPOCHS + 1):
            if epoch_num > 300:
                test_frequency = 10
            # if epoch_num > 700:
            #     test_frequency = 5
            scheduler.step(epoch=epoch_num)
            print('EPOCH {}, lr: {}'.format(epoch_num, optimizer.param_groups[0]['lr']))
            start_time = time.time()
            # loss = backprop(config, optimizer, train_dataset, model, argument=False)
            loss = backprop_batch(config, optimizer, train_dataset, model, argument=False)

            end_time = time.time()
            cprint('Time per epoch: {}'.format(end_time - start_time), 'green')
            print('Loss: {}'.format(loss))

            # # * ------------------------------------------------------
            # # * Update learning rate.
            # if epoch_num in [30, 100, 200]:  # * 1e-4, 1e-5
            #     lr = lr / 10
            #     print('Change learning rate to {}'.format(lr))
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr

            # * --------------------------------------------------------
            # t2, t1, loss = 0, 0, 0
            c, i, e = 0, 0, 0
            if (loss < 100) and ((epoch_num + 1) % test_frequency == 0):
                # * Test action predict accuracy.
                # t1 = accuracy_score_graph_dataset_state(config, train_dataset, model, config.num_objects)
                # print('Train action accuracy: {}'.format(t1))
                # t2 = accuracy_score_graph_dataset_state(config, val_dataset, model, config.num_objects)
                # print('Val action accuracy: {}'.format(t2))

                # * Test policy accuracy.
                cprint('Val data policy test.', 'green')
                c, i, e, _, _ = test_policy_graph_dataset(config, val_dataset, model,
                                                          config.num_objects, TQDM=False)
            # accuracy_list.append((t2, t1, loss, c, i, e))
            accuracy_list.append((0, 0, 0, c, i, e))

            # * Save model
            target_path = f"{config.MODEL_SAVE_PATH}/{seqTool}{model.name}_Trained.ckpt"
            file_path = save_model(config, model, optimizer, epoch_num, accuracy_list, file_path=target_path)

        print('The maximum accuracy on test set is ', str(max(accuracy_list)), ' at epoch ',
              accuracy_list.index(max(accuracy_list)))

        policy_acc = [i[3] for i in accuracy_list]
        print('The maximum policy on test set is ', str(max(policy_acc)), ' at epoch ',
              policy_acc.index(max(policy_acc)))

        results_save_path = './' + config.MODEL_SAVE_PATH + 'results.pkl'
        with open(results_save_path, 'wb') as f:
            pickle.dump(accuracy_list, f)
