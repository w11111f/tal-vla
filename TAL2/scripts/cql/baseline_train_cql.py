import pickle
import torch
import numpy as np
from src.utils.misc import setup_seed
from src.config.config import init_args
from src.tal.utils_training import save_model
from src.envs.CONSTANTS import EnvironmentConfig
from src.datasets.graph_dataset import GraphDataset_State
from src.baselines.cql_dqn.buffer import ReplayBuffer
from src.baselines.cql_dqn.agent import CQLAgent
from src.baselines.cql_dqn.utils import test_policy_graph_dataset_cql

if __name__ == '__main__':
    rnd_seed = 0
    setup_seed(seed=rnd_seed)
    print('==' * 10)
    print('Set random seed = {}'.format(rnd_seed))
    print('==' * 10)

    args = init_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.model_name = 'DDQN'
    args.num_epochs = 3000
    config = EnvironmentConfig(args)

    # * ----------------------------------------------------------------
    # * Load data.
    with open('checkpoints/home/action_list_train_dataset.pkl', 'rb') as f:
        action_set = pickle.load(f)

    graphs_dir = './data/home/'
    train_data_path = './data/train_dataset.pkl'
    buffer = ReplayBuffer(config, graphs_dir, train_data_path)
    val_data_path = './data/val_dataset.pkl'
    val_dataset = GraphDataset_State(config, graphs_dir, val_data_path)

    # * ----------------------------------------------------------------
    # * Train
    agent = CQLAgent(config, action_set)
    accuracy_list = []
    history_loss = []
    # for epoch_num in range(agent.epoch + 1, config.NUM_EPOCHS):

    start_epoch = 2000
    for epoch_num in range(start_epoch, config.NUM_EPOCHS):
        print('Epoch: {}'.format(epoch_num))
        # * New !!!
        # agent.scheduler.step(epoch=epoch_num)
        agent.scheduler.step(epoch=epoch_num - 1000)

        # * Update target network.
        if (epoch_num + 1) % 20 == 0:
            print('Update target network...')
            agent.soft_update(tau=0.01)  # * Hard update.

        total_loss = 0.0
        for iter_num, experiences in enumerate(buffer):
            loss, cql_loss, bellman_error = agent.learn(experiences, epoch_num)

            total_loss = total_loss + loss
        print('Total Loss: {} '.format(total_loss))
        history_loss.append(total_loss)

        if epoch_num % 1 == 0:
            save_model(config, agent.network, None, epoch_num, None)
            save_model(config, agent.target_net, None, epoch_num, None, target_net=True)

        # * ------------------------------------------------------------
        # * Test
        # if (total_loss < 100) and ((epoch_num + 1) % 10 == 0):
        test_freq = 50
        # if epoch_num > 600:
        #     test_freq = 20
        # elif epoch_num > 800:
        #     test_freq = 10

        c, i, e = 0, 0, 0
        if ((total_loss < 50) and (epoch_num + 1) % test_freq == 0):
            print('Policy Test: val_dataset...')
            c, i, e, _, _ = test_policy_graph_dataset_cql(config, val_dataset, agent, TQDM=False)
        accuracy_list.append((total_loss, c, i, e))

    if len(accuracy_list) != 0:
        policy_acc = [i[1] for i in accuracy_list]
        print('The maximum policy on val set is ', str(max(policy_acc)), ' at epoch ',
              policy_acc.index(max(policy_acc)))
    print('The lowest loss is ', str(min(history_loss)), ' at epoch ', np.argmin(history_loss))

    results_save_path = './' + config.MODEL_SAVE_PATH + 'results_cql.pkl'
    with open(results_save_path, 'wb') as f:
        pickle.dump(accuracy_list, f)
